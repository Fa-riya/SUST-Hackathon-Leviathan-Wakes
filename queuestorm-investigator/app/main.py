"""
main.py
-------
FastAPI service for the QueueStorm Investigator.

Endpoints:
    GET  /health          -> {"status": "ok"}
    POST /analyze-ticket  -> structured analysis (Section 6 schema)

Status-code contract (Section 4.1):
    200  valid request, schema-conformant body
    400  malformed JSON or missing/invalid required fields
    422  schema valid but semantically invalid (empty complaint)
    500  internal error (safe message only; never a stack trace or secret)

The service binds 0.0.0.0:$PORT (default 8000) and never crashes on bad input.
"""

from __future__ import annotations

import logging
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .llm import llm_enabled, maybe_polish_reply
from .models import AnalyzeRequest, AnalyzeResponse
from .reasoning import analyze
from .utils import detect_language

logger = logging.getLogger("queuestorm")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

app = FastAPI(
    title="QueueStorm Investigator",
    description="Evidence-grounded support copilot for digital finance complaints.",
    version="1.0.0",
)


def _error(status: int, message: str) -> JSONResponse:
    """Uniform, non-sensitive error envelope."""
    return JSONResponse(status_code=status, content={"error": message})


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze-ticket")
async def analyze_ticket(request: Request):
    # ---- 1. Parse JSON (malformed -> 400) -------------------------------
    try:
        payload = await request.json()
    except Exception:
        return _error(400, "Malformed JSON body.")

    if not isinstance(payload, dict):
        return _error(400, "Request body must be a JSON object.")

    # ---- 2. Required-field checks (-> 400) ------------------------------
    ticket_id = payload.get("ticket_id")
    complaint = payload.get("complaint")
    if not isinstance(ticket_id, str) or ticket_id.strip() == "":
        return _error(400, "Missing or invalid required field: ticket_id.")
    if "complaint" not in payload or not isinstance(complaint, str):
        return _error(400, "Missing or invalid required field: complaint.")

    # ---- 3. Semantic validation (empty complaint -> 422) ----------------
    if complaint.strip() == "":
        return _error(422, "Field 'complaint' must not be empty.")

    # ---- 4. Build a lenient request model (type issues -> 400) ----------
    try:
        req = AnalyzeRequest.model_validate(payload)
    except ValidationError:
        # Retry with optional fields dropped so a single bad optional value
        # cannot fail an otherwise valid request.
        safe_payload = {"ticket_id": ticket_id, "complaint": complaint}
        for key in ("language", "channel", "user_type", "campaign_context", "metadata"):
            if key in payload:
                safe_payload[key] = payload[key]
        th = payload.get("transaction_history")
        if isinstance(th, list):
            safe_payload["transaction_history"] = th
        try:
            req = AnalyzeRequest.model_validate(safe_payload)
        except ValidationError:
            req = AnalyzeRequest(ticket_id=ticket_id, complaint=complaint)

    # ---- 5. Investigate (any unexpected error -> safe 500) --------------
    try:
        result = analyze(req)

        # Optional LLM polish of the (already safe) customer reply.
        if llm_enabled():
            language = detect_language(req.complaint, req.language)
            result["customer_reply"] = maybe_polish_reply(
                result["customer_reply"], language
            )

        # Final schema enforcement: build through the strict response model so
        # we can never emit an out-of-vocabulary enum or extra field.
        validated = AnalyzeResponse(**result)
        return JSONResponse(status_code=200, content=validated.model_dump())
    except Exception as exc:  # noqa: BLE001 - we deliberately swallow + log
        logger.exception("analysis_failed ticket_id=%s", payload.get("ticket_id"))
        _ = exc  # avoid leaking details to the client
        return _error(500, "Internal error while analyzing the ticket.")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
