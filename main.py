"""
main.py — Backend API Orchestrator
------------------------------------
Wires together everyone's actual modules (as committed in this repo) into
one FastAPI endpoint the frontend can call in place of generateMockResults()
in App.jsx.

Flow:
  1. Accept a document image + doc_type + a captured selfie photo.
  2. Run OCR on the document.
  3. Map OCR's raw fields into the named fields the Validation module wants
     (see adapters.py — this mapping is a documented heuristic).
  4. Run Document Validation, Tampering Detection, and Face Verification.
  5. LIVENESS: no real module exists yet (modules/liveness_module.py is
     empty) — uses a clearly-flagged non-gating placeholder. See
     adapters.py's module docstring before your final submission.
  6. Feed everything into risk_engine.py's hard-gate + weighted-fusion logic.
  7. Log to the audit trail (never blocks the response if logging fails).
  8. Return JSON shaped exactly like generateMockResults() in App.jsx, so
     the frontend's fetch() can replace the mock call directly.

Run with:  uvicorn main:app --reload
"""

import logging
import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from modules.ocr_module import process_document
from modules.validation_module import DocumentValidator
from modules.tampering_module import analyze_tampering
from modules.face_verification import verify_face_with_cross_check
from modules.risk_engine import run_risk_engine, RiskEngineInput
from modules.audit_module import compute_document_hash, log_verification, get_recent_entries

import adapters

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app = FastAPI(title="Smart India Hackathon — Fake Document Detection")

# Frontend runs on a different dev-server port (Vite) than the API —
# CORS needs to be open for local development. Tighten allow_origins
# before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/verify")
async def verify(
    document_image: UploadFile = File(...),
    selfie_photo: UploadFile = File(...),
    doc_type: str = Form(...),  # "passport" | "visa" | "national_id" | "driving_license" | "permit"
):
    with tempfile.TemporaryDirectory() as tmp:
        doc_path = _save_upload(document_image, tmp, "document.jpg")
        selfie_path = _save_upload(selfie_photo, tmp, "selfie.jpg")

        # --- 1. OCR ---
        try:
            ocr_raw = process_document(doc_path)
        except Exception as exc:
            logger.error(f"OCR failed: {exc}")
            return JSONResponse(status_code=422, content={"error": f"Could not process document image: {exc}"})

        # --- 2. Validation (using fields mapped from OCR) ---
        validation_fields = adapters.ocr_fields_to_validation_fields(ocr_raw)
        validator = DocumentValidator()
        validation_raw = validator.run_all(
            doc_type=doc_type, fields=validation_fields, document_image_path=doc_path
        )

        # --- 3. Tampering ---
        try:
            tampering_result = analyze_tampering(doc_path)
        except ValueError as exc:
            logger.error(f"Tampering analysis failed: {exc}")
            return JSONResponse(status_code=422, content={"error": f"Could not analyze document image: {exc}"})

        # --- 4. Face verification ---
        face_bundle = verify_face_with_cross_check(doc_path, selfie_path)

        # --- 5. Liveness (STUB — see adapters.py docstring) ---
        liveness_fields = adapters.liveness_stub_risk_fields()

        # --- 6. Risk Engine ---
        risk_input = RiskEngineInput(
            document_type=doc_type,
            ocr_confidence=ocr_raw["ocr_confidence"],
            **adapters.validation_result_to_risk_fields(validation_raw),
            **adapters.tampering_result_to_risk_fields(tampering_result),
            **adapters.face_result_to_risk_fields(face_bundle),
            blink_score=liveness_fields["blink_score"],
            pulse_detected=liveness_fields["pulse_detected"],
            bpm=liveness_fields["bpm"],
            identity_graph_flagged=False,  # no identity-graph module exists yet
        )
        risk_result = run_risk_engine(risk_input)

        # --- 7. Audit log (never blocks the response) ---
        audit_entry = _log_audit_safely(
            doc_path, ocr_raw, validation_raw, tampering_result, face_bundle, risk_result
        )

        # --- 8. Response, shaped for the existing frontend ---
        response = adapters.build_frontend_response(
            ocr_raw, validation_raw, tampering_result, face_bundle,
            liveness_fields, risk_result, audit_entry,
        )
        return JSONResponse(response)


def _save_upload(upload: UploadFile, tmp_dir: str, filename: str) -> str:
    dest = Path(tmp_dir) / filename
    with open(dest, "wb") as f:
        shutil.copyfileobj(upload.file, f)
    return str(dest)


def _log_audit_safely(doc_path, ocr_raw, validation_raw, tampering_result, face_bundle, risk_result) -> dict:
    """
    Wraps audit logging in try/except per audit_module.py's own docstring
    guidance — a logging failure (disk full, locked db) must never break
    the verification response the officer is waiting on.

    frontend_score/verdict are recomputed here too (cheap, and keeps this
    function self-contained) rather than threading them in from the caller.
    """
    frontend_score = round((1 - risk_result.score) * 100)
    verdict = "GENUINE" if frontend_score >= 80 else "SUSPICIOUS" if frontend_score >= 50 else "FAKE"

    try:
        doc_hash = compute_document_hash(doc_path)
        ocr_view, validation_view, tampering_view, face_view = adapters.build_audit_view(
            ocr_raw, validation_raw, tampering_result, face_bundle
        )
        log_verification(
            document_hash=doc_hash,
            ocr=ocr_view, validation=validation_view,
            tampering=tampering_view, face=face_view,
            risk_score=frontend_score, verdict=verdict,
        )
        recent = get_recent_entries(limit=1)
        entry = recent[0] if recent else {}
        return {
            "logId": f"LOG-{entry.get('entry_hash', '000000')[:6].upper()}",
            "timestamp": entry.get("timestamp", ""),
            "docHash": doc_hash,
            "prevHash": entry.get("prev_hash"),
            "currHash": entry.get("entry_hash"),
            "verdict": verdict,
        }
    except Exception as exc:
        logger.error(f"Audit logging failed (non-fatal): {exc}")
        return {
            "logId": "LOG-ERROR",
            "timestamp": "",
            "docHash": None,
            "prevHash": None,
            "currHash": None,
            "verdict": verdict,
        }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/audit-log")
async def audit_log(limit: int = 20):
    return get_recent_entries(limit=limit)
