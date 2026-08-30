"""
adapters.py
------------
This is the ONLY file that needs to know every module's actual, individual
return shape. Nobody's module was rewritten to fit a shared schema —
instead, this file translates each one's native output into what
risk_engine.py and audit_module.py actually expect, and into what the
frontend's generateMockResults() shape (see App.jsx) already promises the
UI. main.py calls these adapters; it never touches a module's raw output
directly.

============================================================================
TWO REAL GAPS, STUBBED HONESTLY (fix before final submission)
============================================================================
1. LIVENESS — modules/liveness_module.py is empty. Nobody has built
   blink-detection or rPPG capture yet, but risk_engine.py already
   HARD-GATES on blink_score < 40. Defaulting blink_score to 0 would
   reject every single verification, which is worse than not gating at
   all. Instead this defaults to blink_score=100 (i.e. "assume live,
   don't gate") — WRONG in the sense that it doesn't actually check
   anything, but it's an honest, visible placeholder rather than a
   silent one. Search this file for "LIVENESS STUB" to find and replace
   it once the real module exists.

2. IDENTITY GRAPH — no module tracks "same face embedding under a
   different name/ID" (risk_engine.py's identity_graph_flagged hard
   gate). Always defaults to False (never flags). Same caveat: this is
   an absence-of-checking default, not a real check.

============================================================================
THE OCR -> VALIDATION FIELD MAPPING (a real, unavoidable heuristic)
============================================================================
ocr_module.py's extract_fields() returns {"ID", "Date_1", "Date_2", "Name"}.
validation_module.py expects {"id_number", "dob", "expiry_date", "name"}.
There is no way to know from OCR alone which of Date_1/Date_2 is a birth
date vs. an expiry date — this maps EARLIER date -> dob, LATER date ->
expiry_date, which holds for every normal ID document (you're born before
your ID expires) but hasn't been validated against your team's real
sample documents yet. Worth checking once you have real scanned samples.
"""

from __future__ import annotations

import logging
from datetime import datetime
from types import SimpleNamespace
from typing import Optional

logger = logging.getLogger("adapters")


# ---------------------------------------------------------------------------
# OCR -> field dict for the Validation module
# ---------------------------------------------------------------------------

def ocr_fields_to_validation_fields(ocr_raw: dict) -> dict:
    """
    ocr_raw is process_document()'s return dict (has "fields": {"ID":...,
    "Date_1":..., "Date_2":..., "Name":...}). Converts to the named-field
    dict validation_module.py's DocumentValidator.run_all() expects.
    """
    raw_fields = ocr_raw.get("fields", {})
    fields = {}

    if "Name" in raw_fields:
        fields["name"] = raw_fields["Name"]
    if "ID" in raw_fields:
        fields["id_number"] = raw_fields["ID"]

    date_1 = raw_fields.get("Date_1")
    date_2 = raw_fields.get("Date_2")
    dob, expiry = _order_dates_as_dob_expiry(date_1, date_2)
    if dob:
        fields["dob"] = dob
    if expiry:
        fields["expiry_date"] = expiry

    return fields


def _order_dates_as_dob_expiry(date_1: Optional[str], date_2: Optional[str]):
    """Earlier date -> dob, later date -> expiry_date. See module docstring."""
    candidates = [d for d in (date_1, date_2) if d]
    parsed = []
    for d in candidates:
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
            try:
                parsed.append((datetime.strptime(d, fmt), d))
                break
            except ValueError:
                continue

    if len(parsed) < 2:
        # Only one date found (or none parseable) — can't order them, so
        # just guess it's the expiry (more commonly the field that's
        # present alone on things like visas/permits).
        return None, (candidates[0] if candidates else None)

    parsed.sort(key=lambda p: p[0])
    return parsed[0][1], parsed[1][1]  # (earlier=dob, later=expiry)


# ---------------------------------------------------------------------------
# Validation module -> risk_engine.py's checksum_valid / qr_code_match / etc.
# ---------------------------------------------------------------------------

def validation_result_to_risk_fields(validation_raw: dict) -> dict:
    """
    validation_raw is DocumentValidator.run_all()'s return dict. Pulls the
    checksum check OUT separately (risk_engine.py's ADAPTER NOTE: never use
    overall_score, which still averages checksum in with everything else).
    """
    checks_by_name = {c["name"]: c for c in validation_raw["checks"]}

    def passed_or_none(name: str) -> Optional[bool]:
        c = checks_by_name.get(name)
        if not c or not c["applicable"]:
            return None
        return bool(c["passed"])

    return {
        "checksum_valid": passed_or_none("checksum"),
        "qr_code_match": passed_or_none("qr_match"),
        "expiry_date_valid": passed_or_none("expiry"),
        "field_format_valid": passed_or_none("format"),
    }


# ---------------------------------------------------------------------------
# Tampering module -> risk_engine.py's tamper_ela / tamper_fft / tamper_cnn
# ---------------------------------------------------------------------------

def tampering_result_to_risk_fields(tampering_result) -> dict:
    """
    tampering_result is a TamperingResult dataclass from tampering_module.py.
    Its ela_anomaly_score/fft_spike_score are 0-1 (higher=suspicious);
    risk_engine.py wants 0-100 on the same higher=suspicious convention —
    just a scale conversion, no direction flip needed. tamper_cnn stays
    None (no CNN model exists yet — see that module's own docstring on
    this being an intentional, disclosed limitation).
    """
    return {
        "tamper_ela": round(tampering_result.ela_anomaly_score * 100, 2),
        "tamper_fft": round(tampering_result.fft_spike_score * 100, 2),
        "tamper_cnn": None,
    }


# ---------------------------------------------------------------------------
# Face verification -> risk_engine.py's face_similarity / face_distance
# ---------------------------------------------------------------------------

def face_result_to_risk_fields(face_bundle: dict) -> dict:
    """face_bundle is verify_face_with_cross_check()'s return dict."""
    primary = face_bundle["primary"]
    return {
        "face_similarity": round(primary.match_score * 100, 2),
        "face_distance": primary.distance,
    }


# ---------------------------------------------------------------------------
# LIVENESS STUB — replace once liveness_module.py exists
# ---------------------------------------------------------------------------

def liveness_stub_risk_fields() -> dict:
    """
    See this file's module docstring, section 1. This does NOT check
    anything — it exists only so risk_engine.py's hard gate doesn't reject
    every single request while no liveness module exists yet.
    """
    logger.warning(
        "liveness_module.py is empty — using non-gating placeholder "
        "(blink_score=100). This does not actually verify liveness."
    )
    return {"blink_score": 100.0, "pulse_detected": False, "bpm": None}


# ---------------------------------------------------------------------------
# Audit module — everything needs a .score / .state / .detail duck type
# ---------------------------------------------------------------------------

def build_audit_view(ocr_raw: dict, validation_raw: dict, tampering_result, face_bundle: dict):
    """Returns the four SimpleNamespace objects audit_module.log_verification() needs."""
    primary = face_bundle["primary"]

    ocr_view = SimpleNamespace(
        score=ocr_raw["ocr_confidence"],
        state="OK" if ocr_raw["ocr_confidence"] >= 60 else "LOW_CONFIDENCE",
        detail=f"EasyOCR/Tesseract agreement: {ocr_raw['ocr_agreement']:.0f}%",
    )

    validation_view = SimpleNamespace(
        score=validation_raw["overall_score"],
        state=f"{validation_raw['checks_run']} checks run, {validation_raw['checks_skipped']} skipped",
        detail="; ".join(c["details"] for c in validation_raw["checks"] if c["applicable"]) or "No applicable checks",
    )

    tampering_view = SimpleNamespace(
        score=tampering_result.tampering_score,  # 0-100, higher=genuine
        state="GENUINE" if tampering_result.tampering_score >= 70 else "SUSPICIOUS",
        detail=tampering_result.explanation,
    )

    face_view = SimpleNamespace(
        score=round(primary.match_score * 100, 2),
        state="MATCH" if primary.match else "NO_MATCH",
        detail=f"distance={primary.distance:.3f}, is_real={primary.is_real}",
    )

    return ocr_view, validation_view, tampering_view, face_view


# ---------------------------------------------------------------------------
# Frontend response shaping — matches generateMockResults() in App.jsx
# exactly, so the frontend's fetch() can drop straight in for the mock.
# ---------------------------------------------------------------------------

def build_frontend_response(
    ocr_raw: dict,
    validation_raw: dict,
    tampering_result,
    face_bundle: dict,
    liveness_fields: dict,
    risk_result,
    audit_entry: dict,
) -> dict:
    primary = face_bundle["primary"]
    checks_by_name = {c["name"]: c for c in validation_raw["checks"]}

    def check_bool(name: str) -> bool:
        # Frontend's validation card is boolean-only (no "N/A" state) —
        # an inapplicable check is treated as passed (no evidence against
        # it), matching risk_engine.py's own neutral-0.5 treatment.
        c = checks_by_name.get(name)
        if not c or c["passed"] is None:
            return True
        return bool(c["passed"])

    # risk_engine.py's RiskResult.score is 0-1, HIGHER = more risk.
    # Frontend wants 0-100, HIGHER = more trusted. Documented conversion
    # from risk_engine.py's own ADAPTER NOTE — done in exactly this one place.
    frontend_score = round((1 - risk_result.score) * 100)
    if frontend_score >= 80:
        verdict = "GENUINE"
    elif frontend_score >= 50:
        verdict = "SUSPICIOUS"
    else:
        verdict = "FAKE"

    ocr_display = {
        **{k: v for k, v in ocr_raw.get("fields", {}).items()},
        "OCR Confidence": f"{ocr_raw['ocr_confidence']:.0f}%",
    }

    return {
        "ocr": ocr_display,
        "validation": {
            "Checksum Verification": check_bool("checksum"),
            "QR Code Match": check_bool("qr_match"),
            "Expiry Date Valid": check_bool("expiry"),
            "Field Format (Regex)": check_bool("format"),
        },
        "tampering": {
            "ela": round(tampering_result.ela_anomaly_score * 100),
            "fft": round(tampering_result.fft_spike_score * 100),
            "cnn": 0,  # no CNN model yet — shown as 0 suspicion, not hidden
        },
        "face": {
            "similarity": round(primary.match_score * 100),
            "distance": primary.distance,
        },
        "liveness": {
            "pulseDetected": liveness_fields["pulse_detected"],
            "bpm": liveness_fields["bpm"],
            "blinkScore": liveness_fields["blink_score"],
        },
        "riskScore": frontend_score,
        "verdict": verdict,
        "auditEntry": audit_entry,
        # Extras beyond the mock shape — harmless for the frontend to ignore,
        # useful for debugging or a future "why" panel:
        "hardGated": risk_result.hard_gated,
        "hardGateReason": risk_result.hard_gate_reason,
        "reasons": risk_result.reasons,
    }
