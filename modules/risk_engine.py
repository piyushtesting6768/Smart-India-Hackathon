"""
risk_engine.py
--------------
Core risk-scoring logic for SIH PS188 (AI-Based Fake Identity & Document
Screening System).

Design (two layers, deliberately NOT one weighted average over everything):

  1. HARD GATES — near-binary integrity failures that short-circuit
     straight to RED without being diluted into an average. A checksum is
     either valid or it isn't; a face either matches or it doesn't.

  2. WEIGHTED FUSION — used ONLY for genuinely probabilistic soft signals
     (tamper sub-scores, OCR/validation confidence, borderline face/liveness
     signal). Weights are documented, not guessed.

No ML training step here by design — no labeled data, no time budget for
it in a hackathon window. Pure rule-based fusion logic.

----------------------------------------------------------------------------
CHANGELOG (v2 — folds in issues found reviewing the real teammate modules)
----------------------------------------------------------------------------
1. CHECKSUM IS NOW DOC-TYPE-AGNOSTIC AND ALWAYS A HARD GATE.
   The real validation module computes a real checksum for BOTH passports
   (MRZ check digit) and national IDs (Aadhaar Verhoeff) — not just
   passports/visas as v1 assumed. `checksum_valid` (renamed from
   `mrz_checksum_valid`) now hard-gates for ANY document type when it's
   populated. The adapter must pull this value OUT of the validation
   module's `checks` list itself — never trust `overall_score` from that
   module directly, since it currently still averages checksum in with
   the other three checks.

2. TAMPER FUSION NOW RENORMALIZES WHEN NO REAL CNN SCORE EXISTS.
   The real tampering module doesn't produce a CNN score (openly, by
   design — no GPU/time/dataset for it). If a missing CNN score defaulted
   to 0 ("no suspicion"), it would silently absorb 35% of the tamper
   fusion weight with a phantom "all clear" signal, diluting real ELA/FFT
   evidence on every single case. `tamper_cnn` is now Optional[float]:
   when None, weights are redistributed proportionally across ela/fft only.

3. "ZERO CHECKS APPLICABLE" IS EXPLICITLY DISTINCT FROM "ALL CHECKS FAILED".
   Carried over from v1 and confirmed necessary: `_validation_confidence`
   returns a neutral 0.5 when no soft validation checks apply to this
   document type (missing information), never the same 0.0 you'd get from
   every check actually failing (real evidence of a problem).
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

class RiskBand(str, Enum):
    GREEN = "GREEN"    # genuine
    YELLOW = "YELLOW"  # suspicious — needs human review
    RED = "RED"        # fake / hard-gated


@dataclass
class RiskResult:
    band: RiskBand
    score: float                          # 0.0 (clean) .. 1.0 (max risk), INTERNAL scale
    reasons: List[str] = field(default_factory=list)
    hard_gated: bool = False
    hard_gate_reason: Optional[str] = None

    # ADAPTER NOTE: frontend's verdictFromScore() (confirmed from the real
    # frontend source) is score>=80 GENUINE / >=50 SUSPICIOUS / else FAKE,
    # on a 0-100 scale where HIGHER = more trusted. This engine's `score`
    # is 0-1 where HIGHER = more risk (opposite direction). Convert with:
    #     frontend_score = round((1 - score) * 100)
    #     frontend_verdict = "GENUINE" if frontend_score >= 80 else \
    #                         "SUSPICIOUS" if frontend_score >= 50 else "FAKE"
    # Do this conversion in exactly ONE place in the adapter layer.


# ---------------------------------------------------------------------------
# Config: thresholds & weights — documented, not guessed
# ---------------------------------------------------------------------------

FACE_MATCH_HARD_GATE_THRESHOLD = 50.0          # face_similarity, 0-100
BLINK_LIVENESS_HARD_GATE_THRESHOLD = 40.0      # blink_score, 0-100, Tier 1 only

RED_THRESHOLD = 0.5
YELLOW_THRESHOLD = 0.2

# Tamper sub-fusion weights when ALL THREE signals are present.
# - ELA: most general-purpose signal for splice/paste edits -> highest weight.
# - CNN: useful secondary signal, but only as good as a hackathon-scope
#   model/validation set -> weighted below ELA.
# - FFT: documented as weaker against diffusion-model-generated forgeries,
#   which don't leave the same periodic frequency artifacts as classic
#   splice/copy-move edits -> lowest weight of the three.
TAMPER_WEIGHTS_FULL = {
    "ela": 0.45,
    "cnn": 0.35,
    "fft": 0.20,
}

# Used when tamper_cnn is None (current reality: no CNN model exists yet).
# Same relative ordering (ELA > FFT), renormalized to sum to 1.0 over the
# two signals that actually exist, so tampering keeps its full fusion
# weight instead of silently losing over a third of it to a phantom "clean"
# CNN default.
_ela_fft_sum = TAMPER_WEIGHTS_FULL["ela"] + TAMPER_WEIGHTS_FULL["fft"]
TAMPER_WEIGHTS_NO_CNN = {
    "ela": TAMPER_WEIGHTS_FULL["ela"] / _ela_fft_sum,
    "fft": TAMPER_WEIGHTS_FULL["fft"] / _ela_fft_sum,
}

# Top-level fusion weights across signal GROUPS (soft signals only — any
# signal capable of hard-gating, i.e. checksum/face-match/liveness/identity-
# graph, is excluded entirely from this table).
GROUP_WEIGHTS = {
    "ocr_confidence": 0.15,
    "validation_confidence": 0.20,
    "tampering": 0.30,
    "face_soft": 0.15,
    "liveness_soft": 0.20,
}


# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

@dataclass
class RiskEngineInput:
    document_type: str  # "passport" | "visa" | "national_id" | "driving_license" | "permit"

    # OCR
    ocr_confidence: float = 0.0                    # 0-100

    # Validation — populate ONLY fields that apply to this document_type,
    # matching the validation module's own `applicable` flags.
    #
    # checksum_valid: pull this OUT of the validation module's `checks`
    # list yourself (the check named "checksum") — do NOT use that
    # module's `overall_score`, which still averages checksum in with
    # everything else. None = no checksum scheme exists for this doc
    # type/field combo (module returned applicable=False) — NOT hard-gated,
    # just excluded, same as any other inapplicable check.
    checksum_valid: Optional[bool] = None           # ANY doc type -> HARD GATE if False
    qr_code_match: Optional[bool] = None
    expiry_date_valid: Optional[bool] = None
    field_format_valid: Optional[bool] = None

    # Tampering (0-100, higher = MORE suspicious).
    # tamper_cnn = None means "no CNN score exists" (current reality) and
    # triggers ela/fft-only renormalized fusion. Do NOT default this to 0.0
    # to mean "no CNN" — 0.0 means "CNN ran and found nothing suspicious",
    # a materially different claim.
    tamper_ela: float = 0.0
    tamper_fft: float = 0.0
    tamper_cnn: Optional[float] = None

    # Face verification
    face_similarity: float = 0.0                    # 0-100
    face_distance: Optional[float] = None            # informational only

    # Liveness
    blink_score: float = 0.0                         # 0-100, Tier 1, always trusted
    pulse_detected: bool = False                     # rPPG, Tier 3 — only counts if True
    bpm: Optional[int] = None

    # Identity graph
    identity_graph_flagged: bool = False             # same face embedding, different doc -> HARD GATE


# ---------------------------------------------------------------------------
# Soft-signal scorers
# ---------------------------------------------------------------------------

def _validation_confidence(inp: RiskEngineInput, reasons: List[str]) -> float:
    """0-1 confidence over ONLY the soft validation checks that apply to
    this document type. Checksum is handled separately as a hard gate,
    never scored here."""
    checks = [c for c in (inp.qr_code_match, inp.expiry_date_valid, inp.field_format_valid)
              if c is not None]

    if not checks:
        reasons.append("No applicable soft validation checks recorded for this document type.")
        return 0.5  # neutral — distinct from "all checks failed" (0.0)

    passed = sum(1 for c in checks if c)
    confidence = passed / len(checks)
    if confidence < 1.0:
        reasons.append(f"{len(checks) - passed}/{len(checks)} applicable validation check(s) failed.")
    return confidence


def _tamper_fusion(inp: RiskEngineInput, reasons: List[str]) -> float:
    """0-1 RISK (not confidence) — higher = more suspicious. Renormalizes
    weights across only the signals that actually exist."""
    if inp.tamper_cnn is None:
        w = TAMPER_WEIGHTS_NO_CNN
        fused = (w["ela"] * inp.tamper_ela + w["fft"] * inp.tamper_fft) / 100.0
        reasons.append("No CNN tamper score available — fused from ELA/FFT only (renormalized weights).")
    else:
        w = TAMPER_WEIGHTS_FULL
        fused = (
            w["ela"] * inp.tamper_ela + w["fft"] * inp.tamper_fft + w["cnn"] * inp.tamper_cnn
        ) / 100.0

    if fused > 0.5:
        cnn_str = f"{inp.tamper_cnn:.0f}" if inp.tamper_cnn is not None else "n/a"
        reasons.append(
            f"Tamper fusion elevated ({fused:.2f}) — "
            f"ela={inp.tamper_ela:.0f}, fft={inp.tamper_fft:.0f}, cnn={cnn_str}."
        )
    return fused


def _face_soft(inp: RiskEngineInput) -> float:
    """0-1 confidence, only reached when face_similarity already cleared the
    hard-gate threshold. Scaled across the passing band."""
    span = 100.0 - FACE_MATCH_HARD_GATE_THRESHOLD
    conf = (inp.face_similarity - FACE_MATCH_HARD_GATE_THRESHOLD) / span
    return max(0.0, min(1.0, conf))


def _liveness_soft(inp: RiskEngineInput, reasons: List[str]) -> float:
    """0-1 confidence. Blink score (Tier 1) always counts. Pulse (rPPG,
    Tier 3) is blended in ONLY when successfully detected; otherwise
    degrades gracefully to blink-only — never penalized."""
    blink_conf = inp.blink_score / 100.0

    if inp.pulse_detected:
        return min(1.0, 0.85 * blink_conf + 0.15)
    else:
        reasons.append(
            "Pulse not captured — liveness scored on blink/challenge-response only "
            "(rPPG excluded from scoring, not penalized)."
        )
        return blink_conf


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_risk_engine(inp: RiskEngineInput) -> RiskResult:
    reasons: List[str] = []

    # --- 1. HARD GATES (short-circuit, never averaged) --------------------
    if inp.checksum_valid is False:
        return RiskResult(
            band=RiskBand.RED, score=1.0, hard_gated=True,
            hard_gate_reason="Checksum invalid.",
            reasons=[f"Checksum failed for document type '{inp.document_type}'."],
        )

    if inp.face_similarity < FACE_MATCH_HARD_GATE_THRESHOLD:
        return RiskResult(
            band=RiskBand.RED, score=1.0, hard_gated=True,
            hard_gate_reason="Face does not match.",
            reasons=[f"Face similarity {inp.face_similarity:.1f} below "
                     f"threshold {FACE_MATCH_HARD_GATE_THRESHOLD:.1f}."],
        )

    if inp.blink_score < BLINK_LIVENESS_HARD_GATE_THRESHOLD:
        return RiskResult(
            band=RiskBand.RED, score=1.0, hard_gated=True,
            hard_gate_reason="Liveness failed.",
            reasons=[f"Blink/challenge-response score {inp.blink_score:.1f} "
                     f"below threshold {BLINK_LIVENESS_HARD_GATE_THRESHOLD:.1f}."],
        )

    if inp.identity_graph_flagged:
        return RiskResult(
            band=RiskBand.RED, score=1.0, hard_gated=True,
            hard_gate_reason="Identity graph flagged reuse.",
            reasons=["Same face embedding previously seen under a different identity/document number."],
        )

    # --- 2. WEIGHTED FUSION (soft signals only) ----------------------------
    ocr_conf = inp.ocr_confidence / 100.0
    validation_conf = _validation_confidence(inp, reasons)
    tamper_risk = _tamper_fusion(inp, reasons)
    face_conf = _face_soft(inp)
    liveness_conf = _liveness_soft(inp, reasons)

    trust = (
        GROUP_WEIGHTS["ocr_confidence"] * ocr_conf
        + GROUP_WEIGHTS["validation_confidence"] * validation_conf
        + GROUP_WEIGHTS["tampering"] * (1.0 - tamper_risk)
        + GROUP_WEIGHTS["face_soft"] * face_conf
        + GROUP_WEIGHTS["liveness_soft"] * liveness_conf
    )

    risk_score = max(0.0, min(1.0, 1.0 - trust))

    if risk_score >= RED_THRESHOLD:
        band = RiskBand.RED
    elif risk_score >= YELLOW_THRESHOLD:
        band = RiskBand.YELLOW
    else:
        band = RiskBand.GREEN

    if not reasons:
        reasons.append("All checks within normal range.")

    return RiskResult(band=band, score=risk_score, reasons=reasons, hard_gated=False)


# ---------------------------------------------------------------------------
# Smoke test — python risk_engine.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Case 1: genuine passport, everything passes, real CNN score present
    genuine = RiskEngineInput(
        document_type="passport", ocr_confidence=96,
        checksum_valid=True, qr_code_match=True, expiry_date_valid=True, field_format_valid=True,
        tamper_ela=5, tamper_fft=8, tamper_cnn=4,
        face_similarity=92, blink_score=88, pulse_detected=True, bpm=74,
    )
    print("Genuine (with CNN):        ", run_risk_engine(genuine))

    # Case 2: tampered doc — NO CNN score available (current real-world case)
    tampered = RiskEngineInput(
        document_type="passport", ocr_confidence=90,
        checksum_valid=True, qr_code_match=True, expiry_date_valid=True, field_format_valid=True,
        tamper_ela=82, tamper_fft=60, tamper_cnn=None,
        face_similarity=88, blink_score=85, pulse_detected=False,
    )
    print("Tampered (no CNN):         ", run_risk_engine(tampered))

    # Same case WITH a hypothetical future CNN score, to show the difference
    tampered_with_cnn = RiskEngineInput(
        document_type="passport", ocr_confidence=90,
        checksum_valid=True, qr_code_match=True, expiry_date_valid=True, field_format_valid=True,
        tamper_ela=82, tamper_fft=60, tamper_cnn=85,
        face_similarity=88, blink_score=85, pulse_detected=False,
    )
    print("Tampered (with CNN=85):    ", run_risk_engine(tampered_with_cnn))

    # Case 3: genuine passport, wrong person
    wrong_person = RiskEngineInput(
        document_type="passport", ocr_confidence=94,
        checksum_valid=True, qr_code_match=True, expiry_date_valid=True, field_format_valid=True,
        tamper_ela=6, tamper_fft=9, tamper_cnn=5,
        face_similarity=22, blink_score=90, pulse_detected=True, bpm=70,
    )
    print("Wrong person:              ", run_risk_engine(wrong_person))

    # Case 4: same face, different identity — national_id, Aadhaar checksum passes
    identity_reuse = RiskEngineInput(
        document_type="national_id", ocr_confidence=91,
        checksum_valid=True, expiry_date_valid=True, field_format_valid=True,
        tamper_ela=10, tamper_fft=12, tamper_cnn=8,
        face_similarity=95, blink_score=87, pulse_detected=False,
        identity_graph_flagged=True,
    )
    print("Identity reuse:            ", run_risk_engine(identity_reuse))

    # Case 5: national_id with a FAILED Aadhaar checksum — must hard-gate
    # even though document_type isn't passport/visa (this is the bug v1 had)
    bad_aadhaar = RiskEngineInput(
        document_type="national_id", ocr_confidence=90,
        checksum_valid=False, expiry_date_valid=True, field_format_valid=True,
        tamper_ela=10, tamper_fft=8, tamper_cnn=5,
        face_similarity=91, blink_score=85, pulse_detected=True, bpm=72,
    )
    print("Bad Aadhaar checksum:      ", run_risk_engine(bad_aadhaar))

    # Case 6: rPPG unreadable — should NOT hurt an otherwise clean result
    bad_lighting = RiskEngineInput(
        document_type="driving_license", ocr_confidence=93,
        qr_code_match=True, expiry_date_valid=True, field_format_valid=True,
        tamper_ela=8, tamper_fft=10, tamper_cnn=None,
        face_similarity=90, blink_score=86, pulse_detected=False,
    )
    print("Bad lighting (still GREEN?):", run_risk_engine(bad_lighting))
