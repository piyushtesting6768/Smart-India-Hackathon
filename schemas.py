"""
Shared data models for the pipeline.
Every module returns one of these so main.py can combine them without
caring about each module's internals.
"""

from pydantic import BaseModel
from typing import Optional


class OCRResult(BaseModel):
    fields: dict            # e.g. {"name": "...", "dob": "...", "id_number": "..."}
    avg_confidence: float   # 0-1, averaged across fields


class ValidationResult(BaseModel):
    checksum_ok: bool
    qr_matches_text: bool
    format_ok: bool
    not_expired: bool
    score: float            # 0-1, rolled up from the 4 checks above


class TamperingResult(BaseModel):
    ela_score: float        # 0-1, higher = more suspicious
    fft_spike_score: float  # 0-1, higher = more suspicious (structured/grid-like)
    tampering_score: float  # 0-1 combined, higher = more suspicious


class FaceVerificationResult(BaseModel):
    match: bool
    distance: float          # raw embedding distance (model-dependent)
    match_score: float       # 0-1, normalized (1 = perfect match)
    model_used: str
    is_real: Optional[bool] = None          # DeepFace anti-spoofing verdict on the selfie
    antispoofing_score: Optional[float] = None  # 0-1, confidence the selfie face is real


class LivenessResult(BaseModel):
    method_used: str         # "rppg" or "blink_fallback"
    pulse_detected: Optional[bool] = None
    liveness_score: float    # 0-1, higher = more likely a live human


class RiskVerdict(BaseModel):
    risk_score: float        # 0-100
    verdict: str              # GENUINE / SUSPICIOUS / FAKE
    breakdown: dict           # per-module contribution, for the officer UI