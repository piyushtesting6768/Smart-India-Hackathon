"""
audit_module.py

Audit Trail — hash-chained verification log. Writes one entry per
verification, WITHOUT storing the uploaded image itself — only its SHA-256
hash, plus each module's score/state/detail and the final verdict.

Fully self-contained: no imports from any other file in this project (no
schemas.py, no other module). Accepts plain objects with .score/.state/
.detail attributes (works with a dataclass, a Pydantic model, a
SimpleNamespace, or anything else your teammates' modules happen to
return — this file doesn't care about the exact type, only those three
attributes).

============================================================================
WHY THIS DOESN'T STORE THE RAW IMAGE
============================================================================
Storing real ID photos/selfies in a database raises real privacy questions
(PII, likely regulated depending on jurisdiction/use case). A hash is
enough to prove "this exact file was checked at this exact time" without
keeping a copy of anyone's ID card sitting in a database.

============================================================================
WHY THIS SHOULDN'T INTERRUPT ANYONE ELSE'S MODULE
============================================================================
  - This file has zero imports from any other module — it only needs
    objects with .score / .state / .detail attributes, whatever their
    actual type is.
  - Call this ONLY AFTER all 4 modules have already produced their
    results — it's a pure side effect at the very end of the pipeline,
    never something another module depends on or waits for.
  - Whoever wires this into an API (main.py or similar) should wrap the
    call in try/except so a logging failure (disk full, permissions, a
    locked SQLite file) can NEVER break the actual verification response
    the user is waiting on. See the example at the bottom of this
    docstring.
  - Uses its own SQLite file (audit_log.db, auto-created on first write) —
    no shared server, no config, no coordination needed with anyone else's
    setup or database.

============================================================================
HOW THE HASH CHAIN WORKS
============================================================================
Every entry's hash is computed from its own fields PLUS the previous
entry's hash. If anyone edits or deletes a row directly in the database,
every hash from that point forward stops matching what
verify_chain_integrity() recomputes on demand — tampering with the log
itself becomes detectable. No blockchain infrastructure, just hashlib
(built-in).

============================================================================
USAGE
============================================================================
    from audit_module import compute_document_hash, log_verification

    doc_hash = compute_document_hash(doc_path)
    entry_hash = log_verification(
        document_hash=doc_hash,
        ocr=ocr_result, validation=validation_result,
        tampering=tampering_result, face=face_result,
        risk_score=risk_score, verdict=verdict,
    )
    # ocr/validation/tampering/face just need .score, .state, .detail
    # attributes — any object works, no specific class required.

Wrapped safely in an API route, e.g.:

    try:
        doc_hash = compute_document_hash(doc_path)
        log_verification(document_hash=doc_hash, ocr=ocr, validation=validation,
                          tampering=tampering, face=face, risk_score=risk_score,
                          verdict=verdict)
    except Exception as exc:
        print(f"Audit logging failed (non-fatal): {exc}")
        # response still returns normally — logging failure never blocks it

To check the log hasn't been tampered with later:
    from audit_module import verify_chain_integrity
    is_valid, broken_row_id = verify_chain_integrity()

============================================================================
STANDALONE TEST — works with nothing else, no other files needed
============================================================================
    python audit_module.py
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple

_DB_PATH = Path(__file__).resolve().parent / "audit_log.db"
_GENESIS_HASH = "0" * 64  # chain "root" — the prev_hash of the very first entry ever


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(_DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            document_hash TEXT NOT NULL,
            prev_hash TEXT NOT NULL,
            entry_hash TEXT NOT NULL UNIQUE,
            ocr_score REAL, ocr_state TEXT, ocr_detail TEXT,
            validation_score REAL, validation_state TEXT, validation_detail TEXT,
            tampering_score REAL, tampering_state TEXT, tampering_detail TEXT,
            face_score REAL, face_state TEXT, face_detail TEXT,
            risk_score REAL NOT NULL,
            verdict TEXT NOT NULL
        )
        """
    )
    return conn


def compute_document_hash(image_path: str) -> str:
    """
    SHA-256 of the uploaded file's raw bytes — a fingerprint, not a copy.
    Reads in chunks so this never needs to load a large image fully into
    memory just to hash it.
    """
    sha256 = hashlib.sha256()
    with open(image_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _get_last_entry_hash(conn: sqlite3.Connection) -> str:
    row = conn.execute(
        "SELECT entry_hash FROM audit_log ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return row[0] if row else _GENESIS_HASH


def _entry_payload(
    timestamp: str, document_hash: str, prev_hash: str,
    ocr: Any, validation: Any, tampering: Any, face: Any,
    risk_score: float, verdict: str,
) -> dict:
    """
    The exact fields that get hashed together. ocr/validation/tampering/face
    just need .score / .state / .detail attributes — no specific class
    required, duck typing only.

    Every numeric value is explicitly cast to float and every text value to
    str here. This matters more than it looks: if a caller passes an int
    score (e.g. `score=90`) at log time, but SQLite hands the same value
    back as a float (`90.0`) when verify_chain_integrity() re-reads it,
    `json.dumps(90)` and `json.dumps(90.0)` produce different strings — and
    the recomputed hash would never match the stored one, even though
    nothing was actually tampered with. Casting explicitly here makes the
    hash deterministic regardless of what numeric type happened to be
    passed in.
    """
    return {
        "timestamp": timestamp,
        "document_hash": document_hash,
        "prev_hash": prev_hash,
        "ocr": {"score": float(ocr.score), "state": str(ocr.state), "detail": str(ocr.detail)},
        "validation": {"score": float(validation.score), "state": str(validation.state), "detail": str(validation.detail)},
        "tampering": {"score": float(tampering.score), "state": str(tampering.state), "detail": str(tampering.detail)},
        "face": {"score": float(face.score), "state": str(face.state), "detail": str(face.detail)},
        "risk_score": float(risk_score),
        "verdict": str(verdict),
    }


def log_verification(
    document_hash: str,
    ocr: Any,
    validation: Any,
    tampering: Any,
    face: Any,
    risk_score: float,
    verdict: str,
) -> str:
    """
    Writes one hash-chained row to audit_log.db. Returns the new entry's
    own hash (becomes the next entry's prev_hash automatically — this
    function looks it up itself, callers don't need to track it).

    ocr/validation/tampering/face can be any object with .score, .state,
    .detail attributes (a dataclass, a Pydantic model, a SimpleNamespace,
    whatever each module actually returns) — no shared type is required.

    Never touches the actual document image — only the hash passed in
    (see compute_document_hash above). Can raise (sqlite3.Error, OSError)
    if the database file can't be written — the caller is responsible for
    catching that so a logging failure never breaks whatever response it's
    part of.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    conn = _get_connection()
    try:
        prev_hash = _get_last_entry_hash(conn)
        payload = _entry_payload(
            timestamp, document_hash, prev_hash, ocr, validation, tampering, face, risk_score, verdict
        )
        entry_hash = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

        conn.execute(
            """
            INSERT INTO audit_log (
                timestamp, document_hash, prev_hash, entry_hash,
                ocr_score, ocr_state, ocr_detail,
                validation_score, validation_state, validation_detail,
                tampering_score, tampering_state, tampering_detail,
                face_score, face_state, face_detail,
                risk_score, verdict
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp, document_hash, prev_hash, entry_hash,
                ocr.score, ocr.state, ocr.detail,
                validation.score, validation.state, validation.detail,
                tampering.score, tampering.state, tampering.detail,
                face.score, face.state, face.detail,
                risk_score, verdict,
            ),
        )
        conn.commit()
        return entry_hash
    finally:
        conn.close()


def verify_chain_integrity() -> Tuple[bool, Optional[int]]:
    """
    Recomputes every entry's hash from its stored fields and checks the
    chain end-to-end. If anyone edited a value or deleted a row directly
    in the database, this detects exactly where the chain breaks.

    Returns (is_valid, first_broken_row_id). first_broken_row_id is None
    if is_valid is True, or if the log is empty.
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            """
            SELECT id, timestamp, document_hash, prev_hash, entry_hash,
                   ocr_score, ocr_state, ocr_detail,
                   validation_score, validation_state, validation_detail,
                   tampering_score, tampering_state, tampering_detail,
                   face_score, face_state, face_detail,
                   risk_score, verdict
            FROM audit_log ORDER BY id ASC
            """
        ).fetchall()

        @dataclass
        class _R:
            score: float
            state: str
            detail: str

        expected_prev = _GENESIS_HASH
        for row in rows:
            (row_id, timestamp, document_hash, prev_hash, entry_hash,
             ocr_score, ocr_state, ocr_detail,
             validation_score, validation_state, validation_detail,
             tampering_score, tampering_state, tampering_detail,
             face_score, face_state, face_detail,
             risk_score, verdict) = row

            if prev_hash != expected_prev:
                return False, row_id

            payload = _entry_payload(
                timestamp, document_hash, prev_hash,
                _R(ocr_score, ocr_state, ocr_detail),
                _R(validation_score, validation_state, validation_detail),
                _R(tampering_score, tampering_state, tampering_detail),
                _R(face_score, face_state, face_detail),
                risk_score, verdict,
            )
            recomputed = hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode("utf-8")
            ).hexdigest()

            if recomputed != entry_hash:
                return False, row_id

            expected_prev = entry_hash

        return True, None
    finally:
        conn.close()


def get_recent_entries(limit: int = 20) -> List[dict]:
    """Convenience read for a future admin view or /audit-log endpoint."""
    conn = _get_connection()
    try:
        cols = [
            "id", "timestamp", "document_hash", "prev_hash", "entry_hash",
            "ocr_score", "ocr_state", "ocr_detail",
            "validation_score", "validation_state", "validation_detail",
            "tampering_score", "tampering_state", "tampering_detail",
            "face_score", "face_state", "face_detail",
            "risk_score", "verdict",
        ]
        rows = conn.execute(
            f"SELECT {', '.join(cols)} FROM audit_log ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(zip(cols, row)) for row in rows]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Standalone test — works with nothing else, no other files needed
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from dataclasses import dataclass as _dc

    @_dc
    class _DummyResult:
        score: float
        state: str
        detail: str

    print(f"Audit DB path: {_DB_PATH}")

    dummy = _DummyResult(score=90, state="PASS", detail="test")
    doc_hash = hashlib.sha256(b"fake-image-bytes-for-testing").hexdigest()

    entry_hash = log_verification(
        document_hash=doc_hash,
        ocr=dummy, validation=dummy, tampering=dummy, face=dummy,
        risk_score=90.0, verdict="GENUINE",
    )
    print(f"Logged test entry: {entry_hash}")

    is_valid, broken_id = verify_chain_integrity()
    print(f"Chain valid: {is_valid} (broken row: {broken_id})")

    print(f"Recent entries: {len(get_recent_entries())}")
