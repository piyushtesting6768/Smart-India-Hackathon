""""
Document Validation Module
---------------------------
Part of the AI-Powered Fake Document Detection System.

This module owns the "Document Validation" check (25% weight in the final
Risk Score). It answers: "is the data on this document internally
consistent?" — independent of whether the image itself has been tampered
with (that's the Tampering Detection module's job).

Four checks, each scored pass=100 / fail=0, then averaged:
  1. Checksum validation      — does the ID number's check digit compute correctly?
  2. QR/barcode vs OCR match  — does the QR code agree with the printed text?
  3. Expiry validation        — is the document still valid, are dates sane?
  4. Format validation        — does every field match the expected pattern
                                 for this document type?

Usage:
    validator = DocumentValidator()
    result = validator.run_all(
        doc_type="passport",
        fields={
            "name": "JOHN SMITH",
            "id_number": "M1234567",
            "dob": "1995-04-12",
            "expiry_date": "2029-11-30",
        },
        document_image_path="sample_passport.jpg",  # optional, for QR check
    )
    print(result["overall_score"], result["verdict_contribution"])
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from PIL import Image
from pyzbar.pyzbar import decode as zbar_decode
from stdnum.in_ import aadhaar as stdnum_aadhaar


# ---------------------------------------------------------------------------
# 1. Result data structure
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    """Outcome of a single validation check."""
    name: str
    applicable: bool          # False if this check couldn't be run (e.g. no QR image supplied)
    passed: Optional[bool]    # None if not applicable
    score: Optional[float]    # 0 or 100, None if not applicable
    details: str = ""


# ---------------------------------------------------------------------------
# 2. Format rules per document type
# ---------------------------------------------------------------------------
# These are illustrative starting patterns — the doc explicitly says to be
# honest that thresholds/rules are tunable, so treat these as a first pass
# and refine once you see real sample data from your OCR teammate.

DOCUMENT_FIELD_PATTERNS = {
    "passport": {
        "id_number": r"^[A-PR-WYa-pr-wy][0-9]{7}$",   # Indian passport format: 1 letter + 7 digits
        "dob": r"^\d{4}-\d{2}-\d{2}$",
        "expiry_date": r"^\d{4}-\d{2}-\d{2}$",
    },
    "national_id": {  # Aadhaar
        "id_number": r"^\d{4}\s?\d{4}\s?\d{4}$",
        "dob": r"^\d{4}-\d{2}-\d{2}$",
    },
    "driving_licence": {
        "id_number": r"^[A-Z]{2}[0-9]{2}\s?[0-9]{11}$",  # e.g. state+RTO code + serial (illustrative)
        "dob": r"^\d{4}-\d{2}-\d{2}$",
        "expiry_date": r"^\d{4}-\d{2}-\d{2}$",
    },
    "visa": {
        "id_number": r"^[A-Z0-9]{6,10}$",
        "expiry_date": r"^\d{4}-\d{2}-\d{2}$",
    },
    "permit": {
        "id_number": r"^[A-Z0-9]{4,12}$",
        "expiry_date": r"^\d{4}-\d{2}-\d{2}$",
    },
}


# ---------------------------------------------------------------------------
# 3. Checksum logic
# ---------------------------------------------------------------------------

def mrz_check_digit(data: str) -> int:
    """
    Compute the ICAO 9303 MRZ check digit (used on passport MRZ lines).
    Weights repeat 7,3,1. '<' = 0, digits = themselves, letters A-Z = 10-35.
    """
    weights = [7, 3, 1]
    total = 0
    for i, ch in enumerate(data):
        if ch.isdigit():
            val = int(ch)
        elif ch.isalpha():
            val = ord(ch.upper()) - ord("A") + 10
        else:  # filler character '<'
            val = 0
        total += val * weights[i % 3]
    return total % 10


def validate_checksum(doc_type: str, fields: dict) -> CheckResult:
    """
    Dispatch to the right checksum scheme for the document type.
    Falls back to "not applicable" for document types with no publicly
    documented checksum scheme (e.g. most visas, many driving licences) —
    be upfront about this limitation rather than faking a pass.
    """
    id_number = fields.get("id_number", "").replace(" ", "")

    if doc_type == "national_id":
        # Aadhaar uses a Verhoeff-algorithm checksum; python-stdnum implements it.
        if not id_number:
            return CheckResult("checksum", True, False, 0, "No ID number provided")
        is_valid = stdnum_aadhaar.is_valid(id_number)
        return CheckResult(
            "checksum", True, is_valid, 100 if is_valid else 0,
            "Aadhaar Verhoeff checksum " + ("passed" if is_valid else "failed"),
        )

    if doc_type == "passport" and "mrz_line" in fields:
        # If OCR extracted a raw MRZ line, verify its embedded check digit.
        mrz = fields["mrz_line"]
        # TD3 passport MRZ line 2 layout puts the check digit right after
        # the document number (positions vary — adjust to your OCR output).
        doc_num_field = mrz[0:9]
        expected_check = mrz[9] if len(mrz) > 9 else None
        computed = mrz_check_digit(doc_num_field)
        if expected_check is None or not expected_check.isdigit():
            return CheckResult("checksum", True, False, 0, "MRZ check digit missing/unreadable")
        is_valid = computed == int(expected_check)
        return CheckResult(
            "checksum", True, is_valid, 100 if is_valid else 0,
            f"MRZ check digit {'matched' if is_valid else f'mismatch (expected {expected_check}, got {computed})'}",
        )

    # No known checksum scheme wired up for this doc type / field combo yet.
    return CheckResult(
        "checksum", False, None, None,
        f"No checksum scheme implemented for '{doc_type}' with available fields",
    )


# ---------------------------------------------------------------------------
# 4. QR / barcode vs OCR text match
# ---------------------------------------------------------------------------

def decode_qr_from_image(image_path: str) -> list[str]:
    """Decode any QR/barcodes found in the document image. Returns raw payload strings."""
    img = Image.open(image_path)
    decoded_objects = zbar_decode(img)
    return [obj.data.decode("utf-8", errors="replace") for obj in decoded_objects]


def parse_qr_payload(raw: str) -> dict:
    """
    Turn a raw QR payload into a field dict for comparison against OCR output.

    NOTE: real-world QR schemas vary a lot by issuer (Aadhaar's QR is a signed
    XML/protobuf blob, for example — decoding that is its own mini-project).
    This parser handles the simple case of "key:value|key:value" style
    payloads. Swap this out once you know your sample documents' real format.
    """
    fields = {}
    if "|" in raw or ":" in raw:
        for part in raw.split("|"):
            if ":" in part:
                key, _, value = part.partition(":")
                fields[key.strip().lower()] = value.strip()
    return fields


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip().upper()


def validate_qr_match(ocr_fields: dict, document_image_path: Optional[str]) -> CheckResult:
    """Compare QR-embedded data against OCR'd printed text. Mismatch = strong fake signal."""
    if not document_image_path:
        return CheckResult("qr_match", False, None, None, "No document image supplied for QR decode")

    try:
        payloads = decode_qr_from_image(document_image_path)
    except Exception as exc:
        return CheckResult("qr_match", True, False, 0, f"QR decode error: {exc}")

    if not payloads:
        return CheckResult("qr_match", False, None, None, "No QR/barcode found on document")

    qr_fields = parse_qr_payload(payloads[0])
    if not qr_fields:
        return CheckResult("qr_match", True, False, 0, "QR found but payload unparseable/empty")

    mismatches = []
    compared = 0
    for key in ("name", "id_number", "dob"):
        if key in ocr_fields and key in qr_fields:
            compared += 1
            if _normalize(ocr_fields[key]) != _normalize(qr_fields[key]):
                mismatches.append(key)

    if compared == 0:
        return CheckResult("qr_match", False, None, None, "No overlapping fields between OCR and QR to compare")

    passed = len(mismatches) == 0
    detail = "All overlapping fields matched" if passed else f"Mismatch on: {', '.join(mismatches)}"
    return CheckResult("qr_match", True, passed, 100 if passed else 0, detail)


# ---------------------------------------------------------------------------
# 5. Expiry validation
# ---------------------------------------------------------------------------

_DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d %b %Y"]


def _parse_date(value: str) -> Optional[date]:
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def validate_expiry(fields: dict) -> CheckResult:
    """Checks the document hasn't expired, and that dates are logically ordered."""
    expiry_raw = fields.get("expiry_date")
    if not expiry_raw:
        return CheckResult("expiry", False, None, None, "No expiry_date field present")

    expiry = _parse_date(expiry_raw)
    if expiry is None:
        return CheckResult("expiry", True, False, 0, f"Unparseable expiry date: '{expiry_raw}'")

    if expiry < date.today():
        return CheckResult("expiry", True, False, 0, f"Document expired on {expiry.isoformat()}")

    # Optional sanity check: DOB should be before expiry
    dob_raw = fields.get("dob")
    if dob_raw:
        dob = _parse_date(dob_raw)
        if dob and dob >= expiry:
            return CheckResult("expiry", True, False, 0, "DOB is not before expiry date — inconsistent dates")

    return CheckResult("expiry", True, True, 100, f"Valid, expires {expiry.isoformat()}")


# ---------------------------------------------------------------------------
# 6. Format validation
# ---------------------------------------------------------------------------

def validate_format(doc_type: str, fields: dict) -> CheckResult:
    """Regex-checks every field we have a known pattern for, for this doc type."""
    patterns = DOCUMENT_FIELD_PATTERNS.get(doc_type)
    if not patterns:
        return CheckResult("format", False, None, None, f"No format rules defined for doc_type '{doc_type}'")

    failures = []
    checked = 0
    for field_name, pattern in patterns.items():
        if field_name in fields:
            checked += 1
            if not re.match(pattern, fields[field_name].strip()):
                failures.append(field_name)

    if checked == 0:
        return CheckResult("format", False, None, None, "None of the expected fields were present to check")

    passed = len(failures) == 0
    detail = "All fields matched expected format" if passed else f"Format violation in: {', '.join(failures)}"
    return CheckResult("format", True, passed, 100 if passed else 0, detail)


# ---------------------------------------------------------------------------
# 7. Orchestrator
# ---------------------------------------------------------------------------

class DocumentValidator:
    """Runs all four validation checks and produces the module's 0-100 score."""

    def run_all(self, doc_type: str, fields: dict, document_image_path: Optional[str] = None) -> dict:
        checks = [
            validate_checksum(doc_type, fields),
            validate_qr_match(fields, document_image_path),
            validate_expiry(fields),
            validate_format(doc_type, fields),
        ]

        applicable_scores = [c.score for c in checks if c.applicable and c.score is not None]
        overall_score = round(sum(applicable_scores) / len(applicable_scores), 1) if applicable_scores else 0.0

        return {
            "overall_score": overall_score,          # feeds into Risk Engine as Validation_Score
            "checks_run": len(applicable_scores),
            "checks_skipped": len(checks) - len(applicable_scores),
            "checks": [
                {
                    "name": c.name,
                    "applicable": c.applicable,
                    "passed": c.passed,
                    "score": c.score,
                    "details": c.details,
                }
                for c in checks
            ],
        }


# ---------------------------------------------------------------------------
# 8. Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import qrcode

    validator = DocumentValidator()

    # --- Test 1: Aadhaar with valid checksum, no QR image ---
    print("=== Test 1: Aadhaar, checksum only ===")
    result = validator.run_all(
        doc_type="national_id",
        fields={
            "name": "PRIYA SHARMA",
            "id_number": "234123412346",  # valid test Aadhaar-style checksum number
            "dob": "1998-06-21",
        },
    )
    import json
    print(json.dumps(result, indent=2))

    # --- Test 2: Passport with a mocked QR image that matches OCR fields ---
    print("\n=== Test 2: Passport with matching QR code ===")
    ocr_fields = {
        "name": "JOHN SMITH",
        "id_number": "M1234567",
        "dob": "1995-04-12",
        "expiry_date": "2029-11-30",
    }
    qr_payload = "name:JOHN SMITH|id_number:M1234567|dob:1995-04-12"
    qr_img = qrcode.make(qr_payload)
    qr_img.save("test_passport_qr.png")

    result = validator.run_all(
        doc_type="passport",
        fields=ocr_fields,
        document_image_path="test_passport_qr.png",
    )
    print(json.dumps(result, indent=2))

    # --- Test 3: Passport with QR that DOESN'T match (simulated fake) ---
    print("\n=== Test 3: Passport with mismatched QR (simulated tampering) ===")
    fake_qr_payload = "name:JOHN SMITH|id_number:M9999999|dob:1995-04-12"
    qr_img2 = qrcode.make(fake_qr_payload)
    qr_img2.save("test_fake_qr.png")

    result = validator.run_all(
        doc_type="passport",
        fields=ocr_fields,
        document_image_path="test_fake_qr.png",
    )
    print(json.dumps(result, indent=2))

    # --- Test 4: Expired document ---
    print("\n=== Test 4: Expired document ===")
    result = validator.run_all(
        doc_type="visa",
        fields={"id_number": "AB123456", "expiry_date": "2020-01-01"},
    )
    print(json.dumps(result, indent=2))