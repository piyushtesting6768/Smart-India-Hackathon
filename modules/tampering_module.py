"""
tampering_module.py

Tampering Detection — Module 3 of 4 in the Document Guard pipeline.

Fully self-contained: no imports from any other file in this project
(no schemas.py, no other module). Defines its own result types locally.
Drop this next to your teammates' own standalone module files — nothing
here depends on how they structured theirs.

============================================================================
WHAT THIS DOES
============================================================================
Looks at a single uploaded document image and estimates whether it's been
digitally edited, spliced, or reprinted, using three independent,
explainable, rule-based signals (no trained model, nothing to train, no
extra dependencies beyond Pillow + numpy):

  1. ELA (Error Level Analysis) — re-compresses the image at a known JPEG
     quality and diffs it against the original. A region pasted in from a
     different source (or edited and re-saved) carries a different
     compression "fingerprint" than the rest of the page, so it lights up
     in the diff.

  2. FFT spike analysis — looks at the image's frequency spectrum (with a
     Hann window applied first, to avoid a known FFT artifact called
     spectral leakage) for unnatural periodic patterns — the kind left by
     screen photography, resampling, or some splicing operations.

  3. Editing-software EXIF fingerprint — checks whether the image's
     metadata names a known photo editor (Photoshop, GIMP, etc.) as the
     software that last touched the file. Cheap, free, and a real tell
     when present.

These are blended into a single tampering_score (0-100, higher = more
genuine) meant to plug into an overall Risk Score formula, e.g.:

    Risk Score = 0.20*OCR + 0.25*Validation + 0.35*Tampering + 0.20*Face

Tampering is designed to carry the highest weight of the four modules —
the reasoning being that tampering evidence is the hardest to fake and the
most direct proof of forgery.

============================================================================
KNOWN LIMITATION (say this openly in the demo)
============================================================================
This is rule-based, not a trained model. It's genuinely good at catching
spliced/pasted content (mismatched compression history) but can miss a
careful forgery that's edited and then re-saved once, cleanly, as a single
new JPEG — there's no compression mismatch left to find at that point. A
trained CNN (e.g. ResNet-50/EfficientNet fine-tuned on a tampering dataset)
would close that gap; one wasn't built here due to time/dataset/GPU
constraints. Say this plainly if asked — it's a real, known gap, not a
hidden one.

============================================================================
INTERFACE
============================================================================
    from tampering_module import analyze_tampering, TamperingResult

    result: TamperingResult = analyze_tampering(image_path)
    # result.tampering_score        -> 0-100, higher = more genuine
    # result.explanation            -> one-line string for the UI
    # result.flagged_regions        -> list of SuspiciousRegion boxes

analyze_tampering() raises ValueError on a corrupt/unreadable/too-small
image. Whoever wires this into an API should catch that and return HTTP
422, not let it surface as an unhandled 500. This function does CPU-bound
work (FFT, block scanning) so an async caller should run it via
`fastapi.concurrency.run_in_threadpool` rather than directly inside an
`async def` route, or it'll block the event loop for every other request.

============================================================================
STANDALONE TEST — works with nothing else, no other files needed
============================================================================
    python tampering_module.py path/to/some_image.jpg
"""

from __future__ import annotations

import io
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from PIL import ExifTags, Image, ImageChops, UnidentifiedImageError


# ---------------------------------------------------------------------------
# Result types — defined locally, no shared schema file needed
# ---------------------------------------------------------------------------

@dataclass
class SuspiciousRegion:
    """A bounding box flagged as anomalous, for a frontend to draw a
    highlight box over in a results dashboard. Coordinates are in the
    ORIGINAL image's pixel space, even if analysis internally downscaled
    the image for speed (see `_prepare_image`)."""
    x: int
    y: int
    w: int
    h: int
    reason: str          # "ela_anomaly" | "fft_spike" | "editor_metadata"
    severity: float       # 0-1, how confident we are this region is real evidence


@dataclass
class TamperingResult:
    tampering_score: float                 # 0-100, HIGHER = more genuine
    forgery_probability: float             # 0-1, HIGHER = more likely tampered
    ela_anomaly_score: float               # 0-1
    fft_spike_score: float                 # 0-1
    editor_metadata_score: float = 0.0     # 0-1, 1.0 if a known editor tag was found in EXIF
    flagged_regions: List[SuspiciousRegion] = field(default_factory=list)
    explanation: str = ""


# ---------------------------------------------------------------------------
# 0. Image loading / preparation
# ---------------------------------------------------------------------------

# Very large uploads (a 4000x3000 phone photo, e.g.) make the block scan and
# the FFT needlessly slow without adding detection power — tampering
# artifacts are visible well below full resolution. We analyze at a capped
# resolution and scale flagged-region coordinates back up to the original.
_MAX_ANALYSIS_DIM = 1600

_KNOWN_EDITOR_TAGS = (
    "photoshop", "gimp", "lightroom", "snapseed", "pixlr", "affinity photo",
    "paint.net", "picsart", "canva", "inpaint", "facetune",
)


def _prepare_image(image_path: str) -> Tuple[Image.Image, float]:
    """
    Loads the image robustly and returns (image_for_analysis, scale_factor)
    where scale_factor = original_size / analysis_size, so callers can map
    flagged region coordinates back to the original image.

    Raises ValueError with a clear message on anything unreadable/corrupt.
    """
    try:
        image = Image.open(image_path)
        image.load()  # force-read pixel data now, not lazily later
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Could not read image file: {exc}") from exc

    # Normalize color mode. Palette (P), RGBA, and CMYK images all need to
    # become plain RGB before we do pixel-diff math, or ELA compares
    # differently-shaped arrays and either crashes or silently mis-scores.
    if image.mode not in ("RGB",):
        image = image.convert("RGB")

    w, h = image.size
    if w < 64 or h < 64:
        raise ValueError(f"Image too small to analyze ({w}x{h}px, need at least 64x64).")

    longest_side = max(w, h)
    if longest_side > _MAX_ANALYSIS_DIM:
        scale = longest_side / _MAX_ANALYSIS_DIM
        new_size = (max(1, round(w / scale)), max(1, round(h / scale)))
        analysis_image = image.resize(new_size, Image.LANCZOS)
        return analysis_image, scale

    return image, 1.0


# ---------------------------------------------------------------------------
# 1. Error Level Analysis (ELA)
# ---------------------------------------------------------------------------

def _run_ela(image: Image.Image, quality: int = 90) -> np.ndarray:
    """
    Re-saves the image as JPEG at a fixed quality and diffs it against the
    original. Returns a single-channel float array (H x W) where higher
    values = stronger recompression error = more likely to have been
    edited/pasted in after the fact.
    """
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer)
    resaved.load()

    diff = ImageChops.difference(image, resaved)
    diff_arr = np.asarray(diff).astype(np.float32)

    # Collapse RGB diff into a single "error intensity" map
    ela_map = diff_arr.sum(axis=2)

    # Normalizing by the single max pixel is fragile — one hot pixel (a
    # sensor speck, a stray JPEG block artifact) drags the whole map toward
    # zero and hides everything else. Normalize by a high percentile
    # instead, which is robust to a handful of extreme outliers.
    p99 = float(np.percentile(ela_map, 99))
    if p99 > 0:
        ela_map = np.clip(ela_map / p99, 0.0, 1.0)

    return ela_map


def _score_ela(ela_map: np.ndarray, block_size: int) -> Tuple[float, List[SuspiciousRegion]]:
    """
    Turns the raw ELA heatmap into (a) a single anomaly score and (b) a list
    of flagged regions, by scanning the image in blocks and comparing each
    block's error level to the image-wide median.

    A uniformly-captured genuine document has fairly consistent ELA error
    across blocks (a real photo/scan was saved once). A pasted/edited region
    stands out as a block (or cluster of blocks) with error level well above
    the rest of the page.
    """
    h, w = ela_map.shape
    block_means = []
    blocks = []  # (y, x, mean)

    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block = ela_map[y:y + block_size, x:x + block_size]
            block_means.append(float(block.mean()))
            blocks.append((y, x, block_means[-1]))

    if not block_means:
        return 0.0, []

    block_arr = np.asarray(block_means)
    median = float(np.median(block_arr))

    # Standard deviation is self-defeating here for outlier detection: the
    # very outlier blocks we're trying to find are what inflate the std,
    # raising the bar and hiding the tampering. Median Absolute Deviation
    # (MAD) is the standard robust alternative — it isn't dragged around by
    # the outliers themselves.
    mad = float(np.median(np.abs(block_arr - median))) or 1e-6
    robust_std = mad * 1.4826  # scale factor makes MAD ~ std for normal data

    flagged: List[SuspiciousRegion] = []
    max_z = 0.0

    for y, x, m in blocks:
        z = (m - median) / robust_std
        if z > 3.0:  # block is a robust statistical outlier vs. the rest of the page
            severity = min(1.0, (z - 3.0) / 6.0 + 0.4)
            flagged.append(
                SuspiciousRegion(
                    x=x, y=y, w=block_size, h=block_size,
                    reason="ela_anomaly",
                    severity=round(severity, 3),
                )
            )
            max_z = max(max_z, z)

    coverage_ratio = len(flagged) / len(blocks)
    intensity = min(1.0, max_z / 12.0)
    anomaly_score = min(1.0, 0.6 * intensity + 0.4 * min(1.0, coverage_ratio * 8))

    return round(anomaly_score, 4), flagged


# ---------------------------------------------------------------------------
# 2. FFT spike analysis
# ---------------------------------------------------------------------------

def _run_fft(image: Image.Image) -> np.ndarray:
    """
    Converts the image to grayscale, applies a 2D Hann window, and computes
    the FFT magnitude spectrum (log-scaled, shifted so low frequencies are
    centered).

    A raw (unwindowed) FFT on an image's hard-cutoff edges creates a bright
    cross/plus pattern from "spectral leakage" that has nothing to do with
    tampering. The Hann window (fades the image to zero at its edges before
    transforming) removes that artifact so real periodic tampering signals
    (moire, resampling) stand out instead of being masked by leakage noise.
    """
    gray = np.asarray(image.convert("L")).astype(np.float32)
    h, w = gray.shape

    window = np.outer(np.hanning(h), np.hanning(w))
    windowed = gray * window

    f = np.fft.fft2(windowed)
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))

    return magnitude


def _score_fft(magnitude: np.ndarray) -> float:
    """
    Looks for unnatural periodic spikes in the frequency spectrum, away from
    the DC component at the center. Returns a 0-1 score: higher = more
    evidence of periodic editing/recapture artifacts.
    """
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2

    yy, xx = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius_exclude = min(h, w) * 0.05

    outer_mask = dist_from_center > radius_exclude
    outer_values = magnitude[outer_mask]

    if outer_values.size == 0:
        return 0.0

    mean = float(outer_values.mean())
    std = float(outer_values.std()) or 1e-6

    z_scores = (outer_values - mean) / std
    spike_count = int(np.sum(z_scores > 6))
    spike_ratio = spike_count / outer_values.size

    fft_score = min(1.0, spike_ratio * 4000)

    return round(float(fft_score), 4)


def _flag_fft_region_note(fft_score: float) -> List[SuspiciousRegion]:
    """
    FFT spikes describe the whole image's frequency content, not a specific
    bounding box, so we surface it as a full-frame region (0,0,0,0 is the
    "whole image" convention for a frontend) instead of a tight box.
    """
    if fft_score > 0.5:
        return [
            SuspiciousRegion(
                x=0, y=0, w=0, h=0,
                reason="fft_spike",
                severity=round(fft_score, 3),
            )
        ]
    return []


# ---------------------------------------------------------------------------
# 3. Editing-software EXIF fingerprint
# ---------------------------------------------------------------------------

def _check_editor_metadata(image_path: str) -> float:
    """
    Checks EXIF metadata for a "Software" tag naming a known photo editor.
    Nearly free to compute and catches a real, common case: a forger opens
    the scan in Photoshop/GIMP, edits it, and exports — many tools stamp
    their name into EXIF on save unless the forger specifically strips it.
    A genuine phone/scanner capture almost never carries this tag.

    Returns 1.0 if a known editor is found, else 0.0. Absence of the tag is
    NOT evidence of genuineness (it's trivial to strip), so this only ever
    pushes the score toward "suspicious," never toward "genuine."
    """
    try:
        with Image.open(image_path) as img:
            exif_raw = img.getexif()
            if not exif_raw:
                return 0.0
            tag_map = {ExifTags.TAGS.get(k, k): v for k, v in exif_raw.items()}
            software = str(tag_map.get("Software", "")).lower()
            if any(editor in software for editor in _KNOWN_EDITOR_TAGS):
                return 1.0
    except Exception:
        # Metadata is a bonus signal — never let a metadata-parsing quirk
        # take down the whole tampering check.
        return 0.0
    return 0.0


# ---------------------------------------------------------------------------
# 4. Public entry point
# ---------------------------------------------------------------------------

def analyze_tampering(
    image_path: str,
    ela_quality: int = 90,
    block_size: Optional[int] = None,
) -> TamperingResult:
    """
    Main entry point. Runs ELA + FFT + EXIF metadata analysis on one image
    and returns a TamperingResult.

    Args:
        image_path: path to the uploaded document image.
        ela_quality: JPEG quality used for the ELA re-save comparison.
                     90 is a good default; lower values increase sensitivity
                     but also increase false positives on genuinely
                     low-quality scans.
        block_size: ELA block size in pixels for the outlier scan. If None
                    (default), it's chosen automatically from the image's
                    analysis resolution.

    Raises:
        ValueError if the image can't be read, is corrupt, or is too small
        to analyze meaningfully.
    """
    image, scale = _prepare_image(image_path)
    h, w = image.size[1], image.size[0]

    if block_size is None:
        block_size = int(np.clip(min(h, w) // 20, 16, 64))

    # --- ELA ---
    ela_map = _run_ela(image, quality=ela_quality)
    ela_anomaly_score, ela_regions = _score_ela(ela_map, block_size=block_size)

    # Scale flagged region coordinates back to the ORIGINAL image's pixel
    # space if we downscaled for analysis.
    if scale != 1.0:
        for r in ela_regions:
            r.x, r.y, r.w, r.h = (round(r.x * scale), round(r.y * scale),
                                   round(r.w * scale), round(r.h * scale))

    # --- FFT ---
    fft_magnitude = _run_fft(image)
    fft_spike_score = _score_fft(fft_magnitude)
    fft_regions = _flag_fft_region_note(fft_spike_score)

    # --- Editor metadata ---
    editor_score = _check_editor_metadata(image_path)

    # --- Blend into a single forgery_probability (0-1, higher = more fake) ---
    # ELA is the most specific/localized signal so it carries the most
    # weight; FFT and the metadata check support it.
    forgery_probability = round(
        min(1.0, 0.55 * ela_anomaly_score + 0.25 * fft_spike_score + 0.20 * editor_score),
        4,
    )

    tampering_score = round((1.0 - forgery_probability) * 100, 2)

    all_regions = ela_regions + fft_regions

    explanation = _build_explanation(ela_anomaly_score, fft_spike_score, editor_score, len(ela_regions))

    return TamperingResult(
        tampering_score=tampering_score,
        forgery_probability=forgery_probability,
        ela_anomaly_score=ela_anomaly_score,
        fft_spike_score=fft_spike_score,
        editor_metadata_score=editor_score,
        flagged_regions=all_regions,
        explanation=explanation,
    )


def _build_explanation(ela_score: float, fft_score: float, editor_score: float, n_regions: int) -> str:
    """Plain-English one-liner for an officer-facing dashboard."""
    parts = []
    if ela_score >= 0.15:
        parts.append(f"{n_regions} region(s) show abnormal compression error consistent with editing/splicing")
    if fft_score >= 0.15:
        parts.append("frequency spectrum shows periodic spikes consistent with re-capture or resampling")
    if editor_score >= 1.0:
        parts.append("image metadata shows it was saved from photo-editing software")
    if not parts:
        return "No significant compression, frequency, or metadata anomalies detected — image appears to be a single, unedited capture."
    return "; ".join(parts) + "."


# ---------------------------------------------------------------------------
# Standalone test — works with nothing else, no other files needed
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tampering_module.py <path_to_image>")
        sys.exit(1)

    try:
        result = analyze_tampering(sys.argv[1])
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    print(f"Tampering Score : {result.tampering_score}/100")
    print(f"Forgery Prob    : {result.forgery_probability}")
    print(f"ELA anomaly     : {result.ela_anomaly_score}")
    print(f"FFT spike       : {result.fft_spike_score}")
    print(f"Editor metadata : {result.editor_metadata_score}")
    print(f"Flagged regions : {len(result.flagged_regions)}")
    print(f"Explanation     : {result.explanation}")
