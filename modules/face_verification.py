"""
Face Verification Module
-------------------------
Compares the face on the uploaded ID document photo against the face
captured in the live selfie/video, and returns a normalized match score
the Risk Engine can consume.

Primary engine : DeepFace (embedding distance)
Cross-check    : face_recognition (dlib) — optional, cheap, catches cases
                 where DeepFace's chosen model disagrees strongly.

Design notes for the prototype:
- We don't fine-tune anything here — both libraries ship pretrained models,
  which is exactly what the "keep as-is" section of the plan calls for.
- DeepFace.verify() already does face detection + alignment + embedding +
  distance threshold internally, so this module is mostly a thin,
  well-error-handled wrapper around it, plus score normalization so it
  fits the 0-1 scale the rest of the pipeline expects.
"""

from __future__ import annotations
import logging
from typing import Union
import numpy as np
from schemas import FaceVerificationResult

logger = logging.getLogger("face_verification")

# Accepts either a file path (upload, hardcoded test file) or a raw BGR
# frame straight out of cv2.VideoCapture (webcam / extracted video frame).
# This is the whole trick to keeping this module stable regardless of
# how the frontend team ends up wiring image capture: main.py always
# hands this module one of these two things, never anything more specific.
ImageInput = Union[str, np.ndarray]

# DeepFace model choice: "ArcFace" — generally more accurate than the
# default "VGG-Face", worth the extra weight download for better
# discrimination on borderline cases like ID-photo-vs-selfie comparisons.
MODEL_NAME = "ArcFace"
DISTANCE_METRIC = "cosine"

# Detector backend: "opencv" (DeepFace's default) needs a Haar cascade
# data file that isn't reliably bundled with every opencv-python build/
# version, which shows up as a confusing "haarcascade...xml violated"
# error that has nothing to do with your images. "mtcnn" is a deep-
# learning-based detector, ships as a DeepFace dependency already, and
# sidesteps that file dependency entirely — more reliable across
# machines for a hackathon prototype.
DETECTOR_BACKEND = "mtcnn"


def verify_face(
    document_image_path: ImageInput,
    selfie_image_path: ImageInput,
) -> FaceVerificationResult:
    """
    Compare the face in the ID document photo to the face in the live selfie.

    Both arguments accept either a file path (str) or a raw BGR frame
    (numpy array) — DeepFace handles both natively, so it doesn't matter
    whether the caller uploaded a file, grabbed a webcam frame, or
    pointed at a hardcoded test image. That choice lives entirely
    outside this module.

    Note: this deliberately does NOT pass anti_spoofing to DeepFace's
    own verify(). DeepFace's built-in anti_spoofing checks BOTH images
    for "is this a live capture" — but the document image is *supposed*
    to be a static photo, not a live human in front of a camera, so it
    gets (correctly, but unhelpfully) flagged as fake and crashes the
    whole match. Anti-spoofing only makes sense on the selfie side,
    which is what check_selfie_is_real() is for — called separately, on
    the selfie only, by verify_face_with_cross_check() below.

    Returns a FaceVerificationResult. Never raises — on any failure
    (no face found, corrupt image, etc.) it returns a result with
    match=False and a match_score of 0.0, and logs the reason, so a
    single bad frame can't crash the whole pipeline.
    """
    try:
        from deepface import DeepFace

        result = DeepFace.verify(
            img1_path=document_image_path,
            img2_path=selfie_image_path,
            model_name=MODEL_NAME,
            distance_metric=DISTANCE_METRIC,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=True,   # raises if no face is found — we want that
        )

        distance = float(result["distance"])
        threshold = float(result["threshold"])
        match = bool(result["verified"])

        match_score = _distance_to_score(distance, threshold)

        return FaceVerificationResult(
            match=match,
            distance=distance,
            match_score=match_score,
            model_used=MODEL_NAME,
        )

    except ValueError as e:
        logger.warning(f"Face verification failed - no face detected: {e}")
        return FaceVerificationResult(
            match=False, distance=1.0, match_score=0.0, model_used=MODEL_NAME
        )
    except Exception as e:
        logger.error(f"Face verification failed unexpectedly: {e}")
        return FaceVerificationResult(
            match=False, distance=1.0, match_score=0.0, model_used=MODEL_NAME
        )


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def check_selfie_is_real(selfie_image_path: ImageInput) -> dict:
    """
    Standalone anti-spoofing check on just the selfie, independent of
    face matching. Useful for surfacing is_real / antispoofing_score in
    the officer dashboard even when the identity check itself passes,
    and for feeding into the Risk Engine's liveness signal alongside
    (or instead of, for a fast prototype) the rPPG check.
    """
    if not _torch_available():
        logger.warning("Skipping anti-spoofing check — torch isn't installed")
        return {"is_real": None, "antispoofing_score": None}

    try:
        from deepface import DeepFace

        faces = DeepFace.extract_faces(
            img_path=selfie_image_path,
            detector_backend=DETECTOR_BACKEND,
            anti_spoofing=True,
            enforce_detection=True,
        )
        if not faces:
            return {"is_real": None, "antispoofing_score": None}

        face = faces[0]
        return {
            "is_real": bool(face.get("is_real")),
            "antispoofing_score": float(face.get("antispoofing_score", 0.0)),
        }
    except Exception as e:
        logger.warning(f"Anti-spoofing check on selfie failed: {e}")
        return {"is_real": None, "antispoofing_score": None}


def _distance_to_score(distance: float, threshold: float) -> float:
    """
    Turn a raw embedding distance into a 0-1 "match_score" where
    1.0 = identical, 0.0 = at-or-beyond DeepFace's own verification
    threshold. This keeps the number meaningful for the Risk Engine
    instead of an unbounded distance value.
    """
    if threshold <= 0:
        return 0.0
    score = 1.0 - (distance / threshold)
    return max(0.0, min(1.0, score))


def cross_check_with_face_recognition(document_image_path: ImageInput, selfie_image_path: ImageInput) -> float | None:
    """
    Optional lightweight second opinion using face_recognition (dlib).
    Returns a 0-1 similarity score, or None if either image has no
    detectable face (so the caller can just fall back to DeepFace alone).

    Accepts a path or an in-memory BGR frame, same as verify_face.

    Use this when you want to flag cases where DeepFace and dlib disagree
    strongly — a decent tell that something's off with the image, not
    just the person.
    """
    try:
        doc_img = _load_rgb(document_image_path)
        selfie_img = _load_rgb(selfie_image_path)

        import face_recognition

        doc_encodings = face_recognition.face_encodings(doc_img)
        selfie_encodings = face_recognition.face_encodings(selfie_img)

        if not doc_encodings or not selfie_encodings:
            logger.warning("face_recognition cross-check: no face found in one or both images")
            return None

        # face_distance: 0 = identical, ~0.6 is the typical match threshold
        face_distance = face_recognition.face_distance([doc_encodings[0]], selfie_encodings[0])[0]
        score = max(0.0, min(1.0, 1.0 - (face_distance / 0.6)))
        return float(score)

    except Exception as e:
        logger.warning(f"face_recognition cross-check unavailable: {e}")
        return None


def _load_rgb(image: ImageInput) -> np.ndarray:
    """Normalizes a path or a raw BGR frame into an RGB array for face_recognition."""
    if isinstance(image, str):
        import face_recognition
        return face_recognition.load_image_file(image)  # already RGB
    return image[:, :, ::-1]  # assume BGR (cv2 convention) -> RGB


def verify_face_with_cross_check(document_image_path: ImageInput, selfie_image_path: ImageInput) -> dict:
    """
    Convenience wrapper for the demo: runs DeepFace as primary, adds the
    dlib cross-check score, and flags a disagreement if the two differ
    by more than DISAGREEMENT_THRESHOLD. This is the function main.py
    calls in practice.
    """
    DISAGREEMENT_THRESHOLD = 0.35

    primary = verify_face(document_image_path, selfie_image_path)
    cross_check_score = cross_check_with_face_recognition(document_image_path, selfie_image_path)

    disagreement_flag = False
    if cross_check_score is not None:
        disagreement_flag = abs(primary.match_score - cross_check_score) > DISAGREEMENT_THRESHOLD

    # Anti-spoofing runs on the selfie ONLY, always separately from the
    # match check above — see verify_face()'s docstring for why.
    spoof_check = check_selfie_is_real(selfie_image_path)
    primary.is_real = spoof_check["is_real"]
    primary.antispoofing_score = spoof_check["antispoofing_score"]

    return {
        "primary": primary,
        "cross_check_score": cross_check_score,
        "disagreement_flag": disagreement_flag,
    }


def extract_face_crop(image_path: str, save_path: str | None = None):
    """
    Optional helper — not required for verification (verify() detects
    faces internally). Useful for showing "here's the face we found" on
    the frontend, or for debugging a bad detection.

    Returns the cropped face as a numpy array (RGB, 0-1 float), and
    writes it to save_path if given. Returns None if no face is found.
    """
    try:
        from deepface import DeepFace
        import numpy as np
        import cv2

        faces = DeepFace.extract_faces(img_path=image_path, enforce_detection=True)
        if not faces:
            return None

        face_arr = faces[0]["face"]  # RGB float array, 0-1 range

        if save_path:
            bgr = cv2.cvtColor((face_arr * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, bgr)

        return face_arr
    except Exception as e:
        logger.warning(f"Face crop extraction failed: {e}")
        return None



if __name__ == "__main__":
    # Quick manual test:
    #   python modules/face_verification.py path/to/id_photo.jpg path/to/selfie.jpg
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) != 3:
        print("Usage: python face_verification.py <document_image> <selfie_image>")
        sys.exit(1)

    out = verify_face_with_cross_check(sys.argv[1], sys.argv[2])
    print(out["primary"].model_dump())
    print("cross_check_score:", out["cross_check_score"])
    print("disagreement_flag:", out["disagreement_flag"])