import cv2
import easyocr
import pytesseract
import re


# =========================================================
# 1. CREATE EASY OCR READER
# =========================================================

reader = easyocr.Reader(['en'])


# =========================================================
# 2. PREPROCESS DOCUMENT
# =========================================================

def preprocess_image(image_path):

    # Read the image
    image = cv2.imread(image_path)

    # Check if image was loaded
    if image is None:
        raise ValueError("Could not open the image.")

    # Convert BGR colour image to grayscale
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # Remove noise
    denoised = cv2.fastNlMeansDenoising(
        gray
    )

    # Improve contrast
    enhanced = cv2.equalizeHist(
        denoised
    )

    return enhanced


# =========================================================
# 3. EASY OCR
# =========================================================

def easyocr_extract(image):

    results = reader.readtext(image)

    detections = []

    for result in results:

        bounding_box = result[0]
        text = result[1]
        confidence = result[2]

        detections.append({
            "text": text,
            "confidence": confidence,
            "bounding_box": bounding_box
        })

    return detections


# =========================================================
# 4. PYTESSERACT OCR
# =========================================================

def tesseract_extract(image):

    text = pytesseract.image_to_string(
        image
    )

    return text


# =========================================================
# 5. NORMALIZE TEXT
# =========================================================

def normalize_text(text):

    text = text.upper()

    text = re.sub(
        r'\s+',
        ' ',
        text
    )

    return text.strip()


# =========================================================
# 6. COMPARE OCR RESULTS
# =========================================================

def compare_ocr_results(
    easy_detections,
    tesseract_text
):

    easy_text = " ".join(
        detection["text"]
        for detection in easy_detections
    )

    easy_text = normalize_text(
        easy_text
    )

    tesseract_text = normalize_text(
        tesseract_text
    )

    if not easy_text or not tesseract_text:
        return 0

    easy_words = set(
        easy_text.split()
    )

    tesseract_words = set(
        tesseract_text.split()
    )

    common_words = (
        easy_words.intersection(
            tesseract_words
        )
    )

    agreement = (
        len(common_words)
        / len(easy_words)
    ) * 100

    return agreement


# =========================================================
# 7. EXTRACT ID
# =========================================================

def extract_id(full_text):

    id_pattern = (
        r'\b[A-Z]{2,5}\d{5,10}\b'
    )

    match = re.search(
        id_pattern,
        full_text.upper()
    )

    if match:
        return match.group()

    return None


# =========================================================
# 8. EXTRACT DATES
# =========================================================

def extract_dates(full_text):

    date_pattern = (
        r'\b\d{2}[/-]\d{2}[/-]\d{4}\b'
    )

    dates = re.findall(
        date_pattern,
        full_text
    )

    return dates


# =========================================================
# 9. EXTRACT NAME
# =========================================================

def extract_name(text_list):

    for i, text in enumerate(text_list):

        normalized = text.upper().strip()

        if normalized in [
            "NAME:",
            "NAME"
        ]:

            if i + 1 < len(text_list):

                return text_list[i + 1]

    return None


# =========================================================
# 10. EXTRACT ALL DOCUMENT FIELDS
# =========================================================

def extract_fields(detections):

    text_list = [
        detection["text"]
        for detection in detections
    ]

    full_text = " ".join(
        text_list
    )

    fields = {}

    # Extract ID
    document_id = extract_id(
        full_text
    )

    if document_id:
        fields["ID"] = document_id

    # Extract dates
    dates = extract_dates(
        full_text
    )

    if len(dates) >= 1:
        fields["Date_1"] = dates[0]

    if len(dates) >= 2:
        fields["Date_2"] = dates[1]

    # Extract name
    name = extract_name(
        text_list
    )

    if name:
        fields["Name"] = name

    return fields


# =========================================================
# 11. CALCULATE OCR CONFIDENCE
# =========================================================

def calculate_confidence(detections):

    if not detections:
        return 0

    total = sum(
        detection["confidence"]
        for detection in detections
    )

    average = (
        total / len(detections)
    ) * 100

    return average


# =========================================================
# 12. COMPLETE OCR PIPELINE
# =========================================================

def process_document(image_path):

    print("\nProcessing document...\n")

    # -----------------------------------------
    # STEP 1: PREPROCESS
    # -----------------------------------------

    processed_image = preprocess_image(
        image_path
    )

    # -----------------------------------------
    # STEP 2: EASY OCR
    # -----------------------------------------

    easy_detections = easyocr_extract(
        processed_image
    )

    # -----------------------------------------
    # STEP 3: PYTESSERACT
    # -----------------------------------------

    tesseract_text = tesseract_extract(
        processed_image
    )

    # -----------------------------------------
    # STEP 4: OCR CONFIDENCE
    # -----------------------------------------

    confidence = calculate_confidence(
        easy_detections
    )

    # -----------------------------------------
    # STEP 5: OCR AGREEMENT
    # -----------------------------------------

    agreement = compare_ocr_results(
        easy_detections,
        tesseract_text
    )

    # -----------------------------------------
    # STEP 6: EXTRACT FIELDS
    # -----------------------------------------

    fields = extract_fields(
        easy_detections
    )

    # -----------------------------------------
    # STEP 7: DISPLAY EASY OCR RESULTS
    # -----------------------------------------

    print("====================================")
    print("           EASY OCR RESULTS")
    print("====================================")

    for detection in easy_detections:

        print(
            f"Text: {detection['text']}"
        )

        print(
            f"Confidence: "
            f"{detection['confidence']:.2f}"
        )

        print(
            f"Bounding Box: "
            f"{detection['bounding_box']}"
        )

        print("------------------------------------")

    # -----------------------------------------
    # STEP 8: DISPLAY TESSERACT RESULTS
    # -----------------------------------------

    print("\n====================================")
    print("         PYTESSERACT RESULTS")
    print("====================================")

    print(tesseract_text)

    # -----------------------------------------
    # STEP 9: DISPLAY SCORES
    # -----------------------------------------

    print(
        f"\nEasyOCR Confidence: "
        f"{confidence:.2f}%"
    )

    print(
        f"OCR Agreement: "
        f"{agreement:.2f}%"
    )

    # -----------------------------------------
    # STEP 10: DISPLAY FIELDS
    # -----------------------------------------

    print("\n====================================")
    print("         EXTRACTED FIELDS")
    print("====================================")

    for key, value in fields.items():

        print(
            f"{key}: {value}"
        )

    # -----------------------------------------
    # STEP 11: RETURN STRUCTURED RESULT
    # -----------------------------------------

    return {

        "easyocr": easy_detections,

        "pytesseract": tesseract_text,

        "ocr_confidence": confidence,

        "ocr_agreement": agreement,

        "fields": fields
    }


# =========================================================
# 13. PROGRAM START
# =========================================================

if __name__ == "__main__":

    image_path = "document.jpg"

    result = process_document(
        image_path
    )
