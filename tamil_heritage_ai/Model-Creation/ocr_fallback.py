# ocr_fallback.py
"""
OCR Fallback using Tesseract with custom Tamil.traineddata.

Tamil.traineddata lives at:
  <project>/tamil_heritage_ai/Tamil.traineddata

Tesseract requires the tessdata directory (parent of Tamil.traineddata),
so we pass --tessdata-dir pointing there.
"""
import cv2
import numpy as np
import os

try:
    import pytesseract
    _TESS_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if os.path.exists(_TESS_EXE):
        pytesseract.pytesseract.tesseract_cmd = _TESS_EXE
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False

# --- Resolve tessdata directory containing Tamil.traineddata ---
def _find_tessdata_dir() -> str | None:
    """
    Walk up from this file to find tamil_heritage_ai/ which contains Tamil.traineddata.
    Returns the directory path so Tesseract can use --tessdata-dir.
    """
    this_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        this_dir,                                   # Model-Creation/
        os.path.dirname(this_dir),                  # tamil_heritage_ai/
        os.path.join(this_dir, ".."),               # one level up
        os.path.join(this_dir, "..", ".."),         # two levels up
    ]
    for candidate in candidates:
        candidate = os.path.normpath(candidate)
        if os.path.isfile(os.path.join(candidate, "Tamil.traineddata")):
            return candidate
    return None

TESSDATA_DIR = _find_tessdata_dir()

def run_ocr_fallback(image_bgr_or_gray: np.ndarray, lang: str = "Tamil") -> dict:
    """
    Runs Tesseract OCR on a single character/region ROI.

    Parameters
    ----------
    image_bgr_or_gray : np.ndarray
        BGR or grayscale image.
    lang : str
        Tesseract language tag. Defaults to 'Tamil' (matches Tamil.traineddata).
        Falls back to 'tam' (built-in) if custom data not found.

    Returns
    -------
    dict  {"ok": bool, "text": str}  or  {"ok": False, "error": str}
    """
    if not HAS_TESSERACT:
        return {"ok": False, "error": "pytesseract not installed. Run: pip install pytesseract"}

    # Convert to RGB for pytesseract
    if len(image_bgr_or_gray.shape) == 2:
        img_rgb = cv2.cvtColor(image_bgr_or_gray, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(image_bgr_or_gray, cv2.COLOR_BGR2RGB)

    # Strategy 1: custom Tamil.traineddata
    if TESSDATA_DIR and lang == "Tamil":
        try:
            custom_cfg = f'--tessdata-dir "{TESSDATA_DIR}" --psm 10 -l Tamil'
            text = pytesseract.image_to_string(img_rgb, config=custom_cfg).strip()
            return {"ok": True, "text": text, "engine": "custom_Tamil.traineddata"}
        except Exception as e1:
            pass  # fall through to next strategy

    # Strategy 2: built-in 'tam' language
    try:
        text = pytesseract.image_to_string(img_rgb, lang="tam", config="--psm 10").strip()
        return {"ok": True, "text": text, "engine": "builtin_tam"}
    except Exception as e2:
        pass

    # Strategy 3: bare call (no lang)
    try:
        text = pytesseract.image_to_string(img_rgb, config="--psm 10").strip()
        return {"ok": True, "text": text, "engine": "default"}
    except Exception as e3:
        return {"ok": False, "error": f"All Tesseract strategies failed. Last error: {e3}"}

