# engines/oc_tesseract.py
# Движок Tesseract (текущий стек проекта): rus+eng, psm 3.

import pytesseract
from PIL import Image

from oc_config import TESSERACT_PATH, TESSERACT_LANG


class TesseractEngine:
    name = 'tesseract'

    def __init__(self):
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

    def ocr(self, png_path: str) -> str:
        img = Image.open(png_path)
        return pytesseract.image_to_string(
            img, lang=TESSERACT_LANG, config='--psm 3').strip()


def get_engine():
    return TesseractEngine()
