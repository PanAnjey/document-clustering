# extractors/image_extractor.py
# Извлечение текста из изображений через Tesseract OCR

import os
from pathlib import Path
from typing import Optional, Dict

import pytesseract
from PIL import Image

from config import cfg
from logger_utils import logger, assess_text_quality


def extract_image(file_path: Path) -> Dict:
    """Извлечение текста из изображения через Tesseract OCR."""
    result = {
        "source": str(file_path),
        "type": "image",
        "text": None,
        "image": str(file_path),
        "error": None,
        "text_quality": "none"
    }

    try:
        pytesseract.pytesseract.tesseract_cmd = cfg.TESSERACT_PATH
        
        img = Image.open(str(file_path))
        
        text = pytesseract.image_to_string(img, lang=cfg.TESSERACT_LANG)
        
        if text and text.strip():
            result['text'] = text.strip()
            result['text_quality'] = assess_text_quality(result['text'])
        
        if not result['text']:
            result['error'] = "Empty or extraction failed"
            
    except Exception as e:
        result['error'] = f"Image extraction error: {e}"

    return result
