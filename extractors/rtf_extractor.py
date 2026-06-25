# extractors/rtf_extractor.py
# Извлечение текста из RTF файлов через Pandoc (основной) + striprtf (fallback)

from pathlib import Path
from typing import Optional, Dict

from config import cfg
from logger_utils import logger, assess_text_quality


def extract_rtf(file_path: Path) -> Dict:
    """Извлечение текста из RTF файла через Pandoc (основной метод)."""
    
    base_result = {
        "source": str(file_path),
        "type": "rtf",
        "text": None,
        "image": None,
        "error": None,
        "text_quality": "none"
    }
    
    # Пытаемся использовать Pandoc как основной метод
    if cfg.PANDOC_ENABLED:
        from .pandoc_extractor import extract_with_pandoc
        result = extract_with_pandoc(file_path, "rtf")
        if result and result.get('text'):
            return result
    
    # Fallback на striprtf если Pandoc не сработал
    try:
        from striprtf.striprtf import rtf_to_text
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            rtf_content = f.read()
        text = rtf_to_text(rtf_content)
        if text and text.strip():
            base_result['text'] = text.strip()
            base_result['text_quality'] = assess_text_quality(text)
            return base_result
        
        base_result['error'] = "Empty or extraction failed"
        return base_result
    except ImportError:
        logger.debug("striprtf not installed, skipping rtf fallback")
        base_result['error'] = "striprtf not available"
        return base_result
    except Exception as e:
        logger.warning(f"striprtf extraction failed for {file_path.name}: {e}")
        base_result['error'] = f"RTF extraction error: {e}"
        return base_result
