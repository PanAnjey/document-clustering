# extractors/aspose_extractor.py
# Извлечение текста из Word файлов через Aspose.Words for Java

import sys
from pathlib import Path
from typing import Any, Dict, Optional

_aspose: Optional[Any] = None
_ASPOSE_AVAILABLE = True


def _get_aspose():
    global _aspose, _ASPOSE_AVAILABLE
    if _aspose is not None:
        return _aspose
    try:
        from config import cfg
        aspose_dir = cfg.ASPOSE_WORDS_JAVA_DIR
        sys.path.insert(0, str(aspose_dir))
        from aspose_words import AsposeWords
        _aspose = AsposeWords()
        return _aspose
    except Exception as e:
        from logger_utils import logger
        logger.error(f"Aspose.Words for Java init failed: {e}")
        _ASPOSE_AVAILABLE = False
        raise


def extract_word_aspose(file_path: Path) -> Dict:
    """Извлечение текста из Word файла через Aspose.Words for Java."""
    result = {
        "source": str(file_path),
        "type": "word",
        "text": None,
        "image": None,
        "error": None,
        "text_quality": "none"
    }

    if not _ASPOSE_AVAILABLE:
        result['error'] = "Aspose.Words for Java not available"
        return result

    try:
        aw = _get_aspose()
        text = aw.extract_text(str(file_path))

        if text and len(text.strip()) > 0:
            from logger_utils import assess_text_quality
            result['text'] = text.strip()
            result['text_quality'] = assess_text_quality(result['text'])
        else:
            result['error'] = "Empty extraction"

    except Exception as e:
        from logger_utils import logger
        logger.error(f"Aspose.Words Java extraction error for {file_path.name}: {e}")
        result['error'] = str(e)

    return result
