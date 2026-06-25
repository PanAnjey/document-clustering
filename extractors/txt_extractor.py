# extractors/txt_extractor.py
# Извлечение текста из TXT/CSV файлов

from pathlib import Path
from typing import Optional, Dict
import csv

from config import cfg
from logger_utils import logger, assess_text_quality


def extract_txt(file_path: Path) -> Dict:
    """Извлечение текста из TXT файла напрямую."""
    result = {
        "source": str(file_path),
        "type": "txt",
        "text": None,
        "image": None,
        "error": None,
        "text_quality": "none"
    }

    try:
        text = file_path.read_text(encoding='utf-8', errors='replace')
        if text and text.strip():
            result['text'] = text.strip()
            result['text_quality'] = assess_text_quality(result['text'])
        else:
            result['error'] = "Empty or extraction failed"
    except Exception as e:
        result['error'] = f"TXT read error: {e}"

    return result


def extract_csv(file_path: Path) -> Dict:
    """Извлечение текста из CSV файла напрямую."""
    result = {
        "source": str(file_path),
        "type": "csv",
        "text": None,
        "image": None,
        "error": None,
        "text_quality": "none"
    }

    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            reader = csv.reader(f)
            rows = []
            for row in reader:
                cleaned = [c.strip() for c in row if c.strip()]
                if cleaned:
                    rows.append(" | ".join(cleaned))
        text = "\n".join(rows)
        if text and text.strip():
            result['text'] = text.strip()
            result['text_quality'] = assess_text_quality(result['text'])
        else:
            result['error'] = "Empty or extraction failed"
    except Exception as e:
        result['error'] = f"CSV read error: {e}"

    return result
