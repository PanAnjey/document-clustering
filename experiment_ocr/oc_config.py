# oc_config.py
# Конфигурация пилота OCR-движков (experiment_ocr).
#
# Цель: выбрать OCR для проектируемого пайплайна (русскоязычные сканы,
# таблицы). Сравниваются:
#   tesseract — текущий движок проекта (rus+eng);
#   paddle    — PaddleOCR (PP-OCRv5, lang='ru', CPU).
# Структура расширяемая: новый движок = модуль в engines/ с классом
# {name}Engine и методом ocr(png_path) -> str (DeepSeek-OCR, Nanonets...).
#
# Методология: 200 pdf_text (эталон = fitz get_text 1-й страницы) +
# 10 pdf_scan (без эталона, для глазной проверки). Рендер 300 dpi.

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from config import cfg  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
RENDER_DIR = BASE_DIR / 'renders'
RESULTS_DIR = BASE_DIR / 'results'
REPORTS_DIR = BASE_DIR / 'reports'
RESULTS_JSONL = RESULTS_DIR / 'o1_results.jsonl'

DB_URL = (
    f"postgresql://{cfg.PG_USER}:{cfg.PG_PASSWORD}"
    f"@{cfg.PG_HOST}:{cfg.PG_PORT}/{cfg.PG_DB}"
)

N_PDF_TEXT = 200      # документов с эталонным текстом (pdf_text)
N_PDF_SCAN = 10       # реальных сканов для глазной проверки (pdf_scan)
DPI = 300
MIN_GT_CHARS = 100    # минимум символов эталона (как в experiment_models m5)

TESSERACT_PATH = cfg.TESSERACT_PATH
TESSERACT_LANG = getattr(cfg, 'TESSERACT_LANG', 'rus+eng')

ENGINES = ['tesseract', 'paddle', 'deepseek', 'nanonets', 'qwen3vl', 'qwen3vl4b',
           'deepseek2', 'gemma4']
