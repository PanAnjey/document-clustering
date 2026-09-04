# download_ocr_models.py
# Скачивание VLM-OCR моделей в D:\MODELS\Transformers (соглашение проекта).
# Запуск: python download_ocr_models.py  (обычный python, НЕ venv_ocr)

import os
import sys

os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from pathlib import Path  # noqa: E402

from huggingface_hub import snapshot_download  # noqa: E402

BASE = Path(r"D:\MODELS\Transformers")
MODELS = [
    'deepseek-ai/DeepSeek-OCR',
    'nanonets/Nanonets-OCR2-3B',
]

for name in MODELS:
    local = BASE / name.replace('/', '_')
    print(f"=== {name} -> {local}", flush=True)
    snapshot_download(name, local_dir=str(local))
    print(f"  done: {name}", flush=True)
print("all downloaded")
