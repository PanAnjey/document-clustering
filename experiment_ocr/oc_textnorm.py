# oc_textnorm.py
# Нормализация текста для честного сравнения OCR с эталоном:
# регистр и переносы строк у OCR и fitz различаются по построению,
# поэтому: lowercase, ё→е, схлопывание всех пробельных последовательностей.

import re

_WS_RE = re.compile(r'\s+')
_CYR_RE = re.compile(r'[а-яА-ЯёЁ]')
_TAG_RE = re.compile(r'<[^>]+>')          # html-теги (VLM-OCR: <table>, <img>…)
_PIPE_RE = re.compile(r'[|#]+')           # markdown-таблицы/заголовки
_DECOR_RE = re.compile(r'[-=_*]{3,}')     # разделители markdown (---, ===)


def normalize(text: str) -> str:
    t = (text or '').lower().replace('ё', 'е')
    t = _TAG_RE.sub(' ', t)      # для tesseract/paddle — no-op
    t = _PIPE_RE.sub(' ', t)
    t = _DECOR_RE.sub(' ', t)
    return _WS_RE.sub(' ', t).strip()


def words(text: str) -> list:
    return normalize(text).split(' ')


def cyrillic_ratio(text: str) -> float:
    letters = [c for c in (text or '') if c.isalpha()]
    if not letters:
        return 0.0
    cyr = sum(1 for c in letters if _CYR_RE.match(c))
    return cyr / len(letters)
