# subject_extractor_excel.py
# Извлечение предметного фрагмента из TSV-дампа Excel (direct text).
#
# Мотивация: первые N символов документа — шапка (тип документа, реквизиты
# сторон, номера, даты). Эмбеддинг по ней кластеризует по ТИПАМ документов.
# Тематика (предмет договора/счёта) живёт в строках номенклатуры — она и
# нужна для тематической кластеризации.
#
# Переиспользует regex'ы проектного subject_extractor.py:
#   _SUBJECT_KEYWORDS_RX — строки-кандидаты номенклатуры,
#   _SKIP_RX             — мусор (реквизиты, подписи, итоги),
#   _clean_subject()     — очистка от цифр/единиц/реквизитов.
#
# Отличия от PDF-варианта: таблиц find_tables() в TSV нет — вместо этого
# из подходящей строки берётся самая «буквенная» ячейка (номенклатурное
# наименование), числовые ячейки (кол-во/цена/сумма) отбрасываются.

import re
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from subject_extractor import _SUBJECT_KEYWORDS_RX, _SKIP_RX, _clean_subject  # noqa: E402

_LETTERS_RX = re.compile(r'[а-яА-ЯёЁa-zA-Z]')
_TRAILING_NUMS_RX = re.compile(r'\s+\d[\d\s.,/]*$')


def _most_alphabetic_cell(line: str) -> str:
    """Самая «буквенная» ячейка TSV-строки (номенклатурное наименование)."""
    best, best_score = '', 0
    for cell in line.split('\t'):
        score = len(_LETTERS_RX.findall(cell))
        if score > best_score:
            best, best_score = cell, score
    # Хвостовые числа (кол-во, цена, сумма в той же ячейке)
    return _TRAILING_NUMS_RX.sub('', best).strip()


def extract_subject_excel(txt: str, max_items: int = 6, max_len: int = 600) -> Optional[str]:
    """Предметный фрагмент из TSV-дампа.

    Сканирует строки сверху вниз, берёт до max_items строк-кандидатов
    номенклатуры (ключевые слова _SUBJECT_KEYWORDS_RX, без мусора _SKIP_RX),
    из каждой — самую буквенную ячейку. Склейка чистится _clean_subject().

    Returns:
        фрагмент ~100-600 символов или None, если предмет не найден.
    """
    if not txt:
        return None
    items = []
    for line in txt.split('\n'):
        if len(line) < 8:
            continue
        if _SKIP_RX.search(line):
            continue
        if not _SUBJECT_KEYWORDS_RX.search(line):
            continue
        cell = _most_alphabetic_cell(line)
        if len(cell) >= 8:
            items.append(cell)
        if len(items) >= max_items:
            break
    if not items:
        return None
    subject = _clean_subject(' '.join(items))
    if len(subject) < 15:
        return None
    return subject[:max_len]
