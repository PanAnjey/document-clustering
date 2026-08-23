"""Гибридный вход модели ориентации: кроп текста (для документов) ИЛИ всё
изображение (для не-текста: чертежи, фото, паспорта).

Зачем: ориентация определяется НАПРАВЛЕНИЕМ текста (для документов) либо ГЛОБАЛЬНОЙ
структурой (рамка чертежа, лицо/MRZ паспорта, гравитация фото). У документов символы
на ужатой целой странице нечитаемы, поэтому берём текст-плотное окно из ПОЛНОГО
разрешения. Но у не-текста читать нечего — для таких изображений кроп бессмыслен,
и мы подаём всё изображение (letterbox).

Различие «текст / не-текст» — дешёвая эвристика по числу «буквенных» connected
components (НЕ нужен отдельный классификатор типов). Логика ОДИНАКОВА в train /
5_retrain / 3_test_inference, чтобы вход совпадал на обучении и инференсе.

Поиск ведётся на уменьшенной копии (скорость), кроп берётся из полного разрешения.
"""
from __future__ import annotations

import random

import cv2
import numpy as np
from PIL import Image

import config as C


def _binarize_text(gray: np.ndarray) -> np.ndarray:
    """Инвертированная адаптивная бинаризация: текст/тёмное = 1, фон = 0."""
    h, w = gray.shape
    bs = 31
    if bs >= min(h, w):
        bs = max(3, (min(h, w) // 2))
    if bs % 2 == 0:
        bs += 1
    return cv2.adaptiveThreshold(gray, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, bs, 10)


def analyze_text(binv: np.ndarray) -> tuple[int, float]:
    """Вернуть (число «буквенных» компонент, медианную высоту символа в px binv).

    Компоненты — мелкие, не слишком большие, с текст-подобной заполненностью bbox.
    Документы дают сотни компонент; чертежи/фото/градиенты — единицы. Морфологическое
    открытие 2×2 подавляет соль-перец (зернистость/шум скана). Медианная высота нужна
    для адаптивного размера окна кропа (устойчивость к разному разрешению/DPI)."""
    b = cv2.morphologyEx(binv.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    h, w = b.shape
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(b, connectivity=8)
    heights = []
    for i in range(1, n):
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ar = stats[i, cv2.CC_STAT_AREA]
        fill = ar / float(max(1, ch * cw))
        if 3 <= ch <= 0.06 * h and 2 <= cw <= 0.30 * w and ar >= 8 and 0.15 <= fill <= 0.95:
            heights.append(ch)
    if not heights:
        return 0, 0.0
    return len(heights), float(np.median(heights))


def count_text_components(binv: np.ndarray) -> int:
    """Обёртка для совместимости: только число «буквенных» компонент."""
    return analyze_text(binv)[0]


def _center_from_density(binv: np.ndarray, win: int) -> tuple[int, int]:
    """Центр (cy, cx) самого текст-плотного окна win×win в координатах binv."""
    h, w = binv.shape
    win = max(1, min(win, h, w))
    dens = cv2.boxFilter(binv.astype(np.float32), -1, (win, win), normalize=False,
                         borderType=cv2.BORDER_CONSTANT)
    j = int(np.argmax(dens))
    return j // w, j % w


def letterbox_white(img: Image.Image, size: int) -> Image.Image:
    """Вписать всё изображение в квадрат size×size с сохранением пропорций + белые поля."""
    w, h = img.size
    scale = size / max(w, h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    r = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(r, ((size - nw) // 2, (size - nh) // 2))
    return canvas


def crop_text_region(img: Image.Image, window: int | None = None,
                     out_size: int | None = None, jitter: int = 0) -> Image.Image:
    """Гибридный вход: текст -> кроп текст-окна; не-текст -> всё изображение (letterbox).

    window  — размер окна в пикселях ПОЛНОГО разрешения (по умолчанию C.CROP_WINDOW).
    out_size— итоговый размер (по умолчанию C.IMAGE_SIZE).
    jitter  — случайный сдвиг центра окна (px) в TRAIN (0 в val/inference).
    """
    window = int(getattr(C, "CROP_WINDOW", 224)) if window is None else int(window)
    out_size = int(C.IMAGE_SIZE) if out_size is None else int(out_size)
    search_max = int(getattr(C, "CROP_SEARCH_MAXSIDE", 800))

    gray_full = np.asarray(img.convert("L"))
    h, w = gray_full.shape

    # уменьшенная копия для теста «текст/не-текст», оценки высоты символа и поиска центра
    s = search_max / float(max(h, w)) if max(h, w) > 0 else 1.0
    if s < 1.0:
        gs = cv2.resize(gray_full, (max(1, int(w * s)), max(1, int(h * s))),
                        interpolation=cv2.INTER_AREA)
    else:
        gs, s = gray_full, 1.0
    binv = _binarize_text(gs)
    ncomp, glyph_h_s = analyze_text(binv)

    # ветвь не-текста: мало «буквенных» компонент -> подаём всё изображение
    if getattr(C, "CROP_FALLBACK_FULL", True):
        if ncomp < int(getattr(C, "CROP_TEXT_MIN_COMPONENTS", 40)):
            return letterbox_white(img, out_size)

    # размер окна: адаптивный (под высоту символа -> устойчивость к разному DPI) либо фикс.
    if getattr(C, "CROP_ADAPTIVE", True) and glyph_h_s > 0:
        glyph_full = glyph_h_s / s   # высота символа в полном разрешении
        target = float(getattr(C, "CROP_TARGET_GLYPH", 14.0))
        win = int(round(glyph_full * out_size / max(1.0, target)))
    else:
        win = int(window)
    win = max(out_size, min(win, h, w))   # не меньше IMAGE_SIZE, не больше короткой стороны

    # ветвь текста: кроп текст-плотного окна из ПОЛНОГО разрешения
    cy_s, cx_s = _center_from_density(binv, max(1, int(round(win * s))))
    cy, cx = int(cy_s / s), int(cx_s / s)
    if jitter:
        cy += random.randint(-jitter, jitter)
        cx += random.randint(-jitter, jitter)

    half = win // 2
    y0 = int(min(max(cy - half, 0), h - win))
    x0 = int(min(max(cx - half, 0), w - win))
    crop = img.crop((x0, y0, x0 + win, y0 + win))
    if crop.size != (out_size, out_size):
        crop = crop.resize((out_size, out_size), Image.BILINEAR)
    return crop
