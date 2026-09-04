"""Постобработка изображения после исправления ориентации:
deskew (выравнивание перекоса) и обрезка чёрных контуров по границам листа.

Дополнения к пайплайну (см. 3_test_inference.py): ПОСЛЕ поворота ориентации
применяется (опционально) deskew, затем (опционально) border crop.

Алгоритмы:
- deskew: проекционный метод. Бинаризация изображения -> для набора углов в
  диапазоне ±DESKEW_MAX_ANGLE поворачиваем на малый угол, считаем горизонтальную
  проекцию (сумма dark-пикселей по строкам) и её дисперсию. Угол с максимальной
  дисперсией — искомый (строки текста максимально «горизонтальны» → проекция
  «пикообразная», дисперсия максимальна). Используется быстрая реализация через
  центральный поворот вокруг центра масс для эффективности.
- border crop: бинаризация по порогу CROP_BORDERS_THRESHOLD; находим bbox области,
  светлее порога {= контент}, отступаем на margin и обрезаем.

Все функции принимают и возвращают PIL.Image (RGB), сохраняя прозрачность/режим.
"""
from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

import config as C


# ---------------- deskew ----------------
def _to_grayscale_arr(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("L"))


def _binarize(gray: np.ndarray) -> np.ndarray:
    """Инвертированная бинаризация: текст/тёмные элементы = 255, фон = 0.
    Используется адаптивный порог: хорошо для неравномерной засветки сканов."""
    # Adaptive threshold с большим blockSize на сканах
    h, w = gray.shape
    bs = 15
    while bs < min(h, w) // 4:
        bs += 2
    binv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, bs, 7)
    return binv


def _score_angle(binv: np.ndarray, angle_deg: float) -> float:
    """Повернуть бинаризованное изображение на -angle_deg (компенсируем skew),
    посчитать горизонтальную проекцию и её дисперсию."""
    h, w = binv.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    rot = cv2.warpAffine(binv, M, (w, h), flags=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    proj = np.sum(rot > 0, axis=1).astype(np.float32)
    if proj.size == 0:
        return 0.0
    mean = proj.mean()
    if mean < 1:
        return 0.0
    return float(np.var(proj) / (mean + 1e-6))


def deskew_image(img: Image.Image,
                 max_angle: float | None = None,
                 step: float | None = None) -> Tuple[Image.Image, float]:
    """Выровнять перекошенный лист. Возвращает (new_image, applied_angle_deg).

    applied_angle: угол, на который фактически повернули изображение (для отчёта).
    Знак: положительный = поворот против часовой (как в PIL.Image.rotate).
    """
    if not C.DESKEW_ENABLED:
        return img, 0.0
    if max_angle is None:
        max_angle = C.DESKEW_MAX_ANGLE
    if step is None:
        step = C.DESKEW_ANGLE_STEP

    gray = _to_grayscale_arr(img)
    # Поиск угла ведём на уменьшенной копии (угол перекоса — глобальная величина,
    # даунскейл не влияет на точность, но cv2.warpAffine в цикле кратно быстрее).
    maxside = getattr(C, "DESKEW_SEARCH_MAXSIDE", 1000)
    h, w = gray.shape
    scale = maxside / float(max(h, w)) if max(h, w) > 0 else 1.0
    if scale < 1.0:
        gray_small = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))),
                                interpolation=cv2.INTER_AREA)
    else:
        gray_small = gray
    binv = _binarize(gray_small)
    if binv.sum() == 0:
        return img, 0.0

    best_angle, best_score = 0.0, -1.0
    # первый проход — крупный шаг
    a = -max_angle
    while a <= max_angle:
        s = _score_angle(binv, a)
        if s > best_score:
            best_score, best_angle = s, a
        a += step if step >= 1.0 else 1.0  # в первом проходе не мельче 1°
    # локальное уточнение вокруг найденного
    fine = max(0.05, C.DESKEW_ANGLE_STEP)
    a = best_angle - 1.0
    while a <= best_angle + 1.0:
        s = _score_angle(binv, a)
        if s > best_score:
            best_score, best_angle = s, a
        a += fine

    # Если найденный угол пренебрежимо мал — не поворачиваем (чтобы не вносить шум)
    if abs(best_angle) < 1e-3:
        return img, 0.0
    # Если угол срывается к границе диапазона — скорее всего, это шум/сбой метода
    if abs(best_angle) >= max_angle - 1e-3:
        return img, 0.0

    rotated = img.rotate(best_angle, expand=True, resample=Image.BICUBIC,
                         fillcolor=(255, 255, 255))
    return rotated, float(best_angle)


# ---------------- border crop ----------------
def crop_borders(img: Image.Image) -> Tuple[Image.Image, bool]:
    """Обрезать тёмные/чёрные контуры по краям листа после поворота/дескеv.
    Возвращает (new_image, cropped_flag)."""
    if not C.CROP_BORDERS_ENABLED:
        return img, False

    gray = _to_grayscale_arr(img)
    h, w = gray.shape

    # Маска контента: пиксели СВЕТЛЕЕ порога (фон/контур - тёмнее)
    mask = gray > C.CROP_BORDERS_THRESHOLD
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return img, False

    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())

    # отступ безопасности внутрь
    m = C.CROP_BORDERS_MARGIN
    x0 = max(0, min(x0 - m, w - 1))
    y0 = max(0, min(y0 - m, h - 1))
    x1 = max(0, min(x1 + m, w - 1))
    y1 = max(0, min(y1 + m, h - 1))

    bbox_w = x1 - x0 + 1
    bbox_h = y1 - y0 + 1
    area_ratio = (bbox_w * bbox_h) / float(w * h)
    if area_ratio < C.CROP_BORDERS_MIN_AREA_RATIO:
        return img, False
    # если bbox почти не изменился — crop не нужен
    if bbox_w >= w - 4 and bbox_h >= h - 4:
        return img, False

    cropped = img.crop((x0, y0, x1 + 1, y1 + 1))
    return cropped, True


# ---------------- комплексная постобработка ----------------
def postprocess(img: Image.Image) -> dict:
    """Применить к уже ориентированному изображению весь набор постобработки.
    Возвращает dict с полями:
      - deskew_angle: float (градусы поворота для deskew)
      - cropped: bool (была ли обрезка границ)
    """
    info = {"deskew_angle": 0.0, "cropped": False}
    if C.DESKEW_ENABLED:
        img, ang = deskew_image(img)
        info["deskew_angle"] = ang
    if C.CROP_BORDERS_ENABLED:
        img, cropped = crop_borders(img)
        info["cropped"] = cropped
    return img, info