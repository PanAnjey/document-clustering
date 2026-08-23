"""Оценка оптимального CROP_WINDOW под разрешение конкретных сканов.

Принцип: ориентация читается по символам, поэтому в финальном входе (IMAGE_SIZE)
символ должен остаться легибельным (~12-16 px высотой). Высота символа в оригинале
оценивается через connected components (медиана высоты «буквенных» компонент).

    CROP_WINDOW ≈ высота_символа_в_оригинале × IMAGE_SIZE / целевая_высота(≈14)

Ограничения: не меньше ~IMAGE_SIZE (нет смысла апскейлить крошечное окно) и не
больше короткой стороны скана.

Запуск:
    python estimate_crop_window.py                  # папка = INFERENCE_INPUT_DIR
    python estimate_crop_window.py "D:\\path\\to\\images"
    python estimate_crop_window.py "D:\\path" --sample 30 --target 14
"""
import argparse
import glob
import os
import random
import sys

import cv2
import numpy as np
from PIL import Image

import config as C

Image.MAX_IMAGE_PIXELS = None


def _glyph_height(path):
    g = np.asarray(Image.open(path).convert("L"))
    h, w = g.shape
    bs = 31
    if bs >= min(h, w):
        bs = max(3, (min(h, w) // 2))
    if bs % 2 == 0:
        bs += 1
    binv = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, bs, 10)
    n, _lab, stats, _c = cv2.connectedComponentsWithStats(binv, connectivity=8)
    hs = []
    for i in range(1, n):
        ch = stats[i, cv2.CC_STAT_HEIGHT]
        cw = stats[i, cv2.CC_STAT_WIDTH]
        ar = stats[i, cv2.CC_STAT_AREA]
        # отфильтровать шум и крупную графику; оставить «буквенные» компоненты
        if 6 <= ch <= 0.04 * h and 2 <= cw <= 0.04 * w and ar >= 8:
            hs.append(ch)
    return (h, w), (float(np.median(hs)) if hs else float("nan")), len(hs)


def main():
    ap = argparse.ArgumentParser(description="Оценка CROP_WINDOW под разрешение сканов.")
    ap.add_argument("folder", nargs="?", default=str(C.INFERENCE_INPUT_DIR),
                    help="папка с изображениями (по умолчанию INFERENCE_INPUT_DIR)")
    ap.add_argument("--sample", type=int, default=20, help="сколько файлов проанализировать")
    ap.add_argument("--target", type=float, default=14.0,
                    help="целевая высота символа в финальном входе, px (≈12..16)")
    args = ap.parse_args()

    files = [f for f in glob.glob(os.path.join(args.folder, "*"))
             if os.path.splitext(f)[1].lower() in C.IMAGE_EXTENSIONS]
    if not files:
        print(f"[ERROR] нет изображений в {args.folder}", file=sys.stderr)
        sys.exit(1)
    random.seed(1)
    random.shuffle(files)
    files = files[:args.sample]

    sides, gh = [], []
    for f in files:
        try:
            (h, w), m, cnt = _glyph_height(f)
            if m == m and cnt >= 30:
                sides.append(max(h, w))
                gh.append(m)
        except Exception:  # noqa: BLE001
            continue
    if not gh:
        print("[ERROR] не удалось оценить высоту символов (мало текста?).", file=sys.stderr)
        sys.exit(1)

    gh = np.array(gh)
    med_g = float(np.median(gh))
    med_side = float(np.median(sides))
    short_side = float(np.median([min(np.asarray(Image.open(f).size)) for f in files[:1]])) if files else med_side
    IMG = C.IMAGE_SIZE

    def clamp(x):
        x = round(x / 32) * 32                 # округление к кратному 32
        return int(min(max(x, IMG), round(med_side)))

    rec = clamp(med_g * IMG / args.target)
    pitch = med_g * 1.8
    print(f"Папка: {args.folder}")
    print(f"Файлов проанализировано: {len(gh)}")
    print(f"Медиана длинной стороны:  {med_side:.0f} px")
    print(f"Медиана высоты символа:   {med_g:.1f} px (в оригинале)")
    print(f"IMAGE_SIZE: {IMG}  целевая высота символа: {args.target:.0f} px")
    print(f"\n  >>> Рекомендуемый CROP_WINDOW = {rec}")
    print(f"      (в окно влезает ~{rec / pitch:.1f} строк; символ в финале ~{med_g * IMG / rec:.1f} px)")
    if rec <= IMG + 1:
        print("      Примечание: источник низкого разрешения - окно ~ native (без даунскейла); "
              "детальнее уже не сделать без более качественных сканов.")


if __name__ == "__main__":
    main()
