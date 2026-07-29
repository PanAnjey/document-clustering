"""Анализ качества извлечения текста из PDF первой страницы.

Определяет документы, где PyMuPDF возвращает «мусор» вместо читаемого текста
(шрифты без ToUnicode / private-use glyphs), и копирует их в отдельную папку
для ручного просмотра.
"""

from __future__ import annotations

import csv
import re
import shutil
import sys
from pathlib import Path
from typing import List

import fitz

PROJECT_ROOT = Path(r"D:\Yandex.Disk\PYTHON\NLTK\Кластеризация")
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.fin_tables.config import cfg
from experiments.fin_tables.run_experiment import _sample_files

LETTER_RX = re.compile(r"[А-Яа-яЁёA-Za-z]")
PVT_RX = re.compile(r"[\ue000-\uf8ff]")  # Private Use Area glyphs
REPLACE_RX = re.compile(r"\ufffd")
DIGIT_RX = re.compile(r"\d")

OUT_DIR = cfg.WORK_DIR / "text_quality_analysis"
GARBLED_DIR = OUT_DIR / "garbled_pdfs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(GARBLED_DIR).mkdir(parents=True, exist_ok=True)


def first_page_text(pdf_path: Path) -> str:
    doc = fitz.open(str(pdf_path))
    try:
        if doc.page_count == 0:
            return ""
        return doc[0].get_text("text")
    finally:
        doc.close()


def analyze_text(text: str) -> dict:
    nonspace = [ch for ch in text if not ch.isspace()]
    total = len(nonspace)
    if total == 0:
        return {
            "total_chars": 0,
            "letters": 0,
            "letter_ratio": 0.0,
            "digits": 0,
            "private_use": 0,
            "replacement": 0,
            "garbled": False,
        }
    letters = len(LETTER_RX.findall(text))
    digits = len(DIGIT_RX.findall(text))
    pvt = len(PVT_RX.findall(text))
    repl = len(REPLACE_RX.findall(text))
    ratio = letters / total
    # «мусор»: мало букв (<30%) но есть символы, либо много private-use/замещающих
    garbled = (total > 50 and ratio < 0.30) or pvt > 20 or repl > 10
    return {
        "total_chars": total,
        "letters": letters,
        "letter_ratio": round(ratio, 3),
        "digits": digits,
        "private_use": pvt,
        "replacement": repl,
        "garbled": garbled,
    }


def main(sample_size: int = 1000) -> None:
    files = _sample_files(sample_size, seed=cfg.RANDOM_SEED)
    rows = []
    garbled_files: List[Path] = []

    for i, p in enumerate(files, 1):
        try:
            text = first_page_text(p)
        except Exception as exc:
            rows.append({
                "file": p.name,
                "path": str(p),
                "error": str(exc),
                "total_chars": 0,
                "letters": 0,
                "letter_ratio": 0.0,
                "digits": 0,
                "private_use": 0,
                "replacement": 0,
                "garbled": False,
                "preview": "",
            })
            continue
        m = analyze_text(text)
        preview = " ".join(text.split())[:300]
        row = {
            "file": p.name,
            "path": str(p),
            "error": "",
            **m,
            "preview": preview,
        }
        rows.append(row)
        if m["garbled"]:
            garbled_files.append(p)
        if i % 100 == 0:
            print(f"processed {i}/{len(files)}", flush=True)

    csv_path = OUT_DIR / "text_quality_report.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file", "path", "error", "total_chars", "letters", "letter_ratio",
            "digits", "private_use", "replacement", "garbled", "preview",
        ])
        writer.writeheader()
        writer.writerows(rows)

    # Copy garbled files (first 200 to avoid huge copies)
    copied = 0
    for p in garbled_files[:200]:
        try:
            shutil.copy2(p, GARBLED_DIR / p.name)
            copied += 1
        except Exception as exc:
            print(f"copy error {p.name}: {exc}", flush=True)

    summary_path = OUT_DIR / "summary.txt"
    total = len(rows)
    garbled_count = sum(1 for r in rows if r.get("garbled"))
    empty_count = sum(1 for r in rows if int(r.get("total_chars", 0)) == 0)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Всего файлов в выборке: {total}\n")
        f.write(f"\"Мусорный\" текст (letter_ratio<30% / private-use): {garbled_count}\n")
        f.write(f"Пустой текст (0 символов): {empty_count}\n")
        f.write(f"Скопировано в {GARBLED_DIR}: {copied}\n\n")
        # distribution of letter_ratio buckets
        buckets = {"0-10%": 0, "10-20%": 0, "20-30%": 0, "30-50%": 0, "50-70%": 0, "70-100%": 0}
        for r in rows:
            ratio = float(r.get("letter_ratio", 0))
            if ratio < 0.10:
                buckets["0-10%"] += 1
            elif ratio < 0.20:
                buckets["10-20%"] += 1
            elif ratio < 0.30:
                buckets["20-30%"] += 1
            elif ratio < 0.50:
                buckets["30-50%"] += 1
            elif ratio < 0.70:
                buckets["50-70%"] += 1
            else:
                buckets["70-100%"] += 1
        f.write("Распределение letter_ratio:\n")
        for k, v in buckets.items():
            f.write(f"  {k}: {v}\n")

    print(f"Готово. Отчёт: {csv_path}")
    print(f"Сводка: {summary_path}")
    print(f"Мусорные PDF скопированы в: {GARBLED_DIR} ( {copied} )")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    main(n)