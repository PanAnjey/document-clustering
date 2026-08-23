# phase_1_split_and_extract.py
import sys
import random
import re
import time
import traceback
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

# Local imports (experiment_config первым — он чинит sys.path)
from experiment_config import (
    SOURCE_ROOT, SOURCE_EXTENSIONS, DB_URL, MAX_EXTRACT_LEN,
    EXTRACT_MODE, WRITE_PDF_ARTIFACT,
)
from config import cfg
from extractors.cells_aspose_extractor import (
    convert_xlsx_to_pdf,
    truncate_pdf_to_first_pages,
    extract_text_from_pdf,
    extract_text_direct,
    shutdown_aspose_cells_server,
)

random.seed(42)

# Как в pipelines/excel_xlsx_pipeline.py: 1 лист → 1 стр., 2 → 2, >=3 → 2 первые.
MAX_PDF_PAGES = 2


def normalize_tsv(text: str) -> str:
    """Нормализация TSV-дампа: схлопнуть серии табов (пустые ячейки),
    убрать хвостовые табы/пробелы, выкинуть пустые строки.
    Экономит бюджет MAX_EXTRACT_LEN (широкие таблицы дают 40+ табов подряд)."""
    lines = []
    for ln in text.splitlines():
        ln = re.sub(r'\t{2,}', '\t', ln).strip('\t ')
        if ln:
            lines.append(ln)
    return '\n'.join(lines)


def ensure_tables(conn):
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS temp_excel_train")
        cur.execute("DROP TABLE IF EXISTS temp_excel_test")
        cur.execute("DROP TABLE IF EXISTS temp_excel_failed")
        cur.execute("""
            CREATE TABLE temp_excel_train (
                id SERIAL PRIMARY KEY,
                file_path TEXT NOT NULL,
                txt TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE temp_excel_test (
                id SERIAL PRIMARY KEY,
                file_path TEXT NOT NULL,
                txt TEXT NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE temp_excel_failed (
                id SERIAL PRIMARY KEY,
                file_path TEXT NOT NULL,
                reason TEXT NOT NULL
            )
        """)
    conn.commit()


def bulk_insert(conn, table, rows):
    if not rows:
        return
    with conn.cursor() as cur:
        execute_values(cur,
            f"INSERT INTO {table} (file_path, txt) VALUES %s",
            rows, page_size=500)
    conn.commit()


def bulk_insert_failed(conn, rows):
    if not rows:
        return
    with conn.cursor() as cur:
        execute_values(cur,
            "INSERT INTO temp_excel_failed (file_path, reason) VALUES %s",
            rows, page_size=500)
    conn.commit()


def extract_text_production(file_path: Path):
    """Извлечение текста для эмбеддингов.

    Режим 'direct' (основной): прямой TSV-дамп через Aspose.Cells .NET
    (extract_text_direct) — без PDF, без проблемы обрезания широких таблиц.
    При WRITE_PDF_ARTIFACT дополнительно генерируется PDF-артефакт
    (конвертер с FitToPagesWide=1) для визуального аудита в pdf_viewer.

    Режим 'pdf' (legacy): Aspose → PDF → первые 1-2 страницы → fitz.

    Fallback при ошибках: PDF-путь → COM Excel.

    Returns:
        (text, None) при успехе, (None, reason) при неудаче.
    """
    out_dir = cfg.EXTRA['excel_xlsx_pages']
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    pdf_path = out_dir / f"{file_path.stem}.pdf"

    # ---- Основной путь: direct text (TSV, без PDF) ----
    if EXTRACT_MODE == 'direct':
        try:
            text = extract_text_direct(file_path)
            if text:
                text = normalize_tsv(text)
            if text:
                # PDF-артефакт для viewer'а (ошибка PDF не валит извлечение)
                if WRITE_PDF_ARTIFACT and not pdf_path.exists():
                    try:
                        made = convert_xlsx_to_pdf(file_path, pdf_path)
                        if made and made.exists():
                            truncate_pdf_to_first_pages(made, max_pages=MAX_PDF_PAGES)
                    except Exception:
                        pass
                return text, None
        except Exception:
            pass  # fallback ниже

    # ---- Legacy/fallback путь: PDF → fitz text ----
    try:
        if not pdf_path.exists():
            pdf_path = convert_xlsx_to_pdf(file_path, pdf_path)  # None при ошибке
        if pdf_path and pdf_path.exists():
            truncate_pdf_to_first_pages(pdf_path, max_pages=MAX_PDF_PAGES)
            text = extract_text_from_pdf(pdf_path, max_pages=0)
            if text:
                return text, None
            # PDF создан, но текста нет — удаляем сироту (как pipeline)
            if EXTRACT_MODE != 'direct':
                try:
                    pdf_path.unlink()
                except Exception:
                    pass
    except Exception as e:
        return None, f"Aspose path error: {e}"

    # Fallback: COM Excel
    try:
        from extractors.excel_extractor import extract_excel
        res = extract_excel(file_path)
        if res.get('text'):
            return res['text'], None
        return None, res.get('error') or "Empty text (COM)"
    except Exception as e:
        return None, f"COM fallback error: {e}"


def process_files(conn, files, table, tag):
    """Извлечение текстов группы файлов и bulk-INSERT в указанную таблицу."""
    batch, failed = [], []
    n_ok = 0
    t0 = time.time()
    total = len(files)
    for i, p in enumerate(files, 1):
        text, err = extract_text_production(p)
        if err is None and text:
            batch.append((str(p), text[:MAX_EXTRACT_LEN]))
            n_ok += 1
        else:
            # Мягкие ошибки (error/пустой текст без исключения) тоже в failed
            failed.append((str(p), (err or "empty text")[:500]))
        if len(batch) >= 500:
            bulk_insert(conn, table, batch)
            batch.clear()
        if len(failed) >= 500:
            bulk_insert_failed(conn, failed)
            failed.clear()
        if i % 200 == 0 or i == total:
            dt = time.time() - t0
            rate = i / dt if dt > 0 else 0
            eta = (total - i) / rate if rate > 0 else 0
            print(f"\r  [{tag}] {i}/{total} ({rate:.1f} f/s, ETA {eta:.0f}s), "
                  f"ok={n_ok}, failed={i - n_ok}   ", end="", flush=True)
    print()
    bulk_insert(conn, table, batch)
    bulk_insert_failed(conn, failed)
    return n_ok


def main():
    conn = psycopg2.connect(DB_URL)
    ensure_tables(conn)

    # Папка смешанная: .xlsx И .xls (38.7% датасета!) — берём оба расширения
    all_files = sorted(
        p for p in SOURCE_ROOT.rglob('*')
        if p.is_file() and p.suffix.lower() in SOURCE_EXTENSIONS
    )
    if not all_files:
        sys.exit('❌ No Excel files found – abort.')
    random.shuffle(all_files)
    mid = len(all_files) // 2
    train_files, test_files = all_files[:mid], all_files[mid:]
    print(f"Found {len(all_files)} files ({', '.join(SOURCE_EXTENSIONS)}): "
          f"{len(train_files)} train / {len(test_files)} test")

    t0 = time.time()
    try:
        ok_train = process_files(conn, train_files, 'temp_excel_train', 'train')
        ok_test = process_files(conn, test_files, 'temp_excel_test', 'test')
    finally:
        # Закрыть persistent Aspose-сервер (JVM/.NET), чтобы не висел процесс
        try:
            shutdown_aspose_cells_server()
        except Exception:
            pass

    conn.close()
    dt = time.time() - t0
    print(f"✅ Phase 1 completed in {dt:.0f}s – "
          f"train ok={ok_train}/{len(train_files)}, test ok={ok_test}/{len(test_files)}")
    print("   Failures written to temp_excel_failed (if any).")


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
