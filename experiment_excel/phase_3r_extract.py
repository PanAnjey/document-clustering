# phase_3r_extract.py
# Раунд 3, шаг 1: извлечение текста из PDF (fitz, первые 2 страницы).
#
# Для LLM-промпта ПРЕДМЕТ/ТИП достаточно первых страниц (конвенция проекта).
# Текст нормализуется и усекается до 3000 символов — как direct-TSV для Excel.
#
# Запуск:
#   python phase_3r_extract.py                          # PDF_Text → temp_pdftext_files
#   python phase_3r_extract.py --root D:\...\PDF_Tables --table temp_pdftables_files

import argparse
import sys
import time
import traceback
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

from experiment_config import DB_URL  # noqa: sys.path+utf8

MAX_PAGES = 2
MAX_LEN = 3000


def ensure_table(conn, table):
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {table}")
        cur.execute(f"""
            CREATE TABLE {table} (
                id SERIAL PRIMARY KEY,
                file_path TEXT NOT NULL UNIQUE,
                txt TEXT NOT NULL
            )
        """)
    conn.commit()


def extract_pdf_text(path: Path):
    """Текст первых MAX_PAGES страниц через fitz; None при ошибке/пусто.
    NUL и прочие управляющие символы вычищаются (psycopg2 их отвергает)."""
    try:
        import fitz
        doc = fitz.open(str(path))
        try:
            parts = []
            for i in range(min(MAX_PAGES, doc.page_count)):
                t = doc[i].get_text('text')
                if t and t.strip():
                    parts.append(t.strip())
            text = '\n\n'.join(parts).strip()
            text = ''.join(c for c in text if c == '\n' or c == '\t' or ord(c) >= 0x20)
            return text[:MAX_LEN] if text else None
        finally:
            doc.close()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=r'D:\FileOrganizer\Sorted\PDF_Text')
    ap.add_argument('--table', default='temp_pdftext_files')
    args = ap.parse_args()
    source_root = Path(args.root)
    table = args.table

    conn = psycopg2.connect(DB_URL)
    ensure_table(conn, table)

    files = sorted(source_root.rglob('*.pdf'))
    if not files:
        sys.exit(f'❌ Нет PDF в {source_root}')
    print(f'{source_root}: {len(files)} файлов')

    batch, failed = [], 0
    t0 = time.time()
    for i, p in enumerate(files, 1):
        txt = extract_pdf_text(p)
        if txt:
            batch.append((str(p), txt))
        else:
            failed += 1
        if len(batch) >= 500:
            with conn.cursor() as cur:
                execute_values(cur,
                    f"INSERT INTO {table} (file_path, txt) VALUES %s",
                    batch, page_size=500)
            conn.commit()
            batch.clear()
        if i % 2000 == 0 or i == len(files):
            dt = time.time() - t0
            rate = i / dt if dt else 0
            print(f'\r  {i}/{len(files)} ({rate:.0f} f/s), failed={failed}   ',
                  end='', flush=True)
    if batch:
        with conn.cursor() as cur:
            execute_values(cur,
                f"INSERT INTO {table} (file_path, txt) VALUES %s",
                batch, page_size=500)
        conn.commit()
    conn.close()
    print(f'\n✅ Extract {table}: {len(files) - failed} ok, {failed} failed, '
          f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
