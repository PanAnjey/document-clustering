# phase_6a_sample_extract.py
# Выборка для контрактной экстракции: 10k PDF_Tables (фин.первичка) +
# 10k PDF_Text — переизвлечение текста по правилу «первая + последняя страница».
#
# Таблица: temp_contract_files (id SERIAL PK, file_path UNIQUE, source, txt)

import sys
import time
import traceback
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

from experiment_config import DB_URL  # noqa: sys.path+utf8
from phase_5v_validate_contract import _DOC_FILTER_SQL

MAX_LEN = 3000
N_SAMPLE = 10000


def extract_first_last(path: Path):
    """Текст первой и последней страниц через fitz; None при ошибке."""
    try:
        import fitz
        doc = fitz.open(str(path))
        try:
            parts = []
            if doc.page_count > 0:
                t = doc[0].get_text('text')
                if t and t.strip():
                    parts.append(t.strip())
                if doc.page_count > 1:
                    t = doc[doc.page_count - 1].get_text('text')
                    if t and t.strip():
                        parts.append(t.strip())
            text = '\n\n'.join(parts).strip()
            text = ''.join(c for c in text if c == '\n' or c == '\t' or ord(c) >= 0x20)
            return text[:MAX_LEN] if text else None
        finally:
            doc.close()
    except Exception:
        return None


def sample_ids(conn, table, subjects, where_sql, limit):
    """Случайные (id, file_path) из files-таблицы с опциональным фильтром ТИП."""
    with conn.cursor() as cur:
        if subjects:
            cur.execute(f"""
                SELECT f.id, f.file_path FROM {table} f
                JOIN {subjects} s ON s.train_id = f.id
                WHERE s.tip IS NOT NULL AND ({where_sql})
                ORDER BY random() LIMIT {int(limit)}
            """)
        else:
            cur.execute(f"""
                SELECT id, file_path FROM {table}
                ORDER BY random() LIMIT {int(limit)}
            """)
        return cur.fetchall()


def main():
    conn = psycopg2.connect(DB_URL)
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS temp_contract_files")
        cur.execute("""
            CREATE TABLE temp_contract_files (
                id SERIAL PRIMARY KEY,
                file_path TEXT NOT NULL UNIQUE,
                source TEXT NOT NULL,
                txt TEXT NOT NULL
            )""")
    conn.commit()

    samples = sample_ids(conn, 'temp_pdftables_files',
                         'temp_llm_subject_pdftables', _DOC_FILTER_SQL, N_SAMPLE)
    print(f'PDF_Tables (фин.первичка): {len(samples)}')
    samples += [(i, p) for i, p in sample_ids(conn, 'temp_pdftext_files',
                                              'temp_llm_subject_pdftext',
                                              _DOC_FILTER_SQL, N_SAMPLE)]
    print(f'+ PDF_Text (фин.первичка): всего {len(samples)}')

    batch, failed = [], 0
    t0 = time.time()
    for i, (fid, fp) in enumerate(samples, 1):
        txt = extract_first_last(Path(fp))
        if txt:
            batch.append((fp, 'tables' if 'PDF_Tables' in fp else 'text', txt))
        else:
            failed += 1
        if len(batch) >= 500:
            with conn.cursor() as cur:
                execute_values(cur,
                    "INSERT INTO temp_contract_files (file_path, source, txt) VALUES %s",
                    batch, page_size=500)
            conn.commit()
            batch.clear()
        if i % 2000 == 0 or i == len(samples):
            dt = time.time() - t0
            print(f'\r  {i}/{len(samples)} ({i / dt:.0f} f/s), failed={failed}   ',
                  end='', flush=True)
    if batch:
        with conn.cursor() as cur:
            execute_values(cur,
                "INSERT INTO temp_contract_files (file_path, source, txt) VALUES %s",
                batch, page_size=500)
        conn.commit()
    conn.close()
    print(f'\n✅ temp_contract_files: {len(samples) - failed} ok, {failed} failed, '
          f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
