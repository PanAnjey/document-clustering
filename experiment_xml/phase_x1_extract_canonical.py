# phase_x1_extract_canonical.py
# Этап X1 эксперимента XML: MySQL events_26.xml → канонизация →
# PostgreSQL temp_xml_canonical.
#
# Метод: потоковое чтение из MySQL по PK-диапазонам (без индекса по
# TypeNamedId полный ORDER BY RAND() невозможен), канонизация в пуле
# процессов, батчевые INSERT (execute_values).
#
# Запуск:
#   python phase_x1_extract_canonical.py                 # полный объём (UTD+Invoice)
#   python phase_x1_extract_canonical.py --limit 2000    # отладочный прогон
#   python phase_x1_extract_canonical.py --types Invoice --limit 500

import argparse
import sys
import time
from multiprocessing import Pool

import psycopg2
import pymysql
from psycopg2.extras import execute_values

from experiment_xml_config import (  # noqa: E402
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB, MYSQL_TABLE,
    DOC_TYPES, DB_URL, T_CANONICAL,
    MAX_TABLE_ROWS, MAX_INFO_BLOCK_CHARS, MAX_OSNLEN, MARKED_N3_INDEX,
)
from xml_canonicalizer import canonicalize

MYSQL_CHUNK = 2000      # строк из MySQL за итерацию
PG_PAGE = 500           # строк за INSERT
WORKERS = 8

DDL = f"""
CREATE TABLE {T_CANONICAL} (
    id SERIAL PRIMARY KEY,
    message_id VARCHAR(36) NOT NULL,
    entity_id VARCHAR(36) NOT NULL,
    type_named_id VARCHAR(255),
    file_name VARCHAR(500),
    is_marked BOOLEAN NOT NULL DEFAULT FALSE,
    doc_function VARCHAR(10),
    doc_number VARCHAR(255),
    doc_date VARCHAR(20),
    seller_inn VARCHAR(20),
    seller_name TEXT,
    seller_region VARCHAR(255),
    seller_city VARCHAR(255),
    buyer_inn VARCHAR(20),
    buyer_name TEXT,
    sum_total VARCHAR(30),
    currency VARCHAR(50),
    gov_contract VARCHAR(60),
    osn TEXT,
    n_rows_total INTEGER,
    n_rows_taken INTEGER,
    src_len INTEGER,
    canon_len INTEGER,
    canonical_text TEXT,
    subject_text TEXT,
    subject_len INTEGER,
    parse_ok BOOLEAN NOT NULL,
    error TEXT,
    created_at TIMESTAMP DEFAULT now(),
    UNIQUE (message_id, entity_id)
)"""

INSERT_SQL = f"""
INSERT INTO {T_CANONICAL} (
    message_id, entity_id, type_named_id, file_name, is_marked, doc_function,
    doc_number, doc_date, seller_inn, seller_name, seller_region, seller_city,
    buyer_inn, buyer_name, sum_total, currency, gov_contract, osn,
    n_rows_total, n_rows_taken, src_len, canon_len, canonical_text,
    subject_text, subject_len, parse_ok, error
) VALUES %s
ON CONFLICT (message_id, entity_id) DO NOTHING"""


def _is_marked_fname(fname):
    """N3 == '1' в ИдФайл R_T_A_O_GGGGMMDD_N1_N2_N3_... → ГИС МТ."""
    if not fname:
        return False
    parts = fname.split('_')
    return len(parts) > MARKED_N3_INDEX and parts[MARKED_N3_INDEX] == '1'


def _canon_worker(row):
    mid, eid, tname, fname, xml = row
    r = canonicalize(xml, max_rows=MAX_TABLE_ROWS,
                     max_info_chars=MAX_INFO_BLOCK_CHARS,
                     max_osn_chars=MAX_OSNLEN)
    m = r['meta']
    return (
        mid, eid, tname, _trim(fname, 500), _is_marked_fname(fname),
        m.get('doc_function'),
        _trim(m.get('doc_number'), 255), _trim(m.get('doc_date'), 20),
        _trim(m.get('seller_inn'), 20), m.get('seller_name'),
        _trim(m.get('seller_region'), 255), _trim(m.get('seller_city'), 255),
        _trim(m.get('buyer_inn'), 20),
        m.get('buyer_name'), _trim(m.get('sum_total'), 30),
        _trim(m.get('currency'), 50), _trim(m.get('gov_contract'), 60),
        m.get('osn'), m.get('n_rows_total'), m.get('n_rows_taken'),
        len(xml or ''), len(r['canonical_text']),
        r['canonical_text'] if r['ok'] else None,
        r['subject_text'] if r['ok'] else None,
        len(r['subject_text']) if r['ok'] else None,
        r['ok'], _trim(r['error'], 500) or None,
    )


def _trim(s, n):
    if s is None:
        return None
    s = str(s)
    return s[:n] if len(s) > n else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='максимум документов (0 = все)')
    ap.add_argument('--types', nargs='+', default=list(DOC_TYPES))
    ap.add_argument('--workers', type=int, default=WORKERS)
    ap.add_argument('--keep', action='store_true',
                    help='не пересоздавать таблицу (дозаливка)')
    args = ap.parse_args()

    pg = psycopg2.connect(DB_URL)
    with pg.cursor() as cur:
        if not args.keep:
            cur.execute(f"DROP TABLE IF EXISTS {T_CANONICAL}")
            cur.execute(DDL)
        else:
            cur.execute(DDL.replace(f"CREATE TABLE {T_CANONICAL}",
                                    f"CREATE TABLE IF NOT EXISTS {T_CANONICAL}"))
    pg.commit()

    my = pymysql.connect(host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER,
                         password=MYSQL_PASSWORD, database=MYSQL_DB,
                         charset='utf8mb4')
    types_sql = ','.join(f"'{t}'" for t in args.types)

    t0 = time.time()
    total_read = total_ok = 0
    last_mid = ''
    pool = Pool(processes=args.workers)

    try:
        while True:
            with my.cursor() as cur:
                cur.execute(
                    f"SELECT MessageId, EntityId, TypeNamedId, FileName, xml_val "
                    f"FROM {MYSQL_TABLE} "
                    f"WHERE TypeNamedId IN ({types_sql}) AND MessageId > %s "
                    f"ORDER BY MessageId LIMIT %s",
                    (last_mid, MYSQL_CHUNK))
                chunk = cur.fetchall()
            if not chunk:
                break
            last_mid = chunk[-1][0]

            buf = []
            for out in pool.imap(_canon_worker, chunk, chunksize=100):
                buf.append(out)
                total_read += 1
                total_ok += 1 if out[25] else 0
                if len(buf) >= PG_PAGE:
                    with pg.cursor() as cur:
                        execute_values(cur, INSERT_SQL, buf, page_size=PG_PAGE)
                    pg.commit()
                    buf.clear()
                if args.limit and total_read >= args.limit:
                    break
            if buf:
                with pg.cursor() as cur:
                    execute_values(cur, INSERT_SQL, buf, page_size=PG_PAGE)
            pg.commit()

            dt = time.time() - t0
            print(f"\r  прочитано {total_read} | ok {total_ok} "
                  f"({total_ok / max(total_read, 1):.1%}) | "
                  f"{total_read / dt:.0f} док/с", end='', flush=True)
            if args.limit and total_read >= args.limit:
                break
    finally:
        pool.terminate()
        my.close()

    pg.commit()
    with pg.cursor() as cur:
        cur.execute(f"SELECT COUNT(*), SUM(parse_ok::int), "
                    f"AVG(canon_len)::int, MAX(canon_len), AVG(src_len)::int "
                    f"FROM {T_CANONICAL}")
        n, ok, avg_c, max_c, avg_s = cur.fetchone()
    print(f"\n\n✅ {T_CANONICAL}: {n} док. | parse_ok {ok} ({ok / max(n, 1):.1%}) | "
          f"canon avg {avg_c} / max {max_c} симв. | src avg {avg_s} симв. | "
          f"{time.time() - t0:.0f} c")
    pg.close()


if __name__ == '__main__':
    main()
