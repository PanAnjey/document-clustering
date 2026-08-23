# _test_canonicalizer.py — быстрый тест канонизатора на живых данных MySQL.
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import pymysql
from experiment_xml.experiment_xml_config import (
    MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB, MYSQL_TABLE,
    MAX_TABLE_ROWS, MAX_INFO_BLOCK_CHARS, MAX_OSNLEN,
)
from experiment_xml.xml_canonicalizer import canonicalize

conn = pymysql.connect(host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER,
                       password=MYSQL_PASSWORD, database=MYSQL_DB,
                       charset='utf8mb4')
cur = conn.cursor()

# 1) Два известных документа (из разведки) + 3 случайных с >5 строк + пара обычных
cur.execute(f"""
    SELECT MessageId, EntityId, TypeNamedId, FileName, xml_val
    FROM {MYSQL_TABLE}
    WHERE TypeNamedId IN ('UniversalTransferDocument','Invoice')
      AND MessageId IN ('d42eea1f-594d-45b9-b427-38fa084ef209',
                        'a67ae468-3503-4834-b608-67fd0826ec68')
""")
rows = cur.fetchall()

# 2) Документы с большим числом строк табличной части
cur.execute(f"""
    SELECT MessageId, EntityId, TypeNamedId, FileName, xml_val
    FROM {MYSQL_TABLE}
    WHERE TypeNamedId = 'UniversalTransferDocument'
      AND MessageId < '11000000-0000-0000-0000-000000000000'
      AND LENGTH(xml_val) > 100000
    LIMIT 3
""")
rows += cur.fetchall()

# 3) Несколько обычных
cur.execute(f"""
    SELECT MessageId, EntityId, TypeNamedId, FileName, xml_val
    FROM {MYSQL_TABLE}
    WHERE TypeNamedId = 'Invoice'
      AND MessageId >= '40000000-0000-0000-0000-000000000000'
      AND MessageId < '41000000-0000-0000-0000-000000000000'
    LIMIT 4
""")
rows += cur.fetchall()

conn.close()

print(f"документов: {len(rows)}\n")
for mid, eid, tname, fname, xml in rows:
    r = canonicalize(xml, max_rows=MAX_TABLE_ROWS,
                     max_info_chars=MAX_INFO_BLOCK_CHARS, max_osn_chars=MAX_OSNLEN)
    src_len = len(xml or '')
    print("=" * 100)
    print(f"{tname} | src={src_len} симв. | rows_total={r['meta'].get('n_rows_total')} | ok={r['ok']} {r['error']}")
    print(f"canonical ({len(r['canonical_text'])} симв.):")
    print(r['canonical_text'])
    print()
