# _test_v2_canon.py — тест обновлённого канонизатора (пп.1-5) на новых типах.
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import pymysql
from xml_canonicalizer import canonicalize, strip_contract_refs

conn = pymysql.connect(host='localhost', port=3308, user='root', password='mysql',
                       database='events_26', charset='utf8mb4')
cur = conn.cursor()

cases = [
    ('UniversalTransferDocument (обычный)',
     "SELECT FileName, xml_val FROM xml WHERE TypeNamedId='UniversalTransferDocument' AND MessageId<'05000000-0000-0000-0000-000000000000' LIMIT 1"),
    ('UTD МАРКИРОВАННЫЙ (N3=1)',
     "SELECT FileName, xml_val FROM xml WHERE TypeNamedId='UniversalTransferDocument' AND FileName LIKE '%\\_0\\_1\\_0\\_0\\_0\\_00.xml' LIMIT 1"),
    ('PerformedWorkAcceptanceCertificate (акт работ)',
     "SELECT FileName, xml_val FROM xml WHERE TypeNamedId='PerformedWorkAcceptanceCertificate' AND MessageId<'05000000-0000-0000-0000-000000000000' LIMIT 1"),
    ('StorageInventoryAcceptanceCertificate',
     "SELECT FileName, xml_val FROM xml WHERE TypeNamedId='StorageInventoryAcceptanceCertificate' AND MessageId<'05000000-0000-0000-0000-000000000000' LIMIT 1"),
    ('UniversalCorrectionDocument (УКД)',
     "SELECT FileName, xml_val FROM xml WHERE TypeNamedId='UniversalCorrectionDocument' AND MessageId<'05000000-0000-0000-0000-000000000000' LIMIT 1"),
    ('InvoiceCorrection (КСФ)',
     "SELECT FileName, xml_val FROM xml WHERE TypeNamedId='InvoiceCorrection' AND MessageId<'05000000-0000-0000-0000-000000000000' LIMIT 1"),
]

for label, sql in cases:
    cur.execute(sql)
    r = cur.fetchone()
    if not r:
        print(f"### {label}: не найден\n")
        continue
    fname, xml = r
    res = canonicalize(xml)
    parts = fname.split('_')
    n3 = parts[7] if len(parts) > 7 else '?'
    print(f"### {label}")
    print(f"    N3={n3} | ok={res['ok']} {res['error']}")
    print(f"    V1: {res['canonical_text'][:400]}")
    print(f"    V2: {res['subject_text'][:300]}")
    print(f"    V3: {strip_contract_refs(res['subject_text'])[:300]}")
    print()
conn.close()
