# _test_ab.py — A/B сравнение представлений V1 (полный канон) vs V2 (тематический)
# на 2000 документах: coverage @ 0.8 и распределение similarity.
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

import numpy as np
import psycopg2
import pymysql

from experiment_xml_config import (MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD,
                                   MYSQL_DB, MYSQL_TABLE, DB_URL, SIMILARITY_THRESHOLD,
                                   CENTROIDS_TABLE)
from xml_canonicalizer import canonicalize
from xml_embed_helper import encode_texts

N = 2000

my = pymysql.connect(host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER,
                     password=MYSQL_PASSWORD, database=MYSQL_DB, charset='utf8mb4')
with my.cursor() as cur:
    cur.execute(f"""SELECT MessageId, TypeNamedId, xml_val FROM {MYSQL_TABLE}
                    WHERE TypeNamedId IN ('UniversalTransferDocument','Invoice')
                    ORDER BY MessageId LIMIT {N}""")
    rows = cur.fetchall()
my.close()

v1_texts, v2_texts, types_ = [], [], []
for mid, tname, xml in rows:
    r = canonicalize(xml)
    v1_texts.append(r['canonical_text'])
    v2_texts.append(r['subject_text'])
    types_.append(tname)

print("пример V1:", v1_texts[0][:200])
print("пример V2:", v2_texts[0][:200])
print(f"V2 длина: avg {np.mean([len(t) for t in v2_texts]):.0f} симв., "
      f"пустых: {sum(1 for t in v2_texts if not t)}")

pg = psycopg2.connect(DB_URL)
with pg.cursor() as cur:
    cur.execute(f"SELECT theme_label, embedding FROM {CENTROIDS_TABLE} ORDER BY theme_label")
    crows = cur.fetchall()
labels = [r[0] for r in crows]
cents = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in crows])
cents /= (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-12)
pg.close()

for name, texts in (('V1 full', v1_texts), ('V2 subject', v2_texts)):
    emb = encode_texts(texts, batch_size=64)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    sims = emb @ cents.T
    best = sims.max(axis=1)
    assigned = np.array([labels[i] for i in sims.argmax(axis=1)])
    unk = best < SIMILARITY_THRESHOLD
    print(f"\n== {name} ==")
    print(f"  coverage @ {SIMILARITY_THRESHOLD}: {(~unk).mean():.1%} | "
          f"sim p10={np.percentile(best,10):.3f} p50={np.percentile(best,50):.3f} "
          f"p90={np.percentile(best,90):.3f} max={best.max():.3f}")
    # распределение по темам (топ-8)
    vals, cnts = np.unique(assigned, return_counts=True)
    top = sorted(zip(cnts, vals), reverse=True)[:8]
    print("  топ темы:", [(int(v), int(c)) for c, v in top])
