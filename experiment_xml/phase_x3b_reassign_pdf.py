# phase_x3b_reassign_pdf.py
# Переназначение LLM-предметов pdf_text / pdf_tables на ПОЛНЫЙ набор из 26
# центроидов temp_theme2_centroids_v2 (исторические temp_pdftext_assign /
# temp_pdftables_assign были построены на старом наборе из 13 тем — отсюда
# нулевые доли тем 12..26 и смещение сравнения с XML).
#
# Результат: temp_pdftext_assign_v2, temp_pdftables_assign_v2
# (та же схема: id, assigned_theme, similarity, is_unknown).
#
# Запуск: python phase_x3b_reassign_pdf.py

import sys
import time

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from experiment_xml_config import (  # noqa: E402
    DB_URL, SIMILARITY_THRESHOLD, CENTROIDS_TABLE,
)
from xml_embed_helper import encode_texts  # noqa: E402

JOBS = [
    ('temp_llm_subject_pdftext', 'temp_pdftext_assign_v2'),
    ('temp_llm_subject_pdftables', 'temp_pdftables_assign_v2'),
]
CHUNK = 5000


def main():
    conn = psycopg2.connect(DB_URL)
    with conn.cursor() as cur:
        cur.execute(f"SELECT theme_label, embedding FROM {CENTROIDS_TABLE} ORDER BY theme_label")
        rows = cur.fetchall()
    labels = [r[0] for r in rows]
    cents = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    cents /= (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-12)
    print(f"центроидов: {len(labels)}")

    for subj_table, out_table in JOBS:
        with conn.cursor() as cur:
            cur.execute(f"SELECT train_id, predmet FROM {subj_table} "
                        f"WHERE predmet IS NOT NULL ORDER BY train_id")
            rows = cur.fetchall()
        ids = [r[0] for r in rows]
        texts = [r[1] or '' for r in rows]
        print(f"\n{subj_table}: {len(ids)} предметов → {out_table}")

        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {out_table}")
            cur.execute(f"""CREATE TABLE {out_table} (
                pdftext_id INTEGER PRIMARY KEY,
                assigned_theme INTEGER NOT NULL,
                similarity REAL NOT NULL,
                is_unknown BOOLEAN NOT NULL)""")
        conn.commit()

        t0 = time.time()
        for start in range(0, len(ids), CHUNK):
            cids = ids[start:start + CHUNK]
            ctexts = texts[start:start + CHUNK]
            emb = encode_texts(ctexts, batch_size=64)
            emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
            sims = emb @ cents.T
            best_idx = sims.argmax(axis=1)
            best = sims[np.arange(len(cids)), best_idx]
            assigned = np.array([labels[i] for i in best_idx])
            unknown = best < SIMILARITY_THRESHOLD
            payload = [(int(i), int(a), float(s), bool(u))
                       for i, a, s, u in zip(cids, assigned, best, unknown)]
            with conn.cursor() as cur:
                execute_values(cur,
                               f"INSERT INTO {out_table} (pdftext_id, assigned_theme, "
                               f"similarity, is_unknown) VALUES %s",
                               payload, page_size=1000)
            conn.commit()
            done = min(start + CHUNK, len(ids))
            print(f"  {done}/{len(ids)} | {done / (time.time() - t0):.0f} док/с", flush=True)

        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*), AVG(similarity)::numeric(5,3), "
                        f"SUM(is_unknown::int) FROM {out_table}")
            n, avg_s, unk = cur.fetchone()
        print(f"  ✅ {out_table}: {n} | avg_sim={avg_s} | unknown={unk} ({unk/n:.1%})")

    conn.close()
    print("Phase X3b completed")


if __name__ == '__main__':
    main()
