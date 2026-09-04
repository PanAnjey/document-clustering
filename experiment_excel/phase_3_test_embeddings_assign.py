# phase_3_test_embeddings_assign.py
import sys
import traceback

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from experiment_config import DB_URL, EMBED_BATCH_SIZE, EMBED_DIM, SIMILARITY_THRESHOLD
from embed_helper import encode_texts


def fetch_test_texts(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT id, txt FROM temp_excel_test ORDER BY id")
        rows = cur.fetchall()
    ids, texts = zip(*rows) if rows else ([], [])
    return list(ids), list(texts)


def fetch_centroids(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT label, embedding FROM temp_hdbscan_centroids")
        rows = cur.fetchall()
    labels = []
    centroids = []
    for lab, blob in rows:
        labels.append(lab)
        centroids.append(np.frombuffer(blob, dtype=np.float32))
    if not centroids:
        return [], np.empty((0, EMBED_DIM), dtype=np.float32)
    return labels, np.stack(centroids)


def store_assignments(conn, test_ids, assigned_labels, sims):
    rows = [(int(tid), int(lab), float(sim)) for tid, lab, sim in zip(test_ids, assigned_labels, sims)]
    with conn.cursor() as cur:
        execute_values(cur,
            "INSERT INTO temp_test_assign (test_id, assigned_label, similarity) VALUES %s",
            rows, page_size=500)
    conn.commit()


def main():
    conn = psycopg2.connect(DB_URL)
    # Ensure target table exists
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS temp_test_assign")
        cur.execute("""
            CREATE TABLE temp_test_assign (
                test_id INTEGER PRIMARY KEY,
                assigned_label INTEGER NOT NULL,
                similarity REAL NOT NULL
            )
        """)
    conn.commit()

    test_ids, test_texts = fetch_test_texts(conn)
    if not test_ids:
        sys.exit('❌ No test texts – abort Phase 3.')

    test_emb = encode_texts(test_texts, batch_size=EMBED_BATCH_SIZE)

    labels, centroids = fetch_centroids(conn)
    if centroids.size == 0:
        sys.exit('❌ No centroids found – ensure Phase 2 succeeded.')

    # Cosine similarity = 1 - cosine distance.
    # Эмбеддинги нормализованы, поэтому эквивалентно скалярному произведению,
    # но оставляем sklearn-вариант — он устойчив к ненормализованным центроидам
    # (среднее нормализованных векторов само не нормализовано).
    from sklearn.metrics import pairwise
    sims = 1 - pairwise.cosine_distances(test_emb, centroids)
    assigned_idx = sims.argmax(axis=1)
    assigned_labels = [labels[i] for i in assigned_idx]
    best_sims = sims.max(axis=1)

    store_assignments(conn, test_ids, assigned_labels, best_sims)
    coverage = (best_sims >= SIMILARITY_THRESHOLD).mean()
    print(f'✅ Phase 3 finished – coverage @ {SIMILARITY_THRESHOLD}: {coverage:.2%} '
          f'(mean sim={best_sims.mean():.3f}, min={best_sims.min():.3f}, max={best_sims.max():.3f})')
    conn.close()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
