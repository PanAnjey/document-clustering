# phase_4_second_level_clustering.py
import sys
import traceback

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from experiment_config import (
    DB_URL, EMBED_DIM,
    KMEANS_MAX_ITER, KMEANS_MAX_K, KMEANS_DOCS_PER_CLUSTER,
)


def fetch_type_embeddings(conn, label):
    """Переиспользуем эмбеддинги, посчитанные в Phase 2 (temp_train_embeddings),
    — GPU на этой фазе вообще не нужен."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT h.train_id, e.embedding
            FROM temp_hdbscan h
            JOIN temp_train_embeddings e ON e.train_id = h.train_id
            WHERE h.label = %s
            ORDER BY h.train_id
            """,
            (label,)
        )
        rows = cur.fetchall()
    if not rows:
        return [], np.empty((0, EMBED_DIM), dtype=np.float32)
    ids = [r[0] for r in rows]
    emb = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    return ids, emb


def store_second_level(conn, records):
    # records: [(train_id, type_label, sub_label, embedding_bytes), ...]
    if not records:
        return
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO temp_second_level (train_id, type_label, sub_label, embedding)
            VALUES %s
            """,
            records,
            page_size=200,
        )
    conn.commit()


def kmeans_labels(emb, n):
    """KMeans с адаптивным k (из плана): k = max(2, min(MAX_K, N // DOCS_PER_CLUSTER)).

    Top2Vec удалён: пакет не установлен в окружении, а исходное использование
    его API было неверным (model.document_ids — не метки кластеров;
    document_vectors выровнены по внутреннему порядку Top2Vec, а не по входу).
    """
    if n < 2:
        return np.zeros(n, dtype=int)
    k = max(2, min(KMEANS_MAX_K, n // KMEANS_DOCS_PER_CLUSTER))
    k = min(k, n)
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, max_iter=KMEANS_MAX_ITER,
                random_state=42, n_init=10)
    return km.fit_predict(emb)


def main():
    conn = psycopg2.connect(DB_URL)
    # Ensure table exists
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS temp_second_level")
        cur.execute(
            """
            CREATE TABLE temp_second_level (
                train_id INTEGER NOT NULL,
                type_label INTEGER NOT NULL,
                sub_label INTEGER NOT NULL,
                embedding BYTEA NOT NULL
            )
            """
        )
    conn.commit()

    # Get distinct HDBSCAN type labels.
    # ВАЖНО: шум (-1) исключаем — это не «тип», а гетерогенный отказник
    # (Phase 2 его тоже исключает при подсчёте центроидов).
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT label FROM temp_hdbscan WHERE label != -1 ORDER BY label")
        type_labels = [r[0] for r in cur.fetchall()]

    if not type_labels:
        sys.exit('❌ No type labels (all noise?) – check Phase 2 output.')

    for t_label in type_labels:
        ids, emb = fetch_type_embeddings(conn, t_label)
        n = len(ids)
        if n == 0:
            continue
        sub_labels = kmeans_labels(emb, n)

        # Store results
        records = []
        for i, sid in enumerate(ids):
            vec = emb[i].astype(np.float32).tobytes()
            records.append((int(sid), int(t_label), int(sub_labels[i]), vec))
        store_second_level(conn, records)
        print(f'✅ Type {t_label}: {n} docs → {len(set(sub_labels.tolist()))} sub-clusters')

    conn.close()
    print('✅ Phase 4 completed – second-level clustering stored.')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
