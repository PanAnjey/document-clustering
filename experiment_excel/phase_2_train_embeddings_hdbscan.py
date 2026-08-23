# phase_2_train_embeddings_hdbscan.py
import sys
import traceback

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from experiment_config import DB_URL, EMBED_BATCH_SIZE, MIN_CLUSTER_SIZE_FACTOR
from embed_helper import encode_texts
import hdbscan


def fetch_training_texts(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT id, txt FROM temp_excel_train ORDER BY id")
        rows = cur.fetchall()
    ids, texts = zip(*rows) if rows else ([], [])
    return list(ids), list(texts)


def store_labels(conn, ids, labels):
    rows = [(int(i), int(l)) for i, l in zip(ids, labels)]
    with conn.cursor() as cur:
        execute_values(cur,
            "INSERT INTO temp_hdbscan (train_id, label) VALUES %s",
            rows, page_size=500)
    conn.commit()


def store_embeddings(conn, ids, embeddings):
    """Сохраняем эмбеддинги train-набора — Phase 4 переиспользует их
    и не будет гонять GPU повторно."""
    rows = [(int(i), vec.astype(np.float32).tobytes()) for i, vec in zip(ids, embeddings)]
    with conn.cursor() as cur:
        execute_values(cur,
            "INSERT INTO temp_train_embeddings (train_id, embedding) VALUES %s",
            rows, page_size=500)
    conn.commit()


def store_centroids(conn, centroids, label_list):
    rows = []
    for lab, vec in zip(label_list, centroids):
        rows.append((int(lab), vec.astype(np.float32).tobytes()))
    with conn.cursor() as cur:
        execute_values(cur,
            "INSERT INTO temp_hdbscan_centroids (label, embedding) VALUES %s",
            rows, page_size=200)
    conn.commit()


def main():
    conn = psycopg2.connect(DB_URL)
    # Ensure tables for HDBSCAN exist
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS temp_hdbscan")
        cur.execute("DROP TABLE IF EXISTS temp_hdbscan_centroids")
        cur.execute("DROP TABLE IF EXISTS temp_train_embeddings")
        cur.execute("""
            CREATE TABLE temp_hdbscan (
                train_id INTEGER PRIMARY KEY,
                label INTEGER NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE temp_hdbscan_centroids (
                label INTEGER PRIMARY KEY,
                embedding BYTEA NOT NULL
            )
        """)
        cur.execute("""
            CREATE TABLE temp_train_embeddings (
                train_id INTEGER PRIMARY KEY,
                embedding BYTEA NOT NULL
            )
        """)
    conn.commit()

    ids, texts = fetch_training_texts(conn)
    if not ids:
        sys.exit('❌ No training texts – abort Phase 2.')

    # Эмбеддинги: L2-нормализованные (N, 768) float32, без image-модели и .npy
    train_emb = encode_texts(texts, batch_size=EMBED_BATCH_SIZE)
    n_train = len(train_emb)
    min_cluster = max(2, int(MIN_CLUSTER_SIZE_FACTOR * n_train))
    print(f"HDBSCAN: n={n_train}, min_cluster_size={min_cluster}")

    # metric='cosine' НЕ поддерживается hdbscan 0.8.44 + sklearn 1.9
    # (ValueError: Unrecognized metric 'cosine'; 'arccos' тоже убран).
    # Эмбеддинги уже L2-нормализованы → euclidean даёт то же ранжирование,
    # что и cosine: d_euclid² = 2 * (1 - cos).
    clusterer = hdbscan.HDBSCAN(
        metric='euclidean',
        min_cluster_size=min_cluster,
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(train_emb)

    # Save labels + embeddings (embeddings переиспользует Phase 4)
    store_labels(conn, ids, labels)
    store_embeddings(conn, ids, train_emb)

    # Compute centroids for real clusters (ignore -1 noise)
    real_labels = [l for l in np.unique(labels) if l != -1]
    noise_count = int(np.sum(labels == -1))
    print(f"HDBSCAN done: {len(real_labels)} clusters, "
          f"noise={noise_count} ({noise_count / n_train:.1%})")
    centroids = []
    label_list = []
    for lab in real_labels:
        mask = labels == lab
        centroid = train_emb[mask].mean(axis=0)
        centroids.append(centroid)
        label_list.append(lab)
    if centroids:
        centroids = np.stack(centroids)
        store_centroids(conn, centroids, label_list)
    else:
        print('⚠️ No real clusters were formed (all noise). '
              'Consider lowering MIN_CLUSTER_SIZE_FACTOR.')

    # Silhouette с сэмплированием — полная матрица O(N²) на 10-16k документов
    # это ~1-2 ГБ RAM; sample_size ограничивает память.
    try:
        n_labels = len(np.unique(labels))
        if 2 <= n_labels < n_train:
            from sklearn.metrics import silhouette_score
            sil = silhouette_score(
                train_emb, labels, metric='cosine',
                sample_size=min(5000, n_train), random_state=42,
            )
            print(f'Silhouette score (sampled): {sil:.3f}')
        else:
            print(f'Silhouette skipped: n_labels={n_labels}')
    except Exception as e:
        print(f'Silhouette skipped: {e}')
    conn.close()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
