# phase_3c_assign_test.py
# Назначение TEST-набора тематическим кластерам train (temp_theme2_centroids).
#
# Цепочка идентична train: ПРЕДМЕТ (temp_llm_subject_test) → nomic-эмбеддинг
# → косинус к центроидам. sim >= SIMILARITY_THRESHOLD — назначение теме,
# sim < порога — буфер «неопознанные» (детектор дрейфа тематики).
#
# Таблица: temp_theme2_test_assign (test_id PK, assigned_theme, similarity, is_unknown)

import sys
import traceback

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from experiment_config import DB_URL, EMBED_BATCH_SIZE, EMBED_DIM, SIMILARITY_THRESHOLD  # noqa
from embed_helper import encode_texts


def fetch_test_subjects(conn):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT train_id, predmet FROM temp_llm_subject_test
            WHERE predmet IS NOT NULL ORDER BY train_id
        """)
        rows = cur.fetchall()
    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    return ids, texts


def fetch_centroids(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT theme_label, embedding FROM temp_theme2_centroids")
        rows = cur.fetchall()
    labels, cents = [], []
    for lab, blob in rows:
        labels.append(lab)
        cents.append(np.frombuffer(blob, dtype=np.float32))
    if not cents:
        return [], np.empty((0, EMBED_DIM), dtype=np.float32)
    return labels, np.stack(cents)


def store(conn, rows):
    with conn.cursor() as cur:
        execute_values(cur,
            """INSERT INTO temp_theme2_test_assign
               (test_id, assigned_theme, similarity, is_unknown) VALUES %s""",
            rows, page_size=500)
    conn.commit()


def main():
    conn = psycopg2.connect(DB_URL)
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS temp_theme2_test_assign")
        cur.execute("""
            CREATE TABLE temp_theme2_test_assign (
                test_id INTEGER PRIMARY KEY,
                assigned_theme INTEGER NOT NULL,
                similarity REAL NOT NULL,
                is_unknown BOOLEAN NOT NULL
            )
        """)
    conn.commit()

    test_ids, texts = fetch_test_subjects(conn)
    if not test_ids:
        sys.exit('❌ temp_llm_subject_test пуст — сначала phase_1c --test')
    labels, centroids = fetch_centroids(conn)
    if centroids.size == 0:
        sys.exit('❌ Нет центроидов — сначала phase_2d на train')
    print(f'Test: {len(test_ids)} документов, центроидов: {len(labels)}')

    emb = encode_texts(texts, batch_size=EMBED_BATCH_SIZE)

    from sklearn.metrics import pairwise
    sims = 1 - pairwise.cosine_distances(emb, centroids)
    assigned_idx = sims.argmax(axis=1)
    assigned = [labels[i] for i in assigned_idx]
    best = sims.max(axis=1)

    rows = [
        (int(tid), int(lab), float(sim), bool(sim < SIMILARITY_THRESHOLD))
        for tid, lab, sim in zip(test_ids, assigned, best)
    ]
    store(conn, rows)

    coverage = float((best >= SIMILARITY_THRESHOLD).mean())
    n_unknown = int((best < SIMILARITY_THRESHOLD).sum())
    print(f'\n✅ Phase 3c finished')
    print(f'   coverage @ {SIMILARITY_THRESHOLD}: {coverage:.2%} '
          f'({len(best) - n_unknown}/{len(best)})')
    print(f'   неопознанные (дрейф-буфер): {n_unknown} ({n_unknown / len(best):.1%})')
    print(f'   sim avg={best.mean():.3f}, min={best.min():.3f}, max={best.max():.3f}')

    # Распределение назначений по темам
    with conn.cursor() as cur:
        cur.execute("""
            SELECT a.assigned_theme, COUNT(*) c, AVG(a.similarity) s,
                   MAX(w.top_words)
            FROM temp_theme2_test_assign a
            LEFT JOIN temp_theme2_topwords w ON w.theme_label = a.assigned_theme
            GROUP BY a.assigned_theme ORDER BY c DESC
        """)
        print('   per-theme:')
        for lab, c, s, words in cur.fetchall():
            print(f'     тема {lab:>2}: {c:>5} док. avg_sim={s:.3f} | {(words or "")[:60]}')
    conn.close()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
