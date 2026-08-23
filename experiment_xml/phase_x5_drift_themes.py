# phase_x5_drift_themes.py
# Этап X5: переоткрытие новых тем из дрейф-буфера (V3, is_unknown=True,
# 181K документов) — аналог experiment_excel/phase_3r_assign.py:
#   HDBSCAN (euclidean) → слияние по центроидам (sim 0.92) →
#   c-TF-IDF топ-слова → центроиды новых тем →
#   назначение ВСЕГО буфера на новые центроиды (sim ≥ 0.8).
#
# HDBSCAN выполняется на случайной подвыборке (--sample-max, т.к. 181K×768
# тяжело), назначение — на полном буфере через центроиды.
#
# Таблицы:
#   temp_xml_drift_centroids (new_theme PK, embedding BYTEA, size, top_words)
#   temp_xml_drift_assign    (xml_id PK, new_theme, similarity)
#
# Запуск:
#   python phase_x5_drift_themes.py                    # полный цикл
#   python phase_x5_drift_themes.py --sample-max 80000 # больше выборка
#   python phase_x5_drift_themes.py --skip-cluster     # только assign по готовым центроидам

import argparse
import sys
import time

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from experiment_xml_config import (  # noqa: E402
    DB_URL, T_CANONICAL, T_EMBEDDINGS, SIMILARITY_THRESHOLD,
)
from xml_canonicalizer import strip_contract_refs  # noqa: E402

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'experiment_excel'))
from phase_2c_themes import c_tfidf_top_words  # noqa: E402
from phase_2d_llm_themes import merge_by_centroids  # noqa: E402

import hdbscan  # noqa: E402

T_DRIFT_CENTROIDS = 'temp_xml_drift_centroids'
T_DRIFT_ASSIGN = 'temp_xml_drift_assign'
MIN_CLUSTER_SIZE = 150      # ~0.25% от 60K выборки
MERGE_SIM = 0.92


def fetch_unknown(conn):
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT a.xml_id, e.subject_clean_embedding, c.subject_text, c.type_named_id
            FROM temp_xml_subject_clean_assign a
            JOIN {T_EMBEDDINGS} e ON e.xml_id = a.xml_id
            JOIN {T_CANONICAL} c ON c.id = a.xml_id
            WHERE a.is_unknown AND e.subject_clean_embedding IS NOT NULL
            ORDER BY a.xml_id""")
        rows = cur.fetchall()
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    emb = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    texts = [strip_contract_refs(r[2] or '') for r in rows]
    types = [r[3] for r in rows]
    return ids, emb, texts, types


def run_cluster(conn, sample_max):
    ids, emb, texts, types = fetch_unknown(conn)
    print(f"дрейф-буфер: {len(ids)} документов")
    rng = np.random.default_rng(42)
    if len(ids) > sample_max:
        idx = np.sort(rng.choice(len(ids), size=sample_max, replace=False))
        print(f"подвыборка для HDBSCAN: {len(idx)} (seed=42)")
    else:
        idx = np.arange(len(ids))
    s_emb, s_texts = emb[idx], [texts[i] for i in idx]
    s_ids = ids[idx]

    print(f"HDBSCAN (min_cluster_size={MIN_CLUSTER_SIZE})...")
    t0 = time.time()
    labels = hdbscan.HDBSCAN(
        metric='euclidean', min_cluster_size=MIN_CLUSTER_SIZE,
        core_dist_n_jobs=-1,
    ).fit_predict(s_emb)
    n_raw = len(set(labels.tolist()) - {-1})
    n_noise = int((labels == -1).sum())
    print(f"  сырых кластеров: {n_raw}, шум: {n_noise} ({n_noise/len(labels):.0%}), "
          f"{time.time()-t0:.0f} c")

    labels = merge_by_centroids(labels, s_emb, sim_threshold=MERGE_SIM)
    real = [l for l in np.unique(labels) if l != -1]
    print(f"  после слияния: {len(real)} тем")

    # топ-слова
    class_texts, order = [], []
    for lab in real:
        class_texts.append(' '.join(s_texts[i] for i in np.where(labels == lab)[0]))
        order.append(lab)
    tops = c_tfidf_top_words(class_texts, top_n=8)
    top_map = dict(zip(order, tops))

    # центроиды (нормализованные)
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {T_DRIFT_CENTROIDS}")
        cur.execute(f"""CREATE TABLE {T_DRIFT_CENTROIDS} (
            new_theme INTEGER PRIMARY KEY,
            embedding BYTEA NOT NULL,
            size INTEGER NOT NULL,
            top_words TEXT)""")
    conn.commit()
    payload = []
    for lab in order:
        c = s_emb[labels == lab].mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-12)
        payload.append((int(lab), c.astype(np.float32).tobytes(),
                        int((labels == lab).sum()), top_map.get(lab, '')))
    with conn.cursor() as cur:
        execute_values(cur,
                       f"INSERT INTO {T_DRIFT_CENTROIDS} (new_theme, embedding, size, top_words) "
                       f"VALUES %s", payload, page_size=100)
    conn.commit()
    print(f"  → {T_DRIFT_CENTROIDS}: {len(payload)} центроидов")
    return ids, emb, texts, types


def run_assign(conn):
    with conn.cursor() as cur:
        cur.execute(f"SELECT new_theme, embedding, top_words FROM {T_DRIFT_CENTROIDS} ORDER BY new_theme")
        rows = cur.fetchall()
    if not rows:
        sys.exit("нет центроидов — сначала кластеризация")
    labels = [r[0] for r in rows]
    words = {r[0]: r[2] for r in rows}
    cents = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    cents /= (np.linalg.norm(cents, axis=1, keepdims=True) + 1e-12)

    ids, emb, texts, types = fetch_unknown(conn)
    print(f"назначение буфера на {len(labels)} новых центроидов: {len(ids)} док.")

    sims = emb @ cents.T
    best_idx = sims.argmax(axis=1)
    best = sims[np.arange(len(ids)), best_idx]
    assigned = np.array([labels[i] for i in best_idx])
    ok = best >= SIMILARITY_THRESHOLD

    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {T_DRIFT_ASSIGN}")
        cur.execute(f"""CREATE TABLE {T_DRIFT_ASSIGN} (
            xml_id INTEGER PRIMARY KEY,
            new_theme INTEGER NOT NULL,
            similarity REAL NOT NULL)""")
    conn.commit()
    payload = [(int(i), int(a), float(s))
               for i, a, s, o in zip(ids, assigned, best, ok) if o]
    with conn.cursor() as cur:
        execute_values(cur,
                       f"INSERT INTO {T_DRIFT_ASSIGN} (xml_id, new_theme, similarity) VALUES %s",
                       payload, page_size=1000)
    conn.commit()

    n_ok = int(ok.sum())
    print(f"  назначено: {n_ok} ({n_ok/len(ids):.1%} буфера), "
          f"остаточный шум: {len(ids)-n_ok} ({1-n_ok/len(ids):.1%})")

    # отчёт: размеры, топ-слова, состав по типам
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT d.new_theme, COUNT(*) n, AVG(d.similarity)::numeric(5,3) s,
                   string_agg(DISTINCT c.type_named_id, ', ') types
            FROM {T_DRIFT_ASSIGN} d
            JOIN {T_CANONICAL} c ON c.id = d.xml_id
            GROUP BY d.new_theme ORDER BY n DESC""")
        print("\n== новые темы ==")
        for t, n, s, tp in cur.fetchall():
            types_short = ','.join(sorted(set(x[:12] for x in tp.split(', '))))
            print(f"  тема {t:>3}: {n:>7} док. avg_sim={s} | {(words.get(t) or '')[:70]} | {types_short}")
    return words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--skip-cluster', action='store_true')
    ap.add_argument('--sample-max', type=int, default=60000)
    args = ap.parse_args()
    conn = psycopg2.connect(DB_URL)
    if not args.skip_cluster:
        run_cluster(conn, args.sample_max)
    run_assign(conn)
    conn.close()
    print('Phase X5 completed')


if __name__ == '__main__':
    main()
