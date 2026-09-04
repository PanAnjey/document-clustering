# phase_2d_llm_themes.py
# Тематическая кластеризация по LLM-саммаризации (temp_llm_subject):
# эмбеддинг строки "ТЕМА. ТИП. НАЗНАЧЕНИЕ" → HDBSCAN → temp_theme2_*.
#
# Работает и на частичном покрытии (после --limit в phase_1c) —
# кластеризует только обработанные LLM документы.
#
# Таблицы:
#   temp_theme2_clusters  (train_id PK, theme_label)
#   temp_theme2_centroids (theme_label PK, embedding BYTEA)
#   temp_theme2_topwords  (theme_label PK, top_words TEXT)

import sys
import traceback

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from experiment_config import DB_URL, EMBED_BATCH_SIZE, MIN_CLUSTER_SIZE_FACTOR  # noqa: sys.path+utf8
from embed_helper import encode_texts
from phase_2c_themes import c_tfidf_top_words
import hdbscan

# Порог слияния кластеров по косинусному сходству центроидов.
# LLM-фразы короткие и однородные → HDBSCAN дробит одну тему на микрокластеры
# (8 вариантов «приема передачи»). Центроиды одной темы имеют sim > 0.9.
MERGE_SIM = 0.92


def merge_by_centroids(labels, emb, sim_threshold=MERGE_SIM):
    """Агломеративное слияние кластеров с близкими центроидами (union-find).
    Шум (-1) не участвует. Возвращает новые labels."""
    labels = labels.copy()
    real = [l for l in np.unique(labels) if l != -1]
    if len(real) < 2:
        return labels

    def centroids(labs):
        return {l: emb[labs == l].mean(axis=0) for l in np.unique(labs) if l != -1}

    parent = {l: l for l in real}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    while True:
        cents = centroids(labels)
        labs = sorted(cents)
        mats = np.stack([cents[l] for l in labs])
        mats = mats / np.maximum(np.linalg.norm(mats, axis=1, keepdims=True), 1e-9)
        sim = mats @ mats.T
        np.fill_diagonal(sim, -1)
        i, j = np.unravel_index(sim.argmax(), sim.shape)
        if sim[i, j] < sim_threshold:
            break
        # union
        ri, rj = find(labs[i]), find(labs[j])
        if ri != rj:
            parent[rj] = ri
        labels = np.array([find(l) if l != -1 else -1 for l in labels])
    # Перенумеровать подряд
    uniq = {old: new for new, old in enumerate(sorted(set(labels.tolist()) - {-1}))}
    return np.array([uniq.get(l, -1) for l in labels])


def fetch_llm_subjects(conn):
    """(ids, predmet-строки). Эмбеддинг — ТОЛЬКО по ПРЕДМЕТу (направление);
    ТИП хранится в temp_llm_subject для вторичного уровня."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT train_id, predmet
            FROM temp_llm_subject
            WHERE predmet IS NOT NULL
            ORDER BY train_id
        """)
        rows = cur.fetchall()
    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    return ids, texts


def store(conn, ids, labels, emb, top_words_map):
    with conn.cursor() as cur:
        execute_values(cur,
            "INSERT INTO temp_theme2_clusters (train_id, theme_label) VALUES %s",
            [(int(i), int(l)) for i, l in zip(ids, labels)], page_size=500)
        real = [l for l in np.unique(labels) if l != -1]
        cent = []
        for lab in real:
            c = emb[labels == lab].mean(axis=0).astype(np.float32)
            cent.append((int(lab), c.tobytes()))
        if cent:
            execute_values(cur,
                "INSERT INTO temp_theme2_centroids (theme_label, embedding) VALUES %s",
                cent, page_size=200)
        if top_words_map:
            execute_values(cur,
                "INSERT INTO temp_theme2_topwords (theme_label, top_words) VALUES %s",
                [(int(l), w) for l, w in top_words_map.items()], page_size=100)
    conn.commit()


def main():
    conn = psycopg2.connect(DB_URL)
    with conn.cursor() as cur:
        for t in ('temp_theme2_clusters', 'temp_theme2_centroids', 'temp_theme2_topwords'):
            cur.execute(f"DROP TABLE IF EXISTS {t}")
        cur.execute("CREATE TABLE temp_theme2_clusters (train_id INTEGER PRIMARY KEY, theme_label INTEGER NOT NULL)")
        cur.execute("CREATE TABLE temp_theme2_centroids (theme_label INTEGER PRIMARY KEY, embedding BYTEA NOT NULL)")
        cur.execute("CREATE TABLE temp_theme2_topwords (theme_label INTEGER PRIMARY KEY, top_words TEXT NOT NULL)")
    conn.commit()

    ids, texts = fetch_llm_subjects(conn)
    if not ids:
        sys.exit('❌ temp_llm_subject пуст — сначала phase_1c_llm_subject.py')
    print(f'LLM-субъекты: {len(ids)} документов')

    # Распределение «сырых» значений ПРЕДМЕТ (до кластеризации) — уже информативно
    with conn.cursor() as cur:
        cur.execute("""
            SELECT predmet, COUNT(*) c FROM temp_llm_subject
            WHERE predmet IS NOT NULL
            GROUP BY predmet ORDER BY c DESC LIMIT 15
        """)
        print('Top-15 сырых ПРЕДМЕТ от LLM:')
        for predmet, c in cur.fetchall():
            print(f'    {c:>5} | {predmet}')

    emb = encode_texts(texts, batch_size=EMBED_BATCH_SIZE)
    n = len(emb)
    min_cluster = max(2, int(MIN_CLUSTER_SIZE_FACTOR * n))
    print(f'HDBSCAN: n={n}, min_cluster_size={min_cluster}')
    labels = hdbscan.HDBSCAN(
        metric='euclidean', min_cluster_size=min_cluster, core_dist_n_jobs=-1,
    ).fit_predict(emb)
    n_raw = len(set(labels.tolist()) - {-1})

    # Слияние микрокластеров-дублей по близким центроидам
    labels = merge_by_centroids(labels, emb, sim_threshold=MERGE_SIM)

    real_labels = [l for l in np.unique(labels) if l != -1]
    noise = int((labels == -1).sum())
    print(f'HDBSCAN done: {n_raw} сырых → {len(real_labels)} тем после слияния '
          f'(sim≥{MERGE_SIM}), шум {noise} ({noise / n:.1%})')

    top_words_map = {}
    if real_labels:
        class_texts, order = [], []
        for lab in real_labels:
            class_texts.append(' '.join(texts[i] for i in np.where(labels == lab)[0]))
            order.append(lab)
        tops = c_tfidf_top_words(class_texts, top_n=8)
        top_words_map = dict(zip(order, tops))

    store(conn, ids, labels, emb, top_words_map)

    for lab in sorted(real_labels, key=lambda l: -int((labels == l).sum())):
        cnt = int((labels == lab).sum())
        print(f'  тема {lab:>2}: {cnt:>5} док. | {top_words_map.get(lab, "")}')

    try:
        if 2 <= len(np.unique(labels)) < n:
            from sklearn.metrics import silhouette_score
            sil = silhouette_score(emb, labels, metric='cosine',
                                   sample_size=min(5000, n), random_state=42)
            print(f'Silhouette (sampled): {sil:.3f}')
    except Exception as e:
        print(f'Silhouette skipped: {e}')
    conn.close()
    print('✅ Phase 2d (LLM themes) completed — temp_theme2_*')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
