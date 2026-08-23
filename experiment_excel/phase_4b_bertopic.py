# phase_4b_bertopic.py
# Альтернативный второй уровень: BERTopic (UMAP → HDBSCAN → c-TF-IDF)
# на ТЕХ ЖЕ эмбеддингах (temp_train_embeddings) — GPU/переобучение не нужно.
#
# Отличия от KMeans-варианта (phase_4):
#   - число под-кластеров определяет HDBSCAN автоматически (не формула k);
#   - пограничные документы помечаются шумом (-1), а не назначаются насильно;
#   - каждая тема получает top-слова (c-TF-IDF) — семантические метки.
#
# Результаты пишутся в temp_second_level_bt (KMeans-результаты НЕ трогаем),
# поэтому сравнение KMeans vs BERTopic делается прямо в БД.
#
# Запуск (venv с bertopic, системные пакеты видны):
#   C:\Windows\Temp\opencode\report_venv\Scripts\python.exe phase_4b_bertopic.py

import sys
import traceback

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from experiment_config import DB_URL, EMBED_DIM  # noqa: sys.path+utf8

MIN_CLUSTER_SIZE_BT = 10
MIN_SAMPLES_BT = 5
TOP_N_WORDS = 8


def fetch_type_data(conn, label):
    """(ids, texts, embeddings) типа: JOIN train_texts + hdbscan + embeddings."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT t.id, t.txt, e.embedding
            FROM temp_excel_train t
            JOIN temp_hdbscan h ON h.train_id = t.id
            JOIN temp_train_embeddings e ON e.train_id = t.id
            WHERE h.label = %s
            ORDER BY t.id
            """,
            (label,)
        )
        rows = cur.fetchall()
    if not rows:
        return [], [], np.empty((0, EMBED_DIM), dtype=np.float32)
    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    emb = np.stack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
    return ids, texts, emb


def store_bt(conn, records):
    if not records:
        return
    with conn.cursor() as cur:
        execute_values(
            cur,
            "INSERT INTO temp_second_level_bt (train_id, type_label, bt_topic, top_words) VALUES %s",
            records,
            page_size=200,
        )
    conn.commit()


def run_bertopic(texts, emb):
    """Возвращает (topics, topic_model). Может бросить ValueError (нет кластеров)."""
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer

    umap_model = UMAP(
        n_neighbors=15, n_components=5, min_dist=0.0,
        metric='cosine', random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=MIN_CLUSTER_SIZE_BT, min_samples=MIN_SAMPLES_BT,
        metric='euclidean', prediction_data=True,
    )
    vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=2)

    model = BERTopic(
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        embedding_model=None,
        top_n_words=TOP_N_WORDS,
        verbose=False,
    )
    topics, _ = model.fit_transform(texts, embeddings=emb)
    return topics, model


def topic_words(model, topic_id, n=5):
    if topic_id == -1:
        return None
    try:
        return ', '.join(w for w, _ in model.get_topic(topic_id)[:n])
    except Exception:
        return None


def main():
    conn = psycopg2.connect(DB_URL)
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS temp_second_level_bt")
        cur.execute(
            """
            CREATE TABLE temp_second_level_bt (
                train_id INTEGER NOT NULL,
                type_label INTEGER NOT NULL,
                bt_topic INTEGER NOT NULL,
                top_words TEXT
            )
            """
        )
    conn.commit()

    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT label FROM temp_hdbscan WHERE label != -1 ORDER BY label")
        type_labels = [r[0] for r in cur.fetchall()]
    if not type_labels:
        sys.exit('❌ No type labels – check Phase 2 output.')

    for t_label in type_labels:
        ids, texts, emb = fetch_type_data(conn, t_label)
        n = len(ids)
        if n == 0:
            continue
        try:
            topics, model = run_bertopic(texts, emb)
        except Exception as e:
            print(f'⚠️ Type {t_label}: BERTopic failed ({type(e).__name__}: {e}) — все в шум')
            topics, model = [-1] * n, None

        topics = np.asarray(topics)
        n_topics = len(set(topics.tolist()) - {-1})
        noise = int((topics == -1).sum())

        records = []
        words_cache = {}
        for i, tid in enumerate(ids):
            t = int(topics[i])
            if t not in words_cache:
                words_cache[t] = topic_words(model, t) if model else None
            records.append((int(tid), int(t_label), t, words_cache[t]))
        store_bt(conn, records)
        print(f'✅ Type {t_label:>2}: {n} docs → {n_topics} тем, шум {noise} ({noise / n:.0%})')
        if model:
            for t in sorted(set(topics.tolist()) - {-1}):
                cnt = int((topics == t).sum())
                print(f'      тема {t}: {cnt:>4} док. | {words_cache.get(t)}')

    conn.close()
    print('✅ Phase 4b (BERTopic) completed — temp_second_level_bt')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
