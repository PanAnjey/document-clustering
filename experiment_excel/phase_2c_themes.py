# phase_2c_themes.py
# Раунд 2: ТЕМАТИЧЕСКАЯ кластеризация (направления деятельности первым уровнем).
#
# Отличие от phase_2: эмбеддинги строятся не по шапке документа (тип, реквизиты),
# а по ПРЕДМЕТНОМУ ФРАГМЕНТУ (subject_extractor_excel) — строки номенклатуры.
# Заранее таксономии нет: HDBSCAN сам обнаруживает направления, читаем их
# по top-словам c-TF-IDF (как в BERTopic, но без UMAP — на исходных векторах).
#
# Таблицы (типовые temp_hdbscan* НЕ трогаем):
#   temp_theme_clusters  (train_id PK, theme_label)
#   temp_theme_centroids (theme_label PK, embedding BYTEA)
#   temp_theme_topwords  (theme_label PK, top_words TEXT)

import sys
import traceback

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from experiment_config import DB_URL, EMBED_BATCH_SIZE, MIN_CLUSTER_SIZE_FACTOR  # noqa: sys.path+utf8
from embed_helper import encode_texts
from subject_extractor_excel import extract_subject_excel
import hdbscan

MAX_FALLBACK_LEN = 1000  # если предмет не найден — первые N символов текста


def fetch_training_texts(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT id, txt FROM temp_excel_train ORDER BY id")
        rows = cur.fetchall()
    ids, texts = zip(*rows) if rows else ([], [])
    return list(ids), list(texts)


def extract_subjects(texts):
    """Предметный фрагмент для каждого текста; fallback — начало текста."""
    subjects, n_fallback = [], 0
    for t in texts:
        s = extract_subject_excel(t or '')
        if s is None:
            s = (t or '')[:MAX_FALLBACK_LEN]
            n_fallback += 1
        subjects.append(s)
    return subjects, n_fallback


def c_tfidf_top_words(class_texts, top_n=8):
    """c-TF-IDF (BERTopic-style): top-слова класса. Чистые числа исключены
    (token_pattern — токены начинаются с буквы), чтобы top-слова были
    предметными, а не датами/номерами."""
    from sklearn.feature_extraction.text import CountVectorizer
    vec = CountVectorizer(token_pattern=r'(?u)\b[^\W\d_]\w+\b', ngram_range=(1, 2))
    X = vec.fit_transform(class_texts)
    words = vec.get_feature_names_out()
    tf = X.toarray().astype(np.float64)
    df = np.asarray((X > 0).sum(axis=0)).ravel()
    avg_words_per_class = tf.sum() / max(len(class_texts), 1)
    idf = np.log(1 + avg_words_per_class / np.maximum(df, 1))
    ctfidf = tf * idf
    tops = []
    for i in range(len(class_texts)):
        idx = ctfidf[i].argsort()[::-1][:top_n]
        tops.append(', '.join(words[j] for j in idx if ctfidf[i, j] > 0))
    return tops


def store(conn, ids, labels, emb, top_words_map):
    with conn.cursor() as cur:
        execute_values(cur,
            "INSERT INTO temp_theme_clusters (train_id, theme_label) VALUES %s",
            [(int(i), int(l)) for i, l in zip(ids, labels)], page_size=500)
        real = [l for l in np.unique(labels) if l != -1]
        cent = []
        for lab in real:
            c = emb[labels == lab].mean(axis=0).astype(np.float32)
            cent.append((int(lab), c.tobytes()))
        if cent:
            execute_values(cur,
                "INSERT INTO temp_theme_centroids (theme_label, embedding) VALUES %s",
                cent, page_size=200)
        if top_words_map:
            execute_values(cur,
                "INSERT INTO temp_theme_topwords (theme_label, top_words) VALUES %s",
                [(int(l), w) for l, w in top_words_map.items()], page_size=100)
    conn.commit()


def main():
    conn = psycopg2.connect(DB_URL)
    with conn.cursor() as cur:
        for t in ('temp_theme_clusters', 'temp_theme_centroids', 'temp_theme_topwords'):
            cur.execute(f"DROP TABLE IF EXISTS {t}")
        cur.execute("CREATE TABLE temp_theme_clusters (train_id INTEGER PRIMARY KEY, theme_label INTEGER NOT NULL)")
        cur.execute("CREATE TABLE temp_theme_centroids (theme_label INTEGER PRIMARY KEY, embedding BYTEA NOT NULL)")
        cur.execute("CREATE TABLE temp_theme_topwords (theme_label INTEGER PRIMARY KEY, top_words TEXT NOT NULL)")
    conn.commit()

    ids, texts = fetch_training_texts(conn)
    if not ids:
        sys.exit('❌ No training texts – abort.')

    subjects, n_fallback = extract_subjects(texts)
    print(f'Subject extraction: {len(subjects) - n_fallback} предметных фрагментов, '
          f'{n_fallback} fallback ({n_fallback / len(subjects):.1%})')

    emb = encode_texts(subjects, batch_size=EMBED_BATCH_SIZE)
    n = len(emb)
    min_cluster = max(2, int(MIN_CLUSTER_SIZE_FACTOR * n))
    print(f'HDBSCAN: n={n}, min_cluster_size={min_cluster}')
    labels = hdbscan.HDBSCAN(
        metric='euclidean', min_cluster_size=min_cluster, core_dist_n_jobs=-1,
    ).fit_predict(emb)

    real_labels = [l for l in np.unique(labels) if l != -1]
    noise = int((labels == -1).sum())
    print(f'HDBSCAN done: {len(real_labels)} тем, шум {noise} ({noise / n:.1%})')

    # Top-слова тем (c-TF-IDF по склеенным предметным фрагментам темы)
    top_words_map = {}
    if real_labels:
        class_texts, order = [], []
        for lab in real_labels:
            class_texts.append(' '.join(subjects[i] for i in np.where(labels == lab)[0]))
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
    print('✅ Phase 2c (themes) completed — temp_theme_*')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
