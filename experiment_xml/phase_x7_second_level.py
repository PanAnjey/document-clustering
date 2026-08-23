# phase_x7_second_level.py
# Этап X7: второй уровень кластеризации — дробление широких тем V3
# (≥ SPLIT_MIN документов) на подтемы. Аналог
# experiment_excel/phase_4_second_level_clustering.py (KMeans, адаптивный k),
# но MiniBatchKMeans для больших тем и c-TF-IDF топ-слова по подтемам.
#
# Мотивация: тема 11 «поставка оборудования» на t-SNE распалась на 2 острова —
# центроидное присвоение скрывает подструктуру. Второй уровень её раскрывает.
#
# Таблицы:
#   temp_xml_l2_assign    (xml_id PK, theme, sub_theme)
#   temp_xml_l2_centroids (theme, sub_theme, embedding BYTEA, size, top_words)
#
# Запуск: python phase_x7_second_level.py [--split-min 5000]

import argparse
import sys
import time

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from experiment_xml_config import (  # noqa: E402
    DB_URL, T_CANONICAL, T_EMBEDDINGS, MARKED_THEME,
)
from xml_canonicalizer import strip_contract_refs  # noqa: E402

import re as _re

# Очистка текстов для c-TF-IDF: убираем служебные токены V2-extras
# (вид:услуга, ед:шт) и короткие предлоги — иначе топ-слова подтем
# маскируются мусором «ед, вид, услуга».
_TFIDF_COLON_RE = _re.compile(r'\b(?:вид|ед):[^\s;\]}]+')
_TFIDF_JUNK_RE = _re.compile(
    r'(?<![\w])(?:на|по|за|от|в|к|с|и|об|до|из|вид|ед|шт)(?![\w])', _re.IGNORECASE)


def tfidf_clean(text):
    t = _TFIDF_COLON_RE.sub(' ', text or '')
    t = _TFIDF_JUNK_RE.sub(' ', t)
    return t

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent.parent / 'experiment_excel'))
from phase_2c_themes import c_tfidf_top_words  # noqa: E402

SPLIT_MIN_DEFAULT = 5000
MAX_K = 8
DOCS_PER_CLUSTER = 12000   # k = max(2, min(MAX_K, round(N / DOCS_PER_CLUSTER)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split-min', type=int, default=SPLIT_MIN_DEFAULT)
    args = ap.parse_args()

    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    # темы-кандидаты на дробление (без ГИС МТ=90 и без unknown)
    cur.execute("""
        SELECT assigned_theme, COUNT(*) FROM temp_xml_subject_clean_assign
        WHERE NOT is_unknown AND assigned_theme != %s
        GROUP BY assigned_theme HAVING COUNT(*) >= %s
        ORDER BY COUNT(*) DESC""", (MARKED_THEME, args.split_min))
    themes = [(t, n) for t, n in cur.fetchall()]
    print(f"тем к дроблению (≥{args.split_min}): {len(themes)}")
    print("  " + ", ".join(f"{t}:{n}" for t, n in themes))

    cur.execute("DROP TABLE IF EXISTS temp_xml_l2_assign")
    cur.execute("""CREATE TABLE temp_xml_l2_assign (
        xml_id INTEGER PRIMARY KEY, theme INTEGER NOT NULL, sub_theme INTEGER NOT NULL)""")
    cur.execute("DROP TABLE IF EXISTS temp_xml_l2_centroids")
    cur.execute("""CREATE TABLE temp_xml_l2_centroids (
        theme INTEGER NOT NULL, sub_theme INTEGER NOT NULL,
        embedding BYTEA NOT NULL, size INTEGER NOT NULL, top_words TEXT,
        PRIMARY KEY (theme, sub_theme))""")
    conn.commit()

    for theme, n_docs in themes:
        t0 = time.time()
        cur.execute(f"""
            SELECT a.xml_id, e.subject_clean_embedding, c.subject_text
            FROM temp_xml_subject_clean_assign a
            JOIN {T_EMBEDDINGS} e ON e.xml_id = a.xml_id
            JOIN {T_CANONICAL} c ON c.id = a.xml_id
            WHERE a.assigned_theme = %s AND NOT a.is_unknown
            ORDER BY a.xml_id""", (theme,))
        rows = cur.fetchall()
        ids = np.array([r[0] for r in rows], dtype=np.int64)
        emb = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
        texts = [tfidf_clean(strip_contract_refs(r[2] or '')) for r in rows]
        del rows

        n = len(ids)
        k = max(2, min(MAX_K, round(n / DOCS_PER_CLUSTER)))
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=4096,
                             n_init=3, max_iter=200)
        labels = km.fit_predict(emb)

        # центроиды (L2-норм) + топ-слова
        cents = []
        for lab in range(k):
            c = emb[labels == lab].mean(axis=0)
            cents.append(c / (np.linalg.norm(c) + 1e-12))
        class_texts = [' '.join(texts[i] for i in np.where(labels == lab)[0])
                       for lab in range(k)]
        tops = c_tfidf_top_words(class_texts, top_n=8)

        execute_values(cur,
                       "INSERT INTO temp_xml_l2_assign (xml_id, theme, sub_theme) VALUES %s",
                       [(int(i), int(theme), int(l)) for i, l in zip(ids, labels)],
                       page_size=1000)
        execute_values(cur,
                       """INSERT INTO temp_xml_l2_centroids
                          (theme, sub_theme, embedding, size, top_words) VALUES %s""",
                       [(int(theme), int(lab), cents[lab].astype(np.float32).tobytes(),
                         int((labels == lab).sum()), tops[lab]) for lab in range(k)],
                       page_size=100)
        conn.commit()
        sizes = sorted([int((labels == l).sum()) for l in range(k)], reverse=True)
        print(f"  тема {theme:>2}: {n} → k={k} | размеры {sizes} | {time.time()-t0:.0f} c")

    # итоговый отчёт
    cur.execute("""
        SELECT theme, sub_theme, size, top_words FROM temp_xml_l2_centroids
        ORDER BY theme, size DESC""")
    cur_theme = None
    print("\n== подтемы ==")
    for theme, sub, size, words in cur.fetchall():
        if theme != cur_theme:
            print(f" тема {theme}:")
            cur_theme = theme
        print(f"   {theme}.{sub}: {size:>7} | {(words or '')[:75]}")
    conn.close()
    print('Phase X7 completed')


if __name__ == '__main__':
    main()
