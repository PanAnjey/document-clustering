# phase_3r_registry_v2.py
# Реестр тем v2: 12 тем Раунда 2 (Excel) + новые темы из дрейф-буфера PDF_Text.
#
# Переоткрытие с min_cluster_size=10 (сбалансировано: не дробит, не сливает):
#   - кластеры ≥ MIN_DOCS документов → новые центроиды в реестре;
#   - кластеры-утечки типов («Счёт на оплату») исключаются вручную.
#
# Таблица: temp_theme2_centroids_v2 (theme_label PK, embedding BYTEA, origin, top_words)
#   origin: 'excel_r2' | 'pdftext_drift'

import sys
import traceback

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from experiment_config import DB_URL, EMBED_BATCH_SIZE  # noqa
from embed_helper import encode_texts
from phase_2c_themes import c_tfidf_top_words
from phase_2d_llm_themes import merge_by_centroids
import hdbscan

MIN_CLUSTER_SIZE = 10
MIN_DOCS = 10
# Кластеры-утечки типов документов (не темы) — исключить по top-словам
TYPE_LEAK_RX = ('счёт на оплату', 'счет на оплату')


def main():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    # 1) Неопознанные документы PDF_Text
    cur.execute("""
        SELECT a.pdftext_id, s.predmet
        FROM temp_pdftext_assign a
        JOIN temp_llm_subject_pdftext s ON s.train_id = a.pdftext_id
        WHERE a.is_unknown ORDER BY a.pdftext_id
    """)
    rows = cur.fetchall()
    if not rows:
        sys.exit('❌ Дрейф-буфер пуст')
    ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]
    print(f'Дрейф-буфер: {len(ids)} документов')

    emb = encode_texts(texts, batch_size=EMBED_BATCH_SIZE)
    labels = hdbscan.HDBSCAN(metric='euclidean', min_cluster_size=MIN_CLUSTER_SIZE,
                             core_dist_n_jobs=-1).fit_predict(emb)
    labels = merge_by_centroids(labels, emb, sim_threshold=0.92)

    real = [l for l in np.unique(labels) if l != -1]
    class_texts, order = [], []
    for lab in real:
        class_texts.append(' '.join(texts[i] for i in np.where(labels == lab)[0]))
        order.append(lab)
    tops = c_tfidf_top_words(class_texts, top_n=6) if real else []
    top_map = dict(zip(order, tops))

    # 2) Фильтр: ≥ MIN_DOCS и не утечка типа
    accepted = []
    for lab in real:
        cnt = int((labels == lab).sum())
        words = top_map.get(lab, '')
        if cnt < MIN_DOCS:
            print(f'  skip (мало, {cnt}): {words[:60]}')
            continue
        if any(leak in words.lower() for leak in TYPE_LEAK_RX):
            print(f'  skip (утечка типа, {cnt}): {words[:60]}')
            continue
        accepted.append((lab, cnt, words))

    if not accepted:
        sys.exit('Нет новых тем для реестра')

    # 3) Реестр v2 = старые центроиды + новые
    cur.execute("DROP TABLE IF EXISTS temp_theme2_centroids_v2")
    cur.execute("""
        CREATE TABLE temp_theme2_centroids_v2 (
            theme_label INTEGER PRIMARY KEY,
            embedding BYTEA NOT NULL,
            origin TEXT NOT NULL,
            top_words TEXT
        )""")
    cur.execute("SELECT c.theme_label, c.embedding, w.top_words FROM temp_theme2_centroids c "
                "LEFT JOIN temp_theme2_topwords w ON w.theme_label = c.theme_label")
    old = cur.fetchall()
    execute_values(cur,
        "INSERT INTO temp_theme2_centroids_v2 (theme_label, embedding, origin, top_words) VALUES %s",
        [(int(l), b, 'excel_r2', w) for l, b, w in old], page_size=50)
    max_label = max(int(l) for l, _b, _w in old)

    old_cents = {int(l): np.frombuffer(b, dtype=np.float32) for l, b, _w in old}
    print(f'\nСтарый реестр: {len(old)} центроидов (max label={max_label})')

    new_rows = []
    for i, (lab, cnt, words) in enumerate(accepted, 1):
        mask = labels == lab
        centroid = emb[mask].mean(axis=0).astype(np.float32)
        centroid = (centroid / max(np.linalg.norm(centroid), 1e-9)).astype(np.float32)
        new_label = max_label + i
        # Ближайший старый центроид (для отчёта о маппинге)
        best_old, best_sim = None, -1.0
        for ol, oc in old_cents.items():
            s = float(centroid @ oc / (np.linalg.norm(centroid) * np.linalg.norm(oc) + 1e-9))
            if s > best_sim:
                best_sim, best_old = s, ol
        new_rows.append((new_label, centroid.tobytes(), 'pdftext_drift', words))
        print(f'  + тема {new_label}: {cnt} док. | {words[:70]}')
        print(f'    ближайшая старая тема: {best_old} (sim={best_sim:.3f})')

    execute_values(cur,
        "INSERT INTO temp_theme2_centroids_v2 (theme_label, embedding, origin, top_words) VALUES %s",
        new_rows, page_size=50)
    conn.commit()
    cur.execute("SELECT origin, COUNT(*) FROM temp_theme2_centroids_v2 GROUP BY origin")
    print(f'\n✅ Реестр v2: {dict(cur.fetchall())}')
    conn.close()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
