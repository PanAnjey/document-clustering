# phase_3r_assign.py
# Раунд 3, шаг 2: назначение PDF к темам Раунда 2 + переоткрытие новых тем.
#
#   ПРЕДМЕТ → nomic-эмбеддинг → центроиды temp_theme2_centroids → sim ≥ 0.8:
#   назначение теме; sim < 0.8: дрейф-буфер → HDBSCAN → НОВЫЕ темы.
#
# Запуск:
#   python phase_3r_assign.py                                  # PDF_Text
#   python phase_3r_assign.py --subjects temp_llm_subject_pdftables --prefix temp_pdftables

import argparse
import sys
import traceback

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from experiment_config import DB_URL, EMBED_BATCH_SIZE, EMBED_DIM, SIMILARITY_THRESHOLD  # noqa
from embed_helper import encode_texts
from phase_2c_themes import c_tfidf_top_words
from phase_2d_llm_themes import merge_by_centroids
import hdbscan

MIN_NEW_THEME_SIZE = 30  # min_cluster_size для переоткрытия тем в дрейф-буфере


def fetch_subjects(conn, table):
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT train_id, predmet FROM {table}
            WHERE predmet IS NOT NULL ORDER BY train_id
        """)
        rows = cur.fetchall()
    return [r[0] for r in rows], [r[1] for r in rows]


def fetch_centroids(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT theme_label, embedding FROM temp_theme2_centroids")
        rows = cur.fetchall()
    labels = [r[0] for r in rows]
    cents = [np.frombuffer(r[1], dtype=np.float32) for r in rows]
    return labels, (np.stack(cents) if cents else np.empty((0, EMBED_DIM), np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subjects', default='temp_llm_subject_pdftext',
                    help='таблица LLM-субъектов (train_id, predmet)')
    ap.add_argument('--prefix', default='temp_pdftext',
                    help='префикс выходных таблиц: {prefix}_assign/_newthemes/_newthemes_top')
    args = ap.parse_args()
    SUBJ, PFX = args.subjects, args.prefix

    conn = psycopg2.connect(DB_URL)
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {PFX}_assign")
        cur.execute(f"DROP TABLE IF EXISTS {PFX}_newthemes")
        cur.execute(f"DROP TABLE IF EXISTS {PFX}_newthemes_top")
        cur.execute(f"""
            CREATE TABLE {PFX}_assign (
                pdftext_id INTEGER PRIMARY KEY,
                assigned_theme INTEGER NOT NULL,
                similarity REAL NOT NULL,
                is_unknown BOOLEAN NOT NULL
            )""")
        cur.execute(f"""
            CREATE TABLE {PFX}_newthemes (
                pdftext_id INTEGER PRIMARY KEY,
                new_theme INTEGER NOT NULL
            )""")
        cur.execute(f"""
            CREATE TABLE {PFX}_newthemes_top (
                new_theme INTEGER PRIMARY KEY,
                top_words TEXT NOT NULL
            )""")
    conn.commit()

    ids, texts = fetch_subjects(conn, SUBJ)
    labels, centroids = fetch_centroids(conn)
    if not ids or centroids.size == 0:
        sys.exit('❌ Нет данных (subjects / temp_theme2_centroids)')
    print(f'{SUBJ}: {len(ids)} документов, центроидов Раунда 2: {len(labels)}')

    emb = encode_texts(texts, batch_size=EMBED_BATCH_SIZE)

    from sklearn.metrics import pairwise
    sims = 1 - pairwise.cosine_distances(emb, centroids)
    best_idx = sims.argmax(axis=1)
    assigned = [labels[i] for i in best_idx]
    best = sims.max(axis=1)
    unknown_mask = best < SIMILARITY_THRESHOLD

    rows = [(int(tid), int(lab), float(s), bool(u))
            for tid, lab, s, u in zip(ids, assigned, best, unknown_mask)]
    with conn.cursor() as cur:
        execute_values(cur,
            f"INSERT INTO {PFX}_assign (pdftext_id, assigned_theme, similarity, is_unknown) VALUES %s",
            rows, page_size=500)
    conn.commit()

    n_unk = int(unknown_mask.sum())
    coverage = 1 - n_unk / len(best)
    print(f'\n✅ Назначение: coverage @ {SIMILARITY_THRESHOLD}: {coverage:.2%} '
          f'({len(best) - n_unk}/{len(best)}), дрейф-буфер: {n_unk} ({n_unk / len(best):.1%})')
    print(f'   sim avg={best.mean():.3f}, min={best.min():.3f}, max={best.max():.3f}')

    # per-theme stats
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT a.assigned_theme, COUNT(*) c, AVG(a.similarity) s, MAX(w.top_words)
            FROM {PFX}_assign a
            LEFT JOIN temp_theme2_topwords w ON w.theme_label = a.assigned_theme
            GROUP BY a.assigned_theme ORDER BY c DESC
        """)
        print('   per-theme:')
        for lab, c, s, words in cur.fetchall():
            print(f'     тема {lab:>2}: {c:>5} док. avg_sim={s:.3f} | {(words or "")[:55]}')

    # ── Переоткрытие новых тем из дрейф-буфера ──
    if n_unk >= MIN_NEW_THEME_SIZE * 2:
        unk_emb = emb[unknown_mask]
        unk_ids = np.array(ids)[unknown_mask]
        unk_texts = [texts[i] for i in np.where(unknown_mask)[0]]
        print(f'\nПереоткрытие тем по {n_unk} неопознанным (min_cluster_size={MIN_NEW_THEME_SIZE})...')
        new_labels = hdbscan.HDBSCAN(
            metric='euclidean', min_cluster_size=MIN_NEW_THEME_SIZE,
            core_dist_n_jobs=-1,
        ).fit_predict(unk_emb)
        n_raw = len(set(new_labels.tolist()) - {-1})
        new_labels = merge_by_centroids(new_labels, unk_emb, sim_threshold=0.92)
        real = [l for l in np.unique(new_labels) if l != -1]
        residual = int((new_labels == -1).sum())
        print(f'  сырых {n_raw} → {len(real)} новых тем после слияния; '
              f'остаточный шум {residual} ({residual / n_unk:.0%} буфера)')

        # top-слова новых тем
        top_map = {}
        if real:
            class_texts, order = [], []
            for lab in real:
                class_texts.append(' '.join(unk_texts[i] for i in np.where(new_labels == lab)[0]))
                order.append(lab)
            tops = c_tfidf_top_words(class_texts, top_n=8)
            top_map = dict(zip(order, tops))

        with conn.cursor() as cur:
            execute_values(cur,
                f"INSERT INTO {PFX}_newthemes (pdftext_id, new_theme) VALUES %s",
                [(int(tid), int(l)) for tid, l in zip(unk_ids, new_labels) if l != -1],
                page_size=500)
            if top_map:
                execute_values(cur,
                    f"INSERT INTO {PFX}_newthemes_top (new_theme, top_words) VALUES %s",
                    [(int(l), w) for l, w in top_map.items()], page_size=100)
        conn.commit()

        for lab in sorted(real, key=lambda l: -int((new_labels == l).sum())):
            cnt = int((new_labels == lab).sum())
            print(f'  НОВАЯ тема {lab:>2}: {cnt:>4} док. | {top_map.get(lab, "")}')
    else:
        print(f'\nДрейф-буфер слишком мал для переоткрытия ({n_unk} док.)')

    conn.close()
    print('✅ Phase 3r assign completed')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
