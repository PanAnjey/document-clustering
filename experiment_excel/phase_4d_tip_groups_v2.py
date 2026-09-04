# phase_4d_tip_groups_v2.py
# Канонизация ТИП документов объединённо: Excel + PDF_Text + PDF_Tables.
# Тот же метод (phase_4c): нормализация → эмбеддинг distinct строк →
# HDBSCAN → merge_by_centroids(0.92) → одиночные = свои группы.
#
# Таблица: temp_tip_groups_v2
#   (tip TEXT PK, group_id, group_name, docs_excel, docs_pdftext, docs_pdftables)

import sys
import traceback

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from experiment_config import DB_URL, EMBED_BATCH_SIZE  # noqa
from embed_helper import encode_texts
from phase_2d_llm_themes import merge_by_centroids
from show_themes import normalize_tip
import hdbscan

MIN_CLUSTER_SIZE_TIP = 3
MERGE_SIM_TIP = 0.92


def fetch_tips(conn):
    """{нормализованный tip: [docs_excel, docs_pdftext, docs_pdftables]}"""
    agg = {}
    for table, idx in (('temp_llm_subject', 0),
                       ('temp_llm_subject_pdftext', 1),
                       ('temp_llm_subject_pdftables', 2)):
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT tip, COUNT(*) FROM {table}
                WHERE tip IS NOT NULL GROUP BY tip
            """)
            for tip, c in cur.fetchall():
                n = normalize_tip(tip)
                agg.setdefault(n, [0, 0, 0])
                agg[n][idx] += c
    return sorted(agg.items(), key=lambda kv: -sum(kv[1]))


def main():
    conn = psycopg2.connect(DB_URL)
    tips = fetch_tips(conn)
    if not tips:
        sys.exit('❌ Нет данных ТИП')
    names = [t for t, _ in tips]
    counts = {t: c for t, c in tips}
    totals = {t: sum(c) for t, c in tips}
    print(f'Distinct ТИП (объединённо, после нормализации): {len(names)}')

    emb = encode_texts(names, batch_size=EMBED_BATCH_SIZE)
    labels = hdbscan.HDBSCAN(
        metric='euclidean', min_cluster_size=MIN_CLUSTER_SIZE_TIP,
        core_dist_n_jobs=-1,
    ).fit_predict(emb)
    labels = merge_by_centroids(labels, emb, sim_threshold=MERGE_SIM_TIP)

    # Одиночные (шум) → собственные группы
    next_id = int(labels.max()) + 1
    for i, l in enumerate(labels):
        if l == -1:
            labels[i] = next_id
            next_id += 1

    groups = {}
    for name, lab in zip(names, labels):
        groups.setdefault(int(lab), []).append(name)
    n_families = sum(1 for g in groups.values() if len(g) > 1)
    print(f'HDBSCAN: {n_families} семей + {len(groups) - n_families} одиночных '
          f'= {len(groups)} канонических групп')

    # Имя группы = самый частотный член
    records = []
    by_docs = sorted(groups.items(),
                     key=lambda kv: -sum(totals[n] for n in kv[1]))
    for lab, members in by_docs:
        members = sorted(members, key=lambda n: -totals[n])
        gname = members[0]
        for n in members:
            de, dt, dtb = counts[n]
            records.append((n, lab, gname, de, dt, dtb))

    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS temp_tip_groups_v2")
        cur.execute("""
            CREATE TABLE temp_tip_groups_v2 (
                tip TEXT PRIMARY KEY,
                group_id INTEGER NOT NULL,
                group_name TEXT NOT NULL,
                docs_excel INTEGER NOT NULL,
                docs_pdftext INTEGER NOT NULL,
                docs_pdftables INTEGER NOT NULL
            )
        """)
        execute_values(cur,
            "INSERT INTO temp_tip_groups_v2 (tip, group_id, group_name, docs_excel, docs_pdftext, docs_pdftables) VALUES %s",
            records, page_size=200)
    conn.commit()

    # Печать: топ-25 общих + PDF-специфичные новые группы
    print('\n--- Топ-25 канонических типов (все потоки) ---')
    print(f'{"группа":<26} {"всего":>6} {"excel":>6} {"pdftext":>8} {"pdftables":>9}')
    for lab, members in by_docs[:25]:
        members = sorted(members, key=lambda n: -totals[n])
        gname = members[0]
        de = sum(counts[n][0] for n in members)
        dt = sum(counts[n][1] for n in members)
        dtb = sum(counts[n][2] for n in members)
        print(f'{gname[:24]:<26} {de + dt + dtb:>6} {de:>6} {dt:>8} {dtb:>9}')

    print('\n--- Новые типы ИЗ PDF (нет в Excel) — топ-20 ---')
    pdf_only = [(lab, members) for lab, members in by_docs
                if all(counts[n][0] == 0 for n in members)]
    for lab, members in pdf_only[:20]:
        members = sorted(members, key=lambda n: -totals[n])
        gname = members[0]
        dt = sum(counts[n][1] for n in members)
        dtb = sum(counts[n][2] for n in members)
        others = f'  ← {", ".join(members[1:6])}' if len(members) > 1 else ''
        print(f'{gname[:40]:<42} {dt + dtb:>6} (text {dt}, tables {dtb}){others}')

    conn.close()
    print('\n✅ Phase 4d completed — temp_tip_groups_v2')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
