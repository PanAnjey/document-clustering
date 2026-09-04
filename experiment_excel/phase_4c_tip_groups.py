# phase_4c_tip_groups.py
# Канонизация свободных строк ТИП от LLM через HDBSCAN:
# 277 вариантов → ~15-30 канонических групп видов документов
# («Счёт»/«Счет», «Акт приема-передачи»/«Акт о приеме-передаче на хранение»).
#
# Строки ТИП короткие → эмбеддинг тривиален; дробление лечится тем же
# merge_by_centroids, что и для тем (phase_2d).
#
# Таблица: temp_tip_groups (tip TEXT PRIMARY KEY, group_id, group_name, docs)

import sys
import traceback

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

from experiment_config import DB_URL, EMBED_BATCH_SIZE  # noqa: sys.path+utf8
from embed_helper import encode_texts
from phase_2d_llm_themes import merge_by_centroids
from show_themes import normalize_tip
import hdbscan

MIN_CLUSTER_SIZE_TIP = 3   # группы из 3+ вариантов строк; одиночные — своя группа
MERGE_SIM_TIP = 0.92       # 0.88 склеивал разные виды актов в один


def fetch_tips(conn):
    """[(нормализованный tip, docs)] — distinct строки ТИП с частотами."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT tip, COUNT(*) FROM temp_llm_subject
            WHERE tip IS NOT NULL GROUP BY tip
        """)
        rows = cur.fetchall()
    by_tip = {}
    for tip, c in rows:
        n = normalize_tip(tip)
        by_tip[n] = by_tip.get(n, 0) + c
    return sorted(by_tip.items(), key=lambda kv: -kv[1])


def store(conn, records):
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS temp_tip_groups")
        cur.execute("""
            CREATE TABLE temp_tip_groups (
                tip TEXT PRIMARY KEY,
                group_id INTEGER NOT NULL,
                group_name TEXT NOT NULL,
                docs INTEGER NOT NULL
            )
        """)
        execute_values(cur,
            "INSERT INTO temp_tip_groups (tip, group_id, group_name, docs) VALUES %s",
            records, page_size=200)
    conn.commit()


def main():
    conn = psycopg2.connect(DB_URL)
    tips = fetch_tips(conn)
    if not tips:
        sys.exit('❌ temp_llm_subject пуст')
    names = [t for t, _ in tips]
    counts = {t: c for t, c in tips}
    print(f'Distinct ТИП (после нормализации): {len(names)}')

    emb = encode_texts(names, batch_size=EMBED_BATCH_SIZE)
    labels = hdbscan.HDBSCAN(
        metric='euclidean', min_cluster_size=MIN_CLUSTER_SIZE_TIP,
        core_dist_n_jobs=-1,
    ).fit_predict(emb)
    n_raw = len(set(labels.tolist()) - {-1})
    labels = merge_by_centroids(labels, emb, sim_threshold=MERGE_SIM_TIP)

    # Шум = одиночные частотные типы («Смета», «Счет на оплату») без вариантов.
    # HDBSCAN кладёт варианты в семьи, а standalone-строки — в шум: каждой такой
    # строке даём СОБСТВЕННУЮ каноническую группу (иначе 8.8k документов пропадут).
    next_id = int(labels.max()) + 1
    for i, l in enumerate(labels):
        if l == -1:
            labels[i] = next_id
            next_id += 1

    groups = {}
    for name, lab in zip(names, labels):
        groups.setdefault(int(lab), []).append(name)
    print(f'HDBSCAN: {n_raw} семей слились до {len([g for g in groups.values() if len(g) > 1])} '
          f'сложных групп + {sum(1 for g in groups.values() if len(g) == 1)} одиночных '
          f'= {len(groups)} канонических групп (sim≥{MERGE_SIM_TIP})')

    # Имя группы = самый частотный член; печать топ-25 по документам
    records = []
    by_docs = sorted(groups.items(),
                     key=lambda kv: -sum(counts[n] for n in kv[1]))
    for lab, members in by_docs:
        members = sorted(members, key=lambda n: -counts[n])
        gname = members[0]
        docs = sum(counts[n] for n in members)
        for n in members:
            records.append((n, lab, gname, counts[n]))
    for lab, members in by_docs[:25]:
        members = sorted(members, key=lambda n: -counts[n])
        gname = members[0]
        docs = sum(counts[n] for n in members)
        others = f'  ← {", ".join(members[1:8])}' if len(members) > 1 else ''
        more = f' +{len(members) - 8}' if len(members) > 8 else ''
        print(f'  группа {lab:>3}: {docs:>5} док. | {gname}{others}{more}')
    if len(by_docs) > 25:
        print(f'  ... ещё {len(by_docs) - 25} групп (в temp_tip_groups)')

    store(conn, records)
    conn.close()
    print('✅ Phase 4c completed — temp_tip_groups')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
