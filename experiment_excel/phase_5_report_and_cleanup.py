# phase_5_report_and_cleanup.py
import sys
import json
import datetime
import pathlib

import psycopg2

from experiment_config import DB_URL, SIMILARITY_THRESHOLD

ALL_TEMP_TABLES = [
    'temp_excel_train', 'temp_excel_test', 'temp_excel_failed',
    'temp_hdbscan', 'temp_hdbscan_centroids', 'temp_train_embeddings',
    'temp_test_assign', 'temp_second_level',
]


def table_exists(conn, name):
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass(%s)", (f"public.{name}",))
        return cur.fetchone()[0] is not None


def fetch_one(conn, query, params=None):
    """Обычный курсор → fetchone() возвращает tuple, индекс [0] корректен.
    (Баг исходной версии: RealDictCursor + fetchone()[0] → KeyError: 0.)"""
    with conn.cursor() as cur:
        cur.execute(query, params)
        row = cur.fetchone()
        return row[0] if row else None


def main():
    conn = psycopg2.connect(DB_URL)
    report = {}
    report['timestamp'] = datetime.datetime.now().isoformat()
    report['similarity_threshold'] = SIMILARITY_THRESHOLD

    # Basic counts (устойчиво к пропущенным фазам — проверяем наличие таблиц)
    if table_exists(conn, 'temp_excel_train'):
        report['total_train'] = fetch_one(conn, "SELECT COUNT(*) FROM temp_excel_train")
    if table_exists(conn, 'temp_excel_test'):
        report['total_test'] = fetch_one(conn, "SELECT COUNT(*) FROM temp_excel_test")
    if table_exists(conn, 'temp_excel_failed'):
        report['total_failed'] = fetch_one(conn, "SELECT COUNT(*) FROM temp_excel_failed")
    if table_exists(conn, 'temp_hdbscan'):
        # Шум (-1) — не тип, считаем отдельно
        report['hdbscan_types'] = fetch_one(
            conn, "SELECT COUNT(DISTINCT label) FROM temp_hdbscan WHERE label != -1")
        report['noise_docs'] = fetch_one(
            conn, "SELECT COUNT(*) FROM temp_hdbscan WHERE label = -1")

    # Coverage of test set — порог из конфига, а не захардкоженный 0.8
    if table_exists(conn, 'temp_test_assign'):
        total = fetch_one(conn, "SELECT COUNT(*) FROM temp_test_assign")
        covered = fetch_one(
            conn, "SELECT COUNT(*) FROM temp_test_assign WHERE similarity >= %s",
            (SIMILARITY_THRESHOLD,))
        report['test_coverage'] = covered / total if total else 0.0
        report['test_assigned'] = total
        with conn.cursor() as cur:
            cur.execute(
                "SELECT AVG(similarity), MIN(similarity), MAX(similarity) FROM temp_test_assign")
            sims = cur.fetchone()
        if sims and sims[0] is not None:
            report['similarity_avg'] = round(float(sims[0]), 4)
            report['similarity_min'] = round(float(sims[1]), 4)
            report['similarity_max'] = round(float(sims[2]), 4)
    else:
        report['test_coverage'] = None

    # Distribution of second-level clusters per type
    if table_exists(conn, 'temp_second_level'):
        with conn.cursor() as cur:
            cur.execute(
                "SELECT type_label, COUNT(DISTINCT sub_label), COUNT(*) "
                "FROM temp_second_level GROUP BY type_label ORDER BY type_label"
            )
            report['second_level'] = [
                {'type_label': r[0], 'sub_clusters': r[1], 'docs': r[2]}
                for r in cur.fetchall()
            ]

    # Write JSON report
    out_dir = pathlib.Path(r'D:\FileOrganizer\Reports')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f'excel_experiment_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'🗒️ Report written to {out_file}')

    # Cleanup ALL temporary tables (включая temp_train_embeddings из Phase 2)
    with conn.cursor() as cur:
        for t in ALL_TEMP_TABLES:
            cur.execute(f'DROP TABLE IF EXISTS {t}')
    conn.commit()
    conn.close()
    print('✅ All temporary tables dropped – database restored.')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print('❌ Error during reporting/cleanup:', e)
        sys.exit(1)
