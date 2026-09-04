# phase_m2_sample.py
# Фаза M2: выборка ~50K документов с чистыми метками тем для документного
# теста. Источник: temp_xml_canonical ⋈ temp_xml_subject_clean_assign (V3).
#
# Метки тем получены в пространстве v1.5 (оговорка в отчёте: лёгкое
# преимущество baseline). Берём sim >= 0.85 (чистые), балансировка —
# не более CAP_PER_THEME на тему (топ по similarity).
#
# Выход: sample_50k.csv (id, theme, similarity, subject_text).

import sys

import pandas as pd
import psycopg2

from em_config import (  # noqa: E402
    DB_URL, T_CANONICAL, T_ASSIGN, MIN_SIMILARITY, MIN_SUBJECT_LEN,
    CAP_PER_THEME, SAMPLE_CSV,
)


def main():
    conn = psycopg2.connect(DB_URL)
    print("fetching candidates from PostgreSQL...")
    df = pd.read_sql(f"""
        SELECT c.id, a.assigned_theme AS theme, a.similarity, c.subject_text
        FROM {T_ASSIGN} a
        JOIN {T_CANONICAL} c ON c.id = a.xml_id
        WHERE ((a.similarity >= %s AND NOT a.is_unknown)
               OR a.assigned_theme = 90)   -- кластер ГИС МТ: override, sim не важна
          AND c.subject_len >= %s
        ORDER BY a.similarity DESC
    """, conn, params=(MIN_SIMILARITY, MIN_SUBJECT_LEN))
    conn.close()
    print(f"  candidates: {len(df)} документов, {df['theme'].nunique()} тем")

    # Балансировка: топ-CAP_PER_THEME по similarity на тему
    sample = (df.groupby('theme', group_keys=False)
                .head(CAP_PER_THEME)
                .sort_values('id')
                .reset_index(drop=True))
    sample['subject_text'] = sample['subject_text'].fillna('')

    sample.to_csv(SAMPLE_CSV, index=False, encoding='utf-8')
    print(f"  sample: {len(sample)} документов → {SAMPLE_CSV}")
    print("\n  распределение по темам:")
    for theme, cnt in sample['theme'].value_counts().sort_index().items():
        print(f"    тема {theme:>3}: {cnt}")


if __name__ == '__main__':
    main()
