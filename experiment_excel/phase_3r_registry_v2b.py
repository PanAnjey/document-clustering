# phase_3r_registry_v2b.py
# Дополнение реестра v2 новыми темами из PDF_Tables (temp_pdftables_newthemes).
# Дедупликация: если центроид новой темы близок к существующему (sim ≥ MERGE_EXISTING),
# тема считается усилением существующей (новые документы мапятся на её label),
# иначе — добавляется новый label.

import sys
import traceback

import numpy as np
import psycopg2

from experiment_config import DB_URL, EMBED_BATCH_SIZE  # noqa
from embed_helper import encode_texts

MERGE_EXISTING = 0.90


def norm(v):
    return (v / max(np.linalg.norm(v), 1e-9)).astype(np.float32)


def main():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    # Новые темы из дрейф-буфера PDF_Tables
    cur.execute("""
        SELECT n.new_theme, s.predmet, w.top_words
        FROM temp_pdftables_newthemes n
        JOIN temp_llm_subject_pdftables s ON s.train_id = n.pdftext_id
        LEFT JOIN temp_pdftables_newthemes_top w ON w.new_theme = n.new_theme
        ORDER BY n.new_theme, n.pdftext_id
    """)
    rows = cur.fetchall()
    if not rows:
        sys.exit('❌ temp_pdftables_newthemes пуст')
    by_theme = {}
    for theme, predmet, words in rows:
        by_theme.setdefault(theme, {'words': words, 'texts': []})
        by_theme[theme]['texts'].append(predmet)
    print(f'Новых тем-кандидатов из PDF_Tables: {len(by_theme)}')

    # Существующий реестр v2
    cur.execute("SELECT theme_label, embedding, top_words FROM temp_theme2_centroids_v2")
    existing = cur.fetchall()
    ex_labels = [int(l) for l, _b, _w in existing]
    ex_cents = {int(l): norm(np.frombuffer(b, dtype=np.float32)) for l, b, _w in existing}
    next_label = max(ex_labels) + 1
    print(f'Реестр v2 сейчас: {len(ex_labels)} центроидов (max={max(ex_labels)})')

    added, merged = [], []
    for theme in sorted(by_theme, key=lambda t: -len(by_theme[t]['texts'])):
        texts = by_theme[theme]['texts']
        words = by_theme[theme]['words'] or ''
        emb = encode_texts(texts, batch_size=EMBED_BATCH_SIZE)
        centroid = norm(emb.mean(axis=0).astype(np.float32))
        # ближайший существующий
        best_lab, best_sim = None, -1.0
        for lab, c in ex_cents.items():
            s = float(centroid @ c)
            if s > best_sim:
                best_sim, best_lab = s, lab
        if best_sim >= MERGE_EXISTING:
            merged.append((theme, best_lab, best_sim, len(texts), words))
            print(f'  ~ тема {theme} ({len(texts)} док.) → существующая {best_lab} '
                  f'(sim={best_sim:.3f}) | {words[:55]}')
        else:
            cur.execute(
                "INSERT INTO temp_theme2_centroids_v2 (theme_label, embedding, origin, top_words) "
                "VALUES (%s, %s, %s, %s)",
                (next_label, centroid.tobytes(), 'pdftables_drift', words))
            ex_cents[next_label] = centroid
            added.append((next_label, len(texts), words, best_sim))
            print(f'  + тема {next_label}: {len(texts)} док. | {words[:60]} '
                  f'(ближайшая sim={best_sim:.3f})')
            next_label += 1
    conn.commit()

    cur.execute("SELECT origin, COUNT(*) FROM temp_theme2_centroids_v2 GROUP BY origin")
    print(f'\n✅ Реестр v2 итог: {dict(cur.fetchall())}, добавлено {len(added)}, '
          f'сматчено с существующими {len(merged)}')
    conn.close()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
