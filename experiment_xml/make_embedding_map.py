# make_embedding_map.py
# Визуализация распределения документов по кластерам как точек на плоскости.
#
# Итоговый кластер документа (приоритет):
#   is_marked → 90 «ГИС МТ»
#   есть в temp_xml_drift_assign → 'D{new_theme}' (новые темы дрейф-буфера)
#   иначе тема V3 → 'T{theme}'
#   V3 unknown без дрейф-темы → 'noise' (серые точки)
#
# Выход: experiment_xml/reports/embedding_map_pca.png (все 402K, PCA)
#        experiment_xml/reports/embedding_map_tsne.png (подвыборка 15K, PCA50→t-SNE)
#
# Запуск: python make_embedding_map.py [--tsne-n 15000]

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import psycopg2

from experiment_xml_config import DB_URL, MARKED_THEME  # noqa: E402

REPORTS_DIR = Path(__file__).resolve().parent / 'reports'

BASE_NAMES = {
    0: 'ВОЛС/линии связи', 1: 'уборка', 2: 'выдача МЦ',
    3: 'аварийно-восст. работы', 4: 'доп. работы', 5: 'логистика',
    6: 'логистика', 7: 'аренда', 8: 'приемка-передача',
    9: 'электроснабжение', 10: 'услуги связи', 11: 'поставка оборудования',
    12: 'акт сверки', 13: 'порт MetroEth', 15: 'регистрация ЮЛ',
    16: 'замена SIM', 17: 'доступ/интернет', 18: 'клининг',
    19: 'поставка шин', 20: 'каналы Ethernet', 21: 'реклама ТВ',
    22: 'ЕГРН', 23: 'PR-услуги', 24: 'HeadHunter', 25: 'закупка ТМЦ',
    26: 'вознаграждение', MARKED_THEME: 'ГИС МТ',
}


def load_data(conn):
    cur = conn.cursor()
    print("загрузка присвоений...")
    cur.execute("SELECT xml_id, assigned_theme, is_unknown FROM temp_xml_subject_clean_assign")
    base = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    cur.execute("SELECT xml_id, new_theme FROM temp_xml_drift_assign")
    drift = {r[0]: r[1] for r in cur.fetchall()}
    cur.execute("SELECT id, is_marked FROM temp_xml_canonical")
    marked = {r[0]: r[1] for r in cur.fetchall()}

    cur.execute("SELECT new_theme, top_words FROM temp_xml_drift_centroids")
    drift_words = {r[0]: (r[1] or '') for r in cur.fetchall()}

    print("загрузка эмбеддингов (402K × 768)...")
    cur.execute("SELECT xml_id, subject_clean_embedding FROM temp_xml_embeddings ORDER BY xml_id")
    rows = cur.fetchall()
    ids = np.array([r[0] for r in rows], dtype=np.int64)
    emb = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
    del rows

    labels = []
    for i in ids:
        if marked.get(int(i)):
            labels.append(f'T{MARKED_THEME}')
        elif int(i) in drift:
            labels.append(f'D{drift[int(i)]}')
        else:
            t, unk = base.get(int(i), (None, True))
            labels.append('noise' if unk else f'T{t}')
    return ids, emb, np.array(labels), drift_words


def cluster_display_names(labels, drift_words):
    """Короткие имена для топ-кластеров."""
    uniq, cnts = np.unique(labels, return_counts=True)
    order = sorted(zip(cnts, uniq), reverse=True)
    names = {}
    for c, lab in order:
        if lab == 'noise':
            names[lab] = 'остаточный шум'
        elif lab.startswith('T'):
            names[lab] = BASE_NAMES.get(int(lab[1:]), lab)
        elif lab.startswith('D') and lab[1:].isdigit():
            w = (drift_words.get(int(lab[1:]), '') or '').split(', ')
            names[lab] = 'D' + lab[1:] + ' ' + ' '.join(w[:3])
        else:
            names[lab] = lab  # L-подтемы и прочие — имя переопределяется снаружи
    return names, order


def plot(coords, labels, names, order, out_path, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(22, 14), dpi=130)
    top_n = 20
    top_labels = [lab for _, lab in order[:top_n]]
    cmap = plt.get_cmap('tab20')
    color_map = {lab: cmap(i % 20) for i, lab in enumerate(top_labels)}

    # noise и не-топ — серым фоном
    mask_bg = np.array([l not in top_labels for l in labels])
    ax.scatter(coords[mask_bg, 0], coords[mask_bg, 1], s=1, c='#c8c8c8',
               alpha=0.35, linewidths=0, rasterized=True)
    # топ-кластеры
    for lab in top_labels:
        m = labels == lab
        if lab == f'T{MARKED_THEME}':
            ax.scatter(coords[m, 0], coords[m, 1], s=25, c='red', marker='X',
                       alpha=0.95, linewidths=0, rasterized=True, zorder=10)
        else:
            ax.scatter(coords[m, 0], coords[m, 1], s=2, c=[color_map[lab]],
                       alpha=0.55, linewidths=0, rasterized=True)

    # подписи центроидов топ-12
    for lab in top_labels[:12]:
        m = labels == lab
        cx, cy = coords[m, 0].mean(), coords[m, 1].mean()
        ax.annotate(names[lab][:38], (cx, cy), fontsize=8, weight='bold',
                    bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.75, lw=0.3))

    legend = [Line2D([0], [0], marker='o', color='w',
                     markerfacecolor=('red' if lab == f'T{MARKED_THEME}' else color_map[lab]),
                     markersize=7, label=f"{names[lab][:45]} ({int(c):,})".replace(',', ' '))
              for c, lab in order[:top_n]]
    legend.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='#c8c8c8',
                         markersize=7, label='прочие/шум'))
    ax.legend(handles=legend, loc='upper right', fontsize=8, framealpha=0.9)
    ax.set_title(title, fontsize=13)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tsne-n', type=int, default=15000)
    args = ap.parse_args()

    conn = psycopg2.connect(DB_URL)
    ids, emb, labels, drift_words = load_data(conn)
    conn.close()
    names, order = cluster_display_names(labels, drift_words)
    print(f"документов: {len(ids)}, кластеров: {len(order)}")

    # ── PCA на всех точках ──
    from sklearn.decomposition import PCA
    t0 = time.time()
    pca = PCA(n_components=2, svd_solver='randomized', random_state=42)
    coords = pca.fit_transform(emb)
    print(f"PCA: {time.time()-t0:.0f} c, объяснённая дисперсия: {pca.explained_variance_ratio_.sum():.1%}")
    plot(coords, labels, names, order,
         REPORTS_DIR / 'embedding_map_pca.png',
         f'XML-документы на плоскости (PCA 2D, {len(ids):,} точек, {len(order)} кластеров)'.replace(',', ' '))

    # ── t-SNE на стратифицированной подвыборке (PCA→50 → t-SNE) ──
    if args.tsne_n > 0 and len(ids) > args.tsne_n:
        rng = np.random.default_rng(42)
        idx = np.sort(rng.choice(len(ids), size=args.tsne_n, replace=False))
        s_emb, s_labels = emb[idx], labels[idx]
    else:
        s_emb, s_labels = emb, labels
    from sklearn.manifold import TSNE
    t0 = time.time()
    emb50 = PCA(n_components=50, svd_solver='randomized', random_state=42).fit_transform(s_emb)
    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42,
                learning_rate='auto')
    coords2 = tsne.fit_transform(emb50)
    print(f"t-SNE ({len(s_emb)} точек): {time.time()-t0:.0f} c")
    names2, order2 = cluster_display_names(s_labels, drift_words)
    plot(coords2, s_labels, names2, order2,
         REPORTS_DIR / 'embedding_map_tsne.png',
         f'XML-документы на плоскости (t-SNE, подвыборка {len(s_emb):,})'.replace(',', ' '))

    print('готово')


if __name__ == '__main__':
    main()
