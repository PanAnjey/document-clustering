# make_star_map.py
# «Космическая» визуализация кластеров: документы как звёзды, кластеры как
# галактики на тёмном фоне. t-SNE (подвыборка до 100K) + свечение плотности.
#
# Выход: experiment_xml/reports/embedding_star_map.png
# Запуск: python make_star_map.py [--n 100000]

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import psycopg2

from experiment_xml_config import DB_URL, MARKED_THEME  # noqa: E402
from make_embedding_map import load_data, cluster_display_names  # noqa: E402

REPORTS_DIR = Path(__file__).resolve().parent / 'reports'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=100000, help='размер подвыборки для t-SNE')
    args = ap.parse_args()

    conn = psycopg2.connect(DB_URL)
    ids, emb, labels, drift_words = load_data(conn)
    conn.close()
    names, order = cluster_display_names(labels, drift_words)
    print(f"документов: {len(ids)}, кластеров: {len(order)}")

    # ── подвыборка + t-SNE ──
    rng = np.random.default_rng(42)
    if len(ids) > args.n:
        idx = np.sort(rng.choice(len(ids), size=args.n, replace=False))
        s_emb, s_labels = emb[idx], labels[idx]
    else:
        s_emb, s_labels = emb, labels
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    t0 = time.time()
    emb50 = PCA(n_components=50, svd_solver='randomized', random_state=42).fit_transform(s_emb)
    coords = TSNE(n_components=2, perplexity=30, init='pca', random_state=42,
                  learning_rate='auto').fit_transform(emb50)
    print(f"t-SNE ({len(s_emb)} точек): {time.time()-t0:.0f} c")

    names2, order2 = cluster_display_names(s_labels, drift_words)

    # ── отрисовка ──
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.ndimage import gaussian_filter

    x, y = coords[:, 0], coords[:, 1]
    # обрезка выбросов по перцентилям
    x0, x1 = np.percentile(x, 0.3), np.percentile(x, 99.7)
    y0, y1 = np.percentile(y, 0.3), np.percentile(y, 99.7)
    mx = (x1 - x0) * 0.03
    x0, x1, y0, y1 = x0 - mx, x1 + mx, y0 - mx, y1 + mx

    fig, ax = plt.subplots(figsize=(24, 14), dpi=170, facecolor='black')
    ax.set_facecolor('black')
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # 1) туманность: свечение плотности ВСЕХ точек
    nb = 1100
    H, xe, ye = np.histogram2d(x, y, bins=nb, range=[[x0, x1], [y0, y1]])
    H = np.log1p(H)
    H = gaussian_filter(H, sigma=2.2)
    nebula = LinearSegmentedColormap.from_list(
        'nebula', ['#000000', '#050514', '#0d1b4c', '#3730a3', '#7c3aed',
                   '#c026d3', '#f97316', '#fde68a', '#ffffff'])
    ax.imshow(H.T, origin='lower', extent=[x0, x1, y0, y1], aspect='auto',
              cmap=nebula, alpha=0.92, interpolation='bilinear', zorder=1)

    # 2) звёзды кластеров: топ-14, двухслойное свечение
    top_labels = [lab for _, lab in order2[:14]]
    palette = plt.get_cmap('gist_ncar')
    colors = {lab: palette(0.08 + 0.86 * i / max(len(top_labels) - 1, 1))
              for i, lab in enumerate(top_labels)}
    sizes = rng.uniform(0.4, 1.6, size=len(s_labels)) ** 2
    for lab in top_labels:
        m = s_labels == lab
        c = colors[lab]
        if lab == f'T{MARKED_THEME}':
            ax.scatter(x[m], y[m], s=30, c='red', marker='X', alpha=0.95,
                       linewidths=0, zorder=12)
            continue
        # гало
        ax.scatter(x[m], y[m], s=sizes[m] * 6, color=c, alpha=0.10,
                   linewidths=0, zorder=3, rasterized=True)
        # ядра звёзд
        ax.scatter(x[m], y[m], s=sizes[m], color=c, alpha=0.75,
                   linewidths=0, zorder=4, rasterized=True)

    # 3) фоновые звёзды (шум + мелкие кластеры) — холодные белые точки
    bg = np.array([l not in top_labels for l in s_labels])
    ax.scatter(x[bg], y[bg], s=rng.uniform(0.2, 0.9, size=int(bg.sum())) ** 2,
               c='#dbeafe', alpha=0.45, linewidths=0, zorder=2, rasterized=True)

    # 4) подписи-созвездия у топ-12
    import matplotlib.patheffects as pe
    for lab in top_labels[:12]:
        m = s_labels == lab
        cx, cy = x[m].mean(), y[m].mean()
        txt = ax.text(cx, cy, names2[lab][:34], color='white', fontsize=10,
                      ha='center', weight='bold', zorder=15)
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground='#0d1b4c')])

    ttl = ax.text(0.5, 1.015, 'Галактики документов XML: 58 кластеров-тем '
                  f'(t-SNE, {len(s_labels):,} звёзд)'.replace(',', ' '),
                  color='#e0e7ff', fontsize=15, ha='center',
                  transform=ax.transAxes)
    fig.tight_layout()
    out = REPORTS_DIR / 'embedding_star_map.png'
    fig.savefig(out, dpi=170, facecolor='black', bbox_inches='tight')
    plt.close(fig)
    print(f"→ {out}")


if __name__ == '__main__':
    main()
