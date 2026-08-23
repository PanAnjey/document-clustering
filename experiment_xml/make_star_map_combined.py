# make_star_map_combined.py
# Совместная «космическая» карта: XML (формализованные) как цветные галактики,
# PDF (неформализованные) как белые/золотые маркеры поверх — визуализация
# того, куда попадают неформализованные документы относительно XML-кластеров.
#
# t-SNE считается на ОБЪЕДИНЁННОМ наборе (значимые взаимные расстояния).
#
# Выход: experiment_xml/reports/embedding_star_map_combined.png
# Запуск: python make_star_map_combined.py [--xml-n 80000] [--pdftables-n 30000]

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import psycopg2

from experiment_xml_config import DB_URL, MARKED_THEME  # noqa: E402
from make_embedding_map import load_data, cluster_display_names  # noqa: E402

REPORTS_DIR = Path(__file__).resolve().parent / 'reports'


def load_pdf(conn):
    cur = conn.cursor()
    out = {}
    for src, assign_t, newth_t in (
            ('pdftext', 'temp_pdftext_assign_v2', 'temp_pdftext_newthemes'),
            ('pdftables', 'temp_pdftables_assign_v2', 'temp_pdftables_newthemes')):
        cur.execute(f"""CREATE TEMP TABLE _tmp_ids AS
                        SELECT doc_id, embedding FROM temp_pdf_embeddings WHERE source=%s""", (src,))
        cur.execute(f"""
            SELECT i.doc_id, i.embedding, a.assigned_theme, a.is_unknown,
                   (SELECT n.new_theme FROM {newth_t} n WHERE n.pdftext_id = i.doc_id) nt
            FROM _tmp_ids i
            JOIN {assign_t} a ON a.pdftext_id = i.doc_id
            ORDER BY i.doc_id""")
        rows = cur.fetchall()
        cur.execute("DROP TABLE _tmp_ids")
        ids = np.array([r[0] for r in rows], dtype=np.int64)
        emb = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
        labels = []
        for _, _, t, unk, nt in rows:
            if not unk:
                labels.append(f'T{t}')
            elif nt is not None:
                labels.append(f'P{nt}')
            else:
                labels.append('noise')
        out[src] = (ids, emb, np.array(labels))
        print(f"  {src}: {len(ids)} док.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xml-n', type=int, default=80000)
    ap.add_argument('--pdftables-n', type=int, default=30000)
    args = ap.parse_args()

    conn = psycopg2.connect(DB_URL)
    ids, emb, labels, drift_words = load_data(conn)
    pdf = load_pdf(conn)
    conn.close()
    names, order = cluster_display_names(labels, drift_words)

    rng = np.random.default_rng(42)
    # подвыборка XML
    if len(ids) > args.xml_n:
        idx = np.sort(rng.choice(len(ids), size=args.xml_n, replace=False))
        x_emb, x_labels = emb[idx], labels[idx]
    else:
        x_emb, x_labels = emb, labels
    # подвыборка pdftables
    pt_ids, pt_emb, pt_labels = pdf['pdftables']
    if len(pt_ids) > args.pdftables_n:
        idx2 = np.sort(rng.choice(len(pt_ids), size=args.pdftables_n, replace=False))
        pt_emb, pt_labels = pt_emb[idx2], pt_labels[idx2]
    px_emb, px_labels = pdf['pdftext'][1], pdf['pdftext'][2]

    # ── объединённый t-SNE ──
    all_emb = np.vstack([x_emb, px_emb, pt_emb])
    print(f"t-SNE на объединённом наборе: {len(all_emb)} точек "
          f"(XML {len(x_emb)}, PDF_Text {len(px_emb)}, PDF_Tables {len(pt_emb)})")
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    t0 = time.time()
    emb50 = PCA(n_components=50, svd_solver='randomized', random_state=42).fit_transform(all_emb)
    coords = TSNE(n_components=2, perplexity=30, init='pca', random_state=42,
                  learning_rate='auto').fit_transform(emb50)
    print(f"  {time.time()-t0:.0f} c")
    n1, n2 = len(x_emb), len(px_emb)
    xy_xml = coords[:n1]
    xy_ptext = coords[n1:n1 + n2]
    xy_ptab = coords[n1 + n2:]

    names2, order2 = cluster_display_names(x_labels, drift_words)

    # ── отрисовка ──
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from scipy.ndimage import gaussian_filter

    x, y = coords[:, 0], coords[:, 1]
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

    # туманность по всем точкам
    nb = 1100
    H, _, _ = np.histogram2d(x, y, bins=nb, range=[[x0, x1], [y0, y1]])
    H = gaussian_filter(np.log1p(H), sigma=2.2)
    nebula = LinearSegmentedColormap.from_list(
        'nebula', ['#000000', '#050514', '#0d1b4c', '#3730a3', '#7c3aed',
                   '#c026d3', '#f97316', '#fde68a', '#ffffff'])
    ax.imshow(H.T, origin='lower', extent=[x0, x1, y0, y1], aspect='auto',
              cmap=nebula, alpha=0.92, interpolation='bilinear', zorder=1)

    # XML галактики (топ-14)
    top_labels = [lab for _, lab in order2[:14]]
    palette = plt.get_cmap('gist_ncar')
    colors = {lab: palette(0.08 + 0.86 * i / max(len(top_labels) - 1, 1))
              for i, lab in enumerate(top_labels)}
    xx, yy = xy_xml[:, 0], xy_xml[:, 1]
    sizes = rng.uniform(0.4, 1.6, size=len(x_labels)) ** 2
    for lab in top_labels:
        m = x_labels == lab
        c = colors[lab]
        if lab == f'T{MARKED_THEME}':
            ax.scatter(xx[m], yy[m], s=30, c='red', marker='X', alpha=0.95,
                       linewidths=0, zorder=12)
            continue
        ax.scatter(xx[m], yy[m], s=sizes[m] * 6, color=c, alpha=0.10,
                   linewidths=0, zorder=3, rasterized=True)
        ax.scatter(xx[m], yy[m], s=sizes[m], color=c, alpha=0.7,
                   linewidths=0, zorder=4, rasterized=True)
    bg = np.array([l not in top_labels for l in x_labels])
    ax.scatter(xx[bg], yy[bg], s=0.3, c='#dbeafe', alpha=0.4,
               linewidths=0, zorder=2, rasterized=True)

    # PDF_Text — белые плюсы, PDF_Tables — золотые треугольники
    ax.scatter(xy_ptext[:, 0], xy_ptext[:, 1], s=10, c='white', marker='+',
               alpha=0.85, linewidths=0.6, zorder=8, rasterized=True)
    ax.scatter(xy_ptab[:, 0], xy_ptab[:, 1], s=6, c='#ffd166', marker='^',
               alpha=0.75, linewidths=0, zorder=7, rasterized=True)

    # подписи
    import matplotlib.patheffects as pe
    for lab in top_labels[:12]:
        m = x_labels == lab
        cx, cy = xx[m].mean(), yy[m].mean()
        txt = ax.text(cx, cy, names2[lab][:34], color='white', fontsize=10,
                      ha='center', weight='bold', zorder=15)
        txt.set_path_effects([pe.withStroke(linewidth=2.5, foreground='#0d1b4c')])

    legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#7c3aed',
               markersize=7, label=f'XML (формализованные) — {len(x_emb):,}'.replace(',', ' ')),
        Line2D([0], [0], marker='+', color='w', markeredgecolor='white',
               markersize=8, label=f'PDF_Text — {len(px_emb):,}'.replace(',', ' ')),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#ffd166',
               markersize=7, label=f'PDF_Tables — {len(pt_emb):,}'.replace(',', ' ')),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='red',
               markersize=8, label='XML ГИС МТ (1 208)'),
    ]
    ax.legend(handles=legend, loc='upper right', fontsize=10, framealpha=0.85,
              facecolor='#0d0d1a', labelcolor='white')

    ax.text(0.5, 1.015, 'Галактики XML и неформализованные документы '
            '(общий t-SNE): где живут PDF относительно XML-кластеров',
            color='#e0e7ff', fontsize=14, ha='center', transform=ax.transAxes)
    fig.tight_layout()
    out = REPORTS_DIR / 'embedding_star_map_combined.png'
    fig.savefig(out, dpi=170, facecolor='black', bbox_inches='tight')
    plt.close(fig)
    print(f"→ {out}")


if __name__ == '__main__':
    main()
