# make_interactive_data.py
# Подготовка данных для интерактивной карты: t-SNE координаты + тексты
# для hover-подсказок → JSON. Дальше HTML строится отдельным скриптом
# в venv с plotly (make_interactive_html.py).
#
# Запуск: python make_interactive_data.py [--xml-n 25000] [--ptab-n 15000]

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import psycopg2

from experiment_xml_config import DB_URL, MARKED_THEME  # noqa: E402
from make_embedding_map import load_data, cluster_display_names  # noqa: E402

REPORTS_DIR = Path(__file__).resolve().parent / 'reports'


def fetch_texts(conn, xml_ids, pt_ids, ptab_ids):
    cur = conn.cursor()
    texts = {}
    if xml_ids:
        fmt = ','.join(['%s'] * len(xml_ids))
        cur.execute(f"SELECT id, type_named_id, subject_text, seller_name, doc_date "
                    f"FROM temp_xml_canonical WHERE id IN ({fmt})", xml_ids)
        for i, t, s, sn, dd in cur.fetchall():
            texts[('xml', int(i))] = {
                'type': t, 'text': (s or '')[:160], 'seller': (sn or '')[:60],
                'date': str(dd or '')}
    for src, ids, table in (('pdftext', pt_ids, 'temp_llm_subject_pdftext'),
                            ('pdftables', ptab_ids, 'temp_llm_subject_pdftables')):
        if ids:
            fmt = ','.join(['%s'] * len(ids))
            cur.execute(f"SELECT train_id, predmet, tip FROM {table} "
                        f"WHERE train_id IN ({fmt})", ids)
            for i, p, tip in cur.fetchall():
                texts[(src, int(i))] = {'type': tip or '', 'text': (p or '')[:160]}
    return texts


def load_l2(conn):
    """Второй уровень: xml_id → (theme, sub_theme) + имена подтем."""
    cur = conn.cursor()
    cur.execute("SELECT xml_id, theme, sub_theme FROM temp_xml_l2_assign")
    l2 = {r[0]: (r[1], r[2]) for r in cur.fetchall()}
    cur.execute("SELECT theme, sub_theme, top_words FROM temp_xml_l2_centroids")
    l2_names = {}
    for t, s, w in cur.fetchall():
        words = (w or '').split(', ')[:3]
        l2_names[(t, s)] = f"{t}.{s} " + ' '.join(words)
    return l2, l2_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xml-n', type=int, default=25000)
    ap.add_argument('--ptab-n', type=int, default=15000)
    args = ap.parse_args()

    conn = psycopg2.connect(DB_URL)
    ids, emb, labels, drift_words = load_data(conn)
    l2, l2_names = load_l2(conn)
    # переопределяем метки XML-документов на подтемы второго уровня
    labels = np.array([f"L{l2[int(i)][0]}.{l2[int(i)][1]}" if int(i) in l2 else lab
                       for i, lab in zip(ids, labels)])

    cur = conn.cursor()
    pdf = {}
    for src, assign_t in (('pdftext', 'temp_pdftext_assign_v2'),
                          ('pdftables', 'temp_pdftables_assign_v2')):
        cur.execute(f"""
            SELECT e.doc_id, e.embedding, a.assigned_theme, a.is_unknown
            FROM temp_pdf_embeddings e
            JOIN {assign_t} a ON a.pdftext_id = e.doc_id
            WHERE e.source = %s ORDER BY e.doc_id""", (src,))
        rows = cur.fetchall()
        pids = np.array([r[0] for r in rows], dtype=np.int64)
        pemb = np.stack([np.frombuffer(r[1], dtype=np.float32) for r in rows])
        plabels = np.array(['noise' if u else f'T{t}' for _, _, t, u in rows])
        pdf[src] = (pids, pemb, plabels)
        print(f"  {src}: {len(pids)}")

    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(len(ids), size=min(args.xml_n, len(ids)), replace=False))
    x_ids, x_emb, x_labels = ids[idx], emb[idx], labels[idx]
    pt_ids, pt_emb, pt_labels = pdf['pdftables']
    if len(pt_ids) > args.ptab_n:
        idx2 = np.sort(rng.choice(len(pt_ids), size=args.ptab_n, replace=False))
        pt_ids, pt_emb, pt_labels = pt_ids[idx2], pt_emb[idx2], pt_labels[idx2]
    px_ids, px_emb, px_labels = pdf['pdftext']

    texts = fetch_texts(conn, [int(i) for i in x_ids], [int(i) for i in px_ids],
                        [int(i) for i in pt_ids])
    conn.close()

    names, order = cluster_display_names(x_labels, drift_words)
    # имена подтем второго уровня (L{t}.{s})
    for (t, s), nm in l2_names.items():
        names[f'L{t}.{s}'] = nm

    # t-SNE
    all_emb = np.vstack([x_emb, px_emb, pt_emb])
    print(f"t-SNE на {len(all_emb)} точках...")
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    t0 = time.time()
    emb50 = PCA(n_components=50, svd_solver='randomized', random_state=42).fit_transform(all_emb)
    coords = TSNE(n_components=2, perplexity=30, init='pca', random_state=42,
                  learning_rate='auto').fit_transform(emb50)
    print(f"  {time.time()-t0:.0f} c")

    n1, n2 = len(x_emb), len(px_emb)
    parts = [
        ('xml', x_ids, x_labels, coords[:n1]),
        ('pdftext', px_ids, px_labels, coords[n1:n1 + n2]),
        ('pdftables', pt_ids, pt_labels, coords[n1 + n2:]),
    ]
    data = []
    for src, pids, plabels, xy in parts:
        for k in range(len(pids)):
            lab = plabels[k]
            info = texts.get((src, int(pids[k])), {})
            data.append({
                'x': round(float(xy[k, 0]), 2), 'y': round(float(xy[k, 1]), 2),
                'src': src, 'cl': lab,
                'nm': names.get(lab, lab),
                'tp': info.get('type', ''),
                'tx': info.get('text', ''),
                'sl': info.get('seller', ''),
                'dt': info.get('date', ''),
            })
    out = REPORTS_DIR / 'map_data.json'
    out.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')),
                   encoding='utf-8')
    print(f"→ {out} ({len(data)} точек, {out.stat().st_size / 1e6:.1f} МБ)")


if __name__ == '__main__':
    main()
