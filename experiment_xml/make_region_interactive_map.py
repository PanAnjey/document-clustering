#!/usr/bin/env python3
# make_region_interactive_map.py -> standalone plotly Scattergeo HTML
import gc, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / '..'))

import numpy as np
import psycopg2

from experiment_xml_config import DB_URL, MARKED_THEME
from make_region_star_map import REGION_COORDS, _resolve_region
try:
    from xml_canonicalizer import REGION_BY_CODE
except Exception:
    REGION_BY_CODE = {}

REPORTS_DIR = Path(__file__).resolve().parent / 'reports'
SAMPLE_N = 60_000


def load_docs(conn):
    print("- loading docs with regions and themes...")
    cur = conn.cursor()
    q = """
        SELECT id, seller_region, seller_inn, buyer_inn, is_marked,
               type_named_id, doc_date, seller_name,
               assigned_theme, is_unknown, new_theme
        FROM temp_xml_canonical
        LEFT JOIN temp_xml_subject_clean_assign a ON a.xml_id = id
        LEFT JOIN  temp_xml_drift_assign         d ON d.xml_id = id"""
    cur.execute(q)
    rows = []
    for r in cur.fetchall():
        region = _resolve_region(r[1] or '', r[2] or '', r[3] or '')
        if r[4]:  # is_marked
            theme_label = f'T{MARKED_THEME}'
        elif r[10] is not None:  # new_theme from drift
            theme_label = f'D{r[10]}'
        elif r[9] is True:  # is_unknown
            theme_label = 'noise'
        else:
            theme_label = f'T{r[8]}' if r[8] is not None else 'noise'
        rows.append((int(r[0]), region, str(r[6] or ''), theme_label))
    cur.close()
    gc.collect()
    print(f"  loaded {len(rows)} docs")
    return rows


def build_geo_map(docs_list, N=SAMPLE_N):
    # filter to known regions only
    coord_docs = []
    for (doc_id, region_name, doc_date, theme_label) in docs_list:
        if region_name != '' and region_name in REGION_COORDS:
            coord_docs.append((doc_id, region_name, doc_date, theme_label))

    rng = np.random.default_rng(42)
    k = min(N, len(coord_docs))
    indices = rng.choice(len(coord_docs), k, replace=False)
    keep = [coord_docs[int(i)] for i in indices]

    print("building geo scatter...")
    lons = []
    lats = []
    labels = []
    hovers = []

    for (doc_id, region_name, doc_date, theme_label) in keep:
        co = REGION_COORDS[region_name]
        lon_jitter = rng.normal(0, 0.35 * np.cos(np.deg2rad(co[1])))
        lat_jitter = rng.normal(0, 0.29)
        lons.append(co[0] + lon_jitter)
        lats.append(co[1] + lat_jitter)
        labels.append(theme_label)
        hovers.append(f"<b>{region_name}</b><br>date: {doc_date}")

    # map unique theme labels to numeric IDs for coloring
    uniq_labels = sorted(set(labels))
    label_to_id = {lbl: idx for idx, lbl in enumerate(uniq_labels)}
    color_ids = [label_to_id[lbl] for lbl in labels]

    import plotly.graph_objects as go

    # два слоя маркеров: гало + ядра звёзд
    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lon=lons, lat=lats, mode='markers',
        marker=dict(size=12, color=color_ids, colorscale="Viridis", opacity=0.15),
        name='halo'))  # гало
    fig.add_trace(go.Scattergeo(
        lon=lons, lat=lats, mode='markers',
        marker=dict(size=3, color=color_ids, colorscale="Viridis", opacity=0.85),
        hovertext=hovers, hoverinfo='text', name='stars'))  # ядра

    fig.update_layout(
        geo=dict(center={'lat': 56, 'lon': 94}, scope='asia', projection_type='mercator',
                 showland=False, showcoastlines=False, showframe=False,
                 bgcolor='black', lakecolor='black'),
        width=1200, height=750,
        paper_bgcolor='black', plot_bgcolor='black',
        title={'text': f'География контрагентов: звёздное небо регионов ({len(lons):,} документов)', 'xanchor': 'center'})

    return fig


if __name__ == '__main__':
    conn = psycopg2.connect(DB_URL)
    all_docs = load_docs(conn)
    print(f"\nTotal {len(all_docs)} docs pulled")
    conn.close()

    geo_map = build_geo_map(all_docs)
    out_path = REPORTS_DIR / 'region_star_interactive.html'
    out_path.write_text(geo_map.to_html(include_plotlyjs='cdn'), encoding='utf-8')
    print(f"\n> {out_path} ({out_path.stat().st_size/1e6:.0f} MB)")