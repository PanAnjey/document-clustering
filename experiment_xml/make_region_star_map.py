# make_region_star_map.py
# Географическая «карта звёздного неба»: каждый документ — звезда в точке
# региона продавца (seller_region из temp_xml_canonical, адрес → регион
# извлечён канонизатором на этапе X1). Цвет звезды = тематический кластер
# (та же разметка, что в make_embedding_map: T-темы / D-дрейф / T90 ГИС МТ).
#
# Эмбеддинги не нужны: координаты задаются географически (lon·cos55°, lat)
# с гауссовым джиттером вокруг центра региона. Если в адресе продавца регион
# не извлечён — фолбэк на регион по префиксу ИНН продавца, затем покупателя
# (покрывает все 27K документов без региона в адресе).
#
# Выход: experiment_xml/reports/region_star_map.png
# Запуск: python make_region_star_map.py [--n 150000]

import argparse
from pathlib import Path

import numpy as np
import psycopg2

from experiment_xml_config import DB_URL, MARKED_THEME  # noqa: E402
from make_embedding_map import cluster_display_names  # noqa: E402

REPORTS_DIR = Path(__file__).resolve().parent / 'reports'

# ── координаты центров регионов (lon, lat), ключи = значения seller_region ──
REGION_COORDS = {
    'Москва': (37.62, 55.75), 'Санкт-Петербург': (30.31, 59.94),
    'Московская обл': (38.60, 55.30), 'Свердловская обл': (60.60, 56.84),
    'Иркутская обл': (104.28, 52.29), 'Новосибирская обл': (82.93, 55.03),
    'Краснодарский край': (38.98, 45.04), 'Самарская обл': (50.15, 53.20),
    'Кировская обл': (49.66, 58.60), 'Башкортостан': (55.97, 54.74),
    'Ставропольский край': (41.97, 45.04), 'Ростовская обл': (39.71, 47.23),
    'Красноярский край': (92.87, 56.01), 'Пермский край': (56.23, 58.01),
    'Татарстан': (49.11, 55.79), 'Приморский край': (131.89, 43.12),
    'Нижегородская обл': (44.00, 56.33), 'Алтайский край': (83.77, 53.35),
    'Саратовская обл': (46.03, 51.53), 'Челябинская обл': (61.40, 55.16),
    'Кемеровская обл': (86.09, 55.39), 'Волгоградская обл': (44.51, 48.71),
    'Хабаровский край': (135.07, 48.48), 'Тюменская обл': (65.53, 57.15),
    'Забайкальский край': (113.50, 52.03), 'Воронежская обл': (39.18, 51.66),
    'Оренбургская обл': (55.10, 51.77), 'Калининградская обл': (20.51, 54.71),
    'Якутия': (129.73, 62.03), 'Омская обл': (73.37, 54.99),
    'Белгородская обл': (36.58, 50.60), 'Ярославская обл': (39.87, 57.63),
    'Ивановская обл': (40.98, 57.00), 'Тульская обл': (37.42, 54.20),
    'Ульяновская обл': (48.40, 54.31), 'Владимирская обл': (40.40, 56.14),
    'Ленинградская обл': (29.10, 59.20), 'Амурская обл': (127.54, 50.29),
    'Костромская обл': (40.93, 57.77), 'Калужская обл': (36.26, 54.51),
    'Архангельская обл': (40.54, 64.54), 'ХМАО': (69.00, 61.00),
    'Сахалинская обл': (142.73, 46.96), 'Пензенская обл': (45.00, 53.20),
    'Дагестан': (47.50, 42.98), 'Липецкая обл': (39.57, 52.61),
    'Калмыкия': (44.27, 46.31), 'Вологодская обл': (39.89, 59.22),
    'Брянская обл': (34.37, 53.25), 'Кабардино-Балкария': (43.62, 43.48),
    'Мурманская обл': (33.08, 68.97), 'Курганская обл': (65.34, 55.44),
    'Удмуртия': (53.23, 56.85), 'Курская обл': (36.19, 51.73),
    'Чукотский АО': (177.51, 64.73), 'Коми': (50.84, 61.67),
    'Камчатский край': (158.65, 53.04), 'Орловская обл': (36.08, 52.97),
    'Чувашия': (47.25, 56.14), 'Карелия': (34.35, 61.79),
    'Томская обл': (84.97, 56.48), 'Смоленская обл': (32.04, 54.78),
    'Псковская обл': (28.33, 57.82), 'Мордовия': (45.18, 54.18),
    'Тверская обл': (35.90, 56.86), 'ЯНАО': (66.60, 66.53),
    'Рязанская обл': (39.74, 54.63), 'Северная Осетия': (44.68, 43.02),
    'Ингушетия': (44.82, 43.37), 'Магаданская обл': (150.83, 59.57),
    'Бурятия': (107.58, 51.83), 'Астраханская обл': (48.03, 46.35),
    'Тамбовская обл': (41.44, 52.72), 'Хакасия': (91.44, 53.72),
    'Новгородская обл': (31.28, 58.52), 'Чечня': (45.69, 43.32),
    'Марий Эл': (47.89, 56.63), 'Адыгея': (40.10, 44.61),
    'Алтай Респ': (85.96, 51.96), 'Тыва': (94.71, 51.72),
    'Еврейская АО': (132.92, 48.79), 'Карачаево-Черкессия': (42.06, 44.22),
    'Ненецкий АО': (53.05, 67.64), 'Севастополь': (33.52, 44.62),
    'Крым': (34.10, 44.95), 'Херсонская область': (32.62, 46.64),
    'ХМАО/ЯНАО': (68.00, 63.80),
}
ALIASES = {
    'Ямало-Ненецкий АО': 'ЯНАО', 'Чеченская Республика': 'Чечня',
    'Удмуртская Республика': 'Удмуртия', 'Удмуртская': 'Удмуртия',
    'Волгоградская': 'Волгоградская обл', 'Московская': 'Московская обл',
    'Калининградская': 'Калининградская обл', 'Алтай': 'Алтайский край',
    'Ханты-Мансийский АО': 'ХМАО', 'Тюменская обл (юг)': 'Тюменская обл',
}

# Фолбэк, если адрес продавца без региона: регион по префиксу ИНН
# (сначала продавца, затем покупателя) — REGION_BY_CODE канонизатора.
from xml_canonicalizer import REGION_BY_CODE  # noqa: E402


def _resolve_region(reg, seller_inn, buyer_inn):
    if reg:
        return ALIASES.get(reg, reg)
    for inn in (seller_inn, buyer_inn):
        if inn and inn[:2].isdigit():
            name = REGION_BY_CODE.get(inn[:2])
            if name:
                return ALIASES.get(name, name)
    return ''
LON_SCALE = float(np.cos(np.deg2rad(55.0)))  # эквиректангулярная поправка


def load_docs(conn):
    cur = conn.cursor()
    print('загрузка документов + присвоений...')
    cur.execute("""
        SELECT c.id, c.seller_region, c.seller_inn, c.buyer_inn, c.is_marked,
               a.assigned_theme, a.is_unknown, d.new_theme
        FROM temp_xml_canonical c
        LEFT JOIN temp_xml_subject_clean_assign a ON a.xml_id = c.id
        LEFT JOIN temp_xml_drift_assign d ON d.xml_id = c.id""")
    rows = cur.fetchall()
    cur.execute("SELECT new_theme, top_words FROM temp_xml_drift_centroids")
    drift_words = {r[0]: (r[1] or '') for r in cur.fetchall()}
    regions, labels = [], []
    for _, reg, s_inn, b_inn, marked, t, unk, nt in rows:
        regions.append(_resolve_region(reg or '', s_inn or '', b_inn or ''))
        if marked:
            labels.append(f'T{MARKED_THEME}')
        elif nt is not None:
            labels.append(f'D{nt}')
        elif unk or t is None:
            labels.append('noise')
        else:
            labels.append(f'T{t}')
    return np.array(regions), np.array(labels), drift_words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=150000)
    args = ap.parse_args()

    conn = psycopg2.connect(DB_URL)
    regions, labels, drift_words = load_docs(conn)
    conn.close()
    print(f'документов: {len(regions)}')

    known = np.array([r in REGION_COORDS for r in regions])
    print(f'с координатами: {known.sum()}, без региона (отброшены): {(~known).sum()}')
    # документы без региона после фолбэка по ИНН отбрасываем
    regions, labels = regions[known], labels[known]

    rng = np.random.default_rng(42)
    if len(regions) > args.n:
        idx = np.sort(rng.choice(len(regions), size=args.n, replace=False))
        regions, labels = regions[idx], labels[idx]

    names, order = cluster_display_names(labels, drift_words)

    # ── координаты точек: центр региона + гауссов джиттер ──
    xy = np.zeros((len(regions), 2), dtype=np.float64)
    for reg, (lon, lat) in REGION_COORDS.items():
        m = regions == reg
        k = int(m.sum())
        if k:
            xy[m, 0] = lon + rng.normal(0, 0.55, k)
            xy[m, 1] = lat + rng.normal(0, 0.45, k)
    xs, ys = xy[:, 0] * LON_SCALE, xy[:, 1]

    # ── отрисовка ──
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from scipy.ndimage import gaussian_filter
    import matplotlib.patheffects as pe

    x0, x1 = 8.0, 180.0 * LON_SCALE + 3.0
    y0, y1 = 41.5, 71.0
    fig, ax = plt.subplots(figsize=(26, 10.5), dpi=170, facecolor='black')
    ax.set_facecolor('black')
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    # сетка координат (градусная сетка)
    for lon in range(20, 181, 20):
        ax.axvline(lon * LON_SCALE, color='#1e293b', lw=0.4, alpha=0.5, zorder=0)
    for lat in range(40, 72, 10):
        ax.axhline(lat, color='#1e293b', lw=0.4, alpha=0.5, zorder=0)

    # туманность плотности
    nb = 900
    H, _, _ = np.histogram2d(xs, ys, bins=nb, range=[[x0, x1], [y0, y1]])
    H = gaussian_filter(np.log1p(H), sigma=1.6)
    nebula = LinearSegmentedColormap.from_list(
        'nebula', ['#000000', '#050514', '#0d1b4c', '#3730a3', '#7c3aed',
                   '#c026d3', '#f97316', '#fde68a', '#ffffff'])
    ax.imshow(H.T, origin='lower', extent=[x0, x1, y0, y1], aspect='auto',
              cmap=nebula, alpha=0.90, interpolation='bilinear', zorder=1)

    # звёзды: топ-12 тем (без шума) цветные, прочие — холодный фон
    top_labels = [lab for _, lab in order if lab != 'noise'][:12]
    palette = plt.get_cmap('gist_ncar')
    colors = {lab: palette(0.08 + 0.86 * i / max(len(top_labels) - 1, 1))
              for i, lab in enumerate(top_labels)}
    sizes = rng.uniform(0.4, 1.6, size=len(labels)) ** 2
    bg = np.array([l not in top_labels for l in labels])
    ax.scatter(xs[bg], ys[bg], s=0.3, c='#dbeafe', alpha=0.35,
               linewidths=0, zorder=2, rasterized=True)
    for lab in top_labels:
        m = labels == lab
        c = colors[lab]
        if lab == f'T{MARKED_THEME}':
            ax.scatter(xs[m], ys[m], s=28, c='red', marker='X', alpha=0.95,
                       linewidths=0, zorder=12)
            continue
        ax.scatter(xs[m], ys[m], s=sizes[m] * 6, color=c, alpha=0.10,
                   linewidths=0, zorder=3, rasterized=True)
        ax.scatter(xs[m], ys[m], s=sizes[m], color=c, alpha=0.70,
                   linewidths=0, zorder=4, rasterized=True)

    # подписи топ-регионов (количество документов), ручные смещения против
    # наложений в плотной европейской части
    LABEL_OFF = {  # (dx_deg_lon, dy_deg_lat, ha)
        'Москва': (0.0, 2.6, 'center'),
        'Московская обл': (5.2, 1.2, 'left'),
        'Санкт-Петербург': (0.0, 2.2, 'center'),
        'Татарстан': (4.5, 1.6, 'left'),
        'Краснодарский край': (-4.5, -1.6, 'right'),
        'Ставропольский край': (4.5, -1.4, 'left'),
        'Нижегородская обл': (4.6, 1.8, 'left'),
        'Самарская обл': (0.0, -2.2, 'center'),
        'Ростовская обл': (0.0, -2.0, 'center'),
        'Башкортостан': (4.5, -1.2, 'left'),
        'Пермский край': (0.0, 1.8, 'center'),
        'Кировская обл': (0.0, 1.8, 'center'),
    }
    reg_uniq, reg_cnt = np.unique(regions, return_counts=True)
    top_regs = sorted(zip(reg_cnt, reg_uniq), reverse=True)[:16]
    for c, reg in top_regs:
        lon, lat = REGION_COORDS[reg]
        dx, dy, ha = LABEL_OFF.get(reg, (0.0, 1.6, 'center'))
        txt = ax.text((lon + dx) * LON_SCALE, lat + dy,
                      f'{reg} · {c:,}'.replace(',', ' '),
                      color='white', fontsize=9, ha=ha, weight='bold', zorder=15)
        txt.set_path_effects([pe.withStroke(linewidth=2.2, foreground='#0d1b4c')])

    cnt_by_label = {lab: c for c, lab in order}
    legend = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[lab],
               markersize=7,
               label=f"{names[lab][:42]} ({int(cnt_by_label[lab]):,})".replace(',', ' '))
        for lab in top_labels]
    legend.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='#dbeafe',
                         markersize=7, label='прочие темы / шум'))
    ax.legend(handles=legend, loc='lower right', fontsize=9, framealpha=0.85,
              facecolor='#0d0d1a', labelcolor='white')

    ax.text(0.5, 1.02,
            'География контрагентов-продавцов: регионы России как созвездия '
            f'({len(labels):,} документов, цвет = тематический кластер)'.replace(',', ' '),
            color='#e0e7ff', fontsize=15, ha='center', transform=ax.transAxes)
    fig.tight_layout()
    out = REPORTS_DIR / 'region_star_map.png'
    fig.savefig(out, dpi=170, facecolor='black', bbox_inches='tight')
    plt.close(fig)
    print(f'→ {out}')


if __name__ == '__main__':
    main()
