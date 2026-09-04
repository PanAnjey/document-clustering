# -*- coding: utf-8 -*-
# make_qwen_starmap.py -- звёздная карта скрытых состояний Qwen3.5-4B до / после
# LoRA. Две панели, чёрный фон, туманность плотности + точки по типам: семь
# типов, которые дообучение сдвинуло сильнее всего, — цветом, остальные — серым.
#
#   python make_qwen_starmap.py  ->  reports/qwen_starmap_before_after.png

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from scipy.linalg import orthogonal_procrustes
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from common import BASE

R = BASE / 'reports'
SEED = 42
PERP = 30

HIGHLIGHT = {                       # англ. ключ -> (рус. подпись, цвет)
    'Letter':                 ('Письмо', '#4da6ff'),
    'CertificateRegistry':    ('Реестр сертификатов', '#ffd11a'),
    'AcceptanceCertificate':  ('Акт приёма-передачи', '#4dff88'),
    'PriceListAgreement':     ('Прайс-лист', '#ff8c1a'),
    'Contract':               ('Договор', '#c77dff'),
    'SupplementaryAgreement': ('Доп. соглашение', '#00e6d8'),
    'ContractAppendix':       ('Приложение к договору', '#ff4d4d'),
}
GREY = '#5b6472'
NEBULA = LinearSegmentedColormap.from_list(
    'nebula', ['#00000000', '#0a1030', '#152a55', '#2f4d86', '#6f86c0', '#c9d6f0'])


def project(emb):
    x = PCA(n_components=min(50, emb.shape[1]), svd_solver='randomized',
            random_state=SEED).fit_transform(emb.astype(np.float32))
    return TSNE(n_components=2, perplexity=PERP, init='pca', random_state=SEED,
               learning_rate='auto').fit_transform(x)


def align(B, A):
    """Повернуть/отразить проекцию B так, чтобы она легла на A (те же точки,
    тот же порядок). Ортогональный Прокруст: жёсткое вращение + отражение,
    внутренняя структура B не искажается — только ориентация."""
    Ac, Bc = A - A.mean(0), B - B.mean(0)
    R, _ = orthogonal_procrustes(Bc, Ac)
    Ba = Bc @ R
    Ba *= np.sqrt((Ac ** 2).sum() / (Ba ** 2).sum())
    return Ba + A.mean(0)


def draw(ax, xy, types, title, lim):
    x, y = xy[:, 0], xy[:, 1]
    x0, x1, y0, y1 = lim
    ax.set_facecolor('black')
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

    H, _, _ = np.histogram2d(x, y, bins=520, range=[[x0, x1], [y0, y1]])
    H = gaussian_filter(np.log1p(H), sigma=2.4)
    ax.imshow(H.T, origin='lower', extent=[x0, x1, y0, y1], aspect='auto',
              cmap=NEBULA, alpha=0.5, interpolation='bilinear', zorder=1)

    other = ~np.isin(types, list(HIGHLIGHT))
    ax.scatter(x[other], y[other], s=5, c=GREY, alpha=0.5, linewidths=0, zorder=2)
    for key, (_lab, col) in HIGHLIGHT.items():
        m = types == key
        if not m.any():
            continue
        ax.scatter(x[m], y[m], s=5, c=col, alpha=0.28, linewidths=0, zorder=3)
        ax.scatter(x[m], y[m], s=15, c=col, alpha=0.95, linewidths=0, zorder=4)
    ax.set_title(title, color='#e8e6df', fontsize=14, pad=10)


def main():
    meta = json.load(open(R / 'star_meta.json', encoding='utf-8'))
    types = np.array([m['type'] for m in meta])
    base = np.load(R / 'star_repr_base.npy')
    ft = np.load(R / 'star_repr_ft.npy')
    print(f'{len(types)} точек, dim {base.shape[1]}', flush=True)

    xy_b, xy_f = project(base), project(ft)
    xy_f = align(xy_f, xy_b)                       # ориентация «после» под «до»

    allxy = np.vstack([xy_b, xy_f])
    x0, x1 = np.percentile(allxy[:, 0], 0.4), np.percentile(allxy[:, 0], 99.6)
    y0, y1 = np.percentile(allxy[:, 1], 0.4), np.percentile(allxy[:, 1], 99.6)
    mx = (x1 - x0) * 0.05
    lim = (x0 - mx, x1 + mx, y0 - mx, y1 + mx)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 8.2))
    fig.patch.set_facecolor('black')
    draw(axes[0], xy_b, types, 'До дообучения', lim)
    draw(axes[1], xy_f, types, 'После дообучения (LoRA)', lim)

    handles = [Line2D([0], [0], marker='o', linestyle='', markersize=8,
                      markerfacecolor=c, markeredgecolor='none', label=l)
               for l, c in HIGHLIGHT.values()]
    handles.append(Line2D([0], [0], marker='o', linestyle='', markersize=8,
                          markerfacecolor=GREY, markeredgecolor='none',
                          label='остальные типы'))
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False,
               labelcolor='#c9c7bf', fontsize=11, bbox_to_anchor=(0.5, 0.005))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.13, wspace=0.03)
    out = R / 'qwen_starmap_before_after.png'
    fig.savefig(out, dpi=132, facecolor='black')
    print('->', out, flush=True)


if __name__ == '__main__':
    main()
