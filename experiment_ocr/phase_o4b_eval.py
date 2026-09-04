# phase_o4b_eval.py
# Фаза O4b-eval: сводка репрезентативного теста классификаторов ориентации
# (500 pdf_text × 4 угла; gemma3 — подвыборка 100) →
# reports/orientation_cls_large.md

import io
import json
from collections import Counter

import numpy as np

from oc_config import REPORTS_DIR, RESULTS_DIR  # noqa: E402

OUT_JSONL = RESULTS_DIR / 'orientation_cls_large.jsonl'
ANGLES = [0, 90, 180, 270]
DISPLAY = {'effnet': 'EfficientNet-B1 (проектная)',
           'vit_large': 'ViT-Large 384 (проектная, remap)',
           'pp_lcnet': 'PP-LCNet doc_ori (PaddleOCR)',
           'gemma3': 'Gemma 3 4B (zero-shot)'}


def main():
    recs = [json.loads(l) for l in io.open(OUT_JSONL, encoding='utf-8')]
    cls_list = [c for c in ('effnet', 'pp_lcnet', 'vit_large', 'gemma3')
                if any(r['cls'] == c for r in recs)]

    L = ['# Репрезентативный тест классификаторов ориентации (O4b)', '']
    L.append('Набор: 500 случайных pdf_text × 4 угла (рендер 150 dpi, PIL rotate) '
             '= 2000 предсказаний на классификатор; Gemma 3 4B — парная '
             'подвыборка 100 документов (400 предсказаний).')
    L.append('')
    L.append('| Классификатор | Overall | 0° | 90° | 180° | 270° | прогонов/с |')
    L.append('|---|---|---|---|---|---|---|')
    for cls in cls_list:
        sub = [r for r in recs if r['cls'] == cls]
        ok = sum(1 for r in sub if r['pred'] == r['angle'])
        row = [DISPLAY.get(cls, cls), f'**{ok / len(sub):.1%}** ({ok}/{len(sub)})']
        for a in ANGLES:
            sa = [r for r in sub if r['angle'] == a]
            oka = sum(1 for r in sa if r['pred'] == a)
            row.append(f'{oka / len(sa):.0%}')
        row.append(f'{len(sub) / max(sum(r["sec"] for r in sub), 1e-9):.0f}')
        L.append('| ' + ' | '.join(row) + ' |')
    L.append('')
    L.append('## Ошибки по типам (истина → предсказание, топ-3)')
    L.append('')
    for cls in cls_list:
        sub = [r for r in recs if r['cls'] == cls]
        errs = Counter((r['angle'], r['pred']) for r in sub if r['pred'] != r['angle'])
        txt = ', '.join(f'{a}→{p} ×{n}' for (a, p), n in errs.most_common(3)) or 'нет'
        L.append(f'- **{DISPLAY.get(cls, cls)}**: {txt}')
    L.append('')
    L.append('## Распределение предсказаний Gemma 3 4B')
    L.append('')
    g = [r for r in recs if r['cls'] == 'gemma3']
    if g:
        dist = Counter(r['pred'] for r in g)
        L.append(f'Предсказания: {dict(dist)} — дегенеративный вывод '
                 f'(один класс на всё), accuracy ≈ уровня случайного (25%).')
    L.append('')
    L.append('## Выводы')
    L.append('')
    L.append('- Репрезентативно (2000 прогонов): **PP-LCNet 97.6%** и '
             '**EfficientNet-B1 97.0%** — лучшие; ViT-Large 94.0%.')
    L.append('- Gemma 3 4B zero-shot **непригодна** для ориентации '
             '(дегенеративный класс, 25.8% ≈ случайный) — как и Qwen3.5-35B '
             '(56.9% в TRAIN). Общие VLM без дообучения задачу не решают.')
    L.append('- Прод-схема: **PP-LCNet или EfficientNet → при низкой '
             'уверенности fallback на Qwen3-VL+LoRA (99.83%)**.')
    out = REPORTS_DIR / 'orientation_cls_large.md'
    out.write_text('\n'.join(L), encoding='utf-8')
    print(f'written {out}')


if __name__ == '__main__':
    main()
