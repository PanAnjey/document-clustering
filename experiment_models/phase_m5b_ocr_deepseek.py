# phase_m5b_ocr_deepseek.py
# Фаза M5b: устойчивость embedding-моделей к OCR-шуму — парное сравнение
# Tesseract vs DeepSeek-OCR.
#
# Отличие от M5: OCR-тексты НЕ генерируются заново — берутся готовые из
# experiment_ocr/results/o1_results.jsonl (парный дизайн: те же 200
# pdf_text, рендер 300 dpi, оба движка на одних документах).
# Протокол идентичен M5: cos(orig, ocr) + identity top-1/top-5
# (находит ли OCR-эмбеддинг свой документ среди всех оригиналов).
#
# Запуск: python phase_m5b_ocr_deepseek.py [--device cuda:0] [--model KEY]

import argparse
import io
import json
from pathlib import Path

import numpy as np

from em_config import MODELS, REPORTS_DIR  # noqa: E402
from em_model_loader import EmbeddingModel  # noqa: E402

OCR_RESULTS = Path(__file__).resolve().parent.parent / 'experiment_ocr' / 'results' / 'o1_results.jsonl'
MIN_GT_CHARS = 100
MIN_OCR_CHARS = 50


def load_pairs():
    """doc_id → (gt, ocr_text) для каждого OCR-движка (дедуп по doc_id)."""
    docs = {}
    with io.open(OCR_RESULTS, encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r['source'] != 'pdf_text':
                continue
            d = docs.setdefault(r['doc_id'], {'gt': r['gt_text']})
            for eng in ('tesseract', 'deepseek'):
                out = r['outputs'].get(eng)
                if out and not out.get('error'):
                    d[eng] = out['text']
    pairs = {eng: [] for eng in ('tesseract', 'deepseek')}
    for d in docs.values():
        if len(d['gt']) < MIN_GT_CHARS:
            continue
        for eng in pairs:
            if len(d.get(eng, '')) >= MIN_OCR_CHARS:
                pairs[eng].append((d['gt'], d[eng]))
    return pairs


def eval_model(em, pairs):
    orig_emb = em.encode([p[0] for p in pairs])
    ocr_emb = em.encode([p[1] for p in pairs])
    sims = ocr_emb @ orig_emb.T
    diag = np.diag(sims)
    order = np.argsort(-sims, axis=1)
    n = len(pairs)
    top1 = float(np.mean([i in order[i, :1] for i in range(n)]))
    top5 = float(np.mean([i in order[i, :5] for i in range(n)]))
    return {'n': n, 'cos_mean': float(diag.mean()),
            'cos_median': float(np.median(diag)), 'cos_min': float(diag.min()),
            'id_top1': top1, 'id_top5': top5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--model', default=None)
    args = ap.parse_args()

    pairs = load_pairs()
    for eng, p in pairs.items():
        print(f"{eng}: {len(p)} пар")

    keys = [args.model] if args.model else list(MODELS)
    results = {eng: [] for eng in pairs}
    for eng, p in pairs.items():
        print(f"\n===== OCR: {eng} =====")
        for key in keys:
            with EmbeddingModel(key, device=args.device) as em:
                res = eval_model(em, p)
            res['model'] = key
            res['dim'] = MODELS[key]['dim']
            results[eng].append(res)
            print(f"  {key}: cos {res['cos_mean']:.3f}/{res['cos_median']:.3f} "
                  f"min {res['cos_min']:.3f} | id@1 {res['id_top1']:.1%} "
                  f"id@5 {res['id_top5']:.1%}")

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / 'm5b_ocr_deepseek.json').write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')

    def table(rows):
        L = ['| Модель | dim | cos avg | cos median | cos min | identity top-1 | top-5 |',
             '|---|---|---|---|---|---|---|']
        for r in rows:
            L.append(f"| {r['model']} | {r['dim']} | {r['cos_mean']:.3f} | "
                     f"{r['cos_median']:.3f} | {r['cos_min']:.3f} | "
                     f"{r['id_top1']:.1%} | {r['id_top5']:.1%} |")
        return L

    L = ['# M5b: устойчивость эмбеддингов к OCR — Tesseract vs DeepSeek-OCR', '',
         f'Парный дизайн: те же {len(pairs["deepseek"])} документов pdf_text '
         f'(набор experiment_ocr/results/docs_set.json), рендер 300 dpi, '
         f'оба OCR на одних и тех же страницах. Протокол = M5.',
         '']
    for eng, title in (('tesseract', 'Tesseract (rus+eng, psm 3)'),
                       ('deepseek', 'DeepSeek-OCR (markdown вывод)')):
        L.append(f'## {title}')
        L.append('')
        L.extend(table(results[eng]))
        L.append('')
    out = REPORTS_DIR / 'm5b_ocr_deepseek.md'
    out.write_text('\n'.join(L), encoding='utf-8')
    print(f"\nwritten {out}")


if __name__ == '__main__':
    main()
