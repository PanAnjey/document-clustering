# phase_m1_lexical.py
# Фаза M1: лексический тест — переносит ли модель семантику терминов
# предметной области (в т.ч. кросс-языково EN↔RU)?
#
# Три проверки:
#   1) Синонимы EN→RU: cos(en_term, ru_term), top-1 accuracy по кандидатам
#   2) Различители: правильный перевод должен быть ближе неправильного
#      (включая исходную проверку пользователя: ProformaInvoice →
#      'счет на оплату' vs 'счет-фактура')
#   3) Margin: cos(правильный) − max cos(неправильные)
#
# Запуск:
#   python phase_m1_lexical.py                # все модели, cuda:0
#   python phase_m1_lexical.py --model e5_large --device cuda:1

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np

from em_config import MODELS, REPORTS_DIR  # noqa: E402
from em_model_loader import EmbeddingModel  # noqa: E402

# ── Синонимы предметной области (en, ru) ─────────────────────────
SYNONYMS = [
    ('proforma invoice', 'счет на оплату'),
    ('invoice', 'счет-фактура'),
    ('certificate of completed work', 'акт выполненных работ'),
    ('acceptance certificate', 'акт приемки'),
    ('consignment note', 'товарная накладная'),
    ('universal transfer document', 'универсальный передаточный документ'),
    ('lease agreement', 'договор аренды'),
    ('agency commission', 'агентское вознаграждение'),
    ('telecommunication services', 'услуги связи'),
    ('equipment supply', 'поставка оборудования'),
    ('flight tickets', 'авиабилеты'),
    ('business travel expenses', 'командировочные расходы'),
    ('electricity reimbursement', 'возмещение электроэнергии'),
    ('base station modernization', 'модернизация базовой станции'),
]

# ── Различители: (термин, правильный, неправильный) ──────────────
DISCRIMINATORS = [
    # исходная проверка пользователя
    ('proforma invoice', 'счет на оплату', 'счет-фактура'),
    ('invoice', 'счет-фактура', 'счет на оплату'),
    ('reconciliation act', 'акт сверки', 'акт выполненных работ'),
    ('acceptance certificate', 'акт приемки', 'акт сверки'),
    ('storage certificate', 'складской акт', 'акт выполненных работ'),
]

RU_CANDIDATES = [ru for _, ru in SYNONYMS] + ['акт сверки', 'складской акт']


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a @ b.T


def eval_model(key: str, device: str) -> dict:
    with EmbeddingModel(key, device=device) as em:
        all_terms = ([en for en, _ in SYNONYMS] + RU_CANDIDATES
                     + [t for t, _, _ in DISCRIMINATORS])
        all_terms = list(dict.fromkeys(all_terms))
        vecs = em.encode(all_terms)
        sims = cosine_matrix(vecs, vecs)
        idx = {t: i for i, t in enumerate(all_terms)}

    # 1) синонимы: cos + top-1 по кандидатам
    syn_cos, top1_hits = [], 0
    for en, ru in SYNONYMS:
        c = sims[idx[en], idx[ru]]
        syn_cos.append(c)
        cand_sims = [(sims[idx[en], idx[cand]], cand) for cand in RU_CANDIDATES]
        best = max(cand_sims)[1]
        if best == ru:
            top1_hits += 1
    # 2) различители
    disc_ok, disc_rows = 0, []
    for term, good, bad in DISCRIMINATORS:
        cg, cb = sims[idx[term], idx[good]], sims[idx[term], idx[bad]]
        ok = bool(cg > cb)
        disc_ok += ok
        disc_rows.append((term, good, float(cg), bad, float(cb), ok))
    # 3) margin
    margins = []
    for en, ru in SYNONYMS:
        others = [sims[idx[en], idx[c]] for c in RU_CANDIDATES if c != ru]
        margins.append(float(sims[idx[en], idx[ru]] - max(others)))

    return {
        'model': key,
        'dim': MODELS[key]['dim'],
        'syn_cos_mean': float(np.mean(syn_cos)),
        'top1_acc': top1_hits / len(SYNONYMS),
        'disc_ok': f"{disc_ok}/{len(DISCRIMINATORS)}",
        'disc_ok_frac': disc_ok / len(DISCRIMINATORS),
        'margin_mean': float(np.mean(margins)),
        'disc_rows': disc_rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=None)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    keys = [args.model] if args.model else list(MODELS)
    results = []
    for key in keys:
        t0 = time.time()
        print(f"\n=== {key} ({MODELS[key]['hf_name']}) ===")
        res = eval_model(key, args.device)
        res['elapsed_s'] = round(time.time() - t0, 1)
        results.append(res)
        print(f"  syn_cos_mean={res['syn_cos_mean']:.3f}  top1={res['top1_acc']:.2%}  "
              f"disc={res['disc_ok']}  margin={res['margin_mean']:.3f}")
        for term, good, cg, bad, cb, ok in res['disc_rows']:
            print(f"    {term!r}: {good} {cg:.3f} vs {bad} {cb:.3f} "
                  f"{'OK' if ok else 'FAIL'}")

    REPORTS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / 'm1_lexical.json').write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = ['# M1: лексический тест (EN↔RU термины предметной области)', '']
    lines.append('| Модель | dim | syn cos avg | top-1 | различители | margin |')
    lines.append('|---|---|---|---|---|---|')
    for r in results:
        lines.append(f"| {r['model']} | {r['dim']} | {r['syn_cos_mean']:.3f} | "
                     f"{r['top1_acc']:.0%} | {r['disc_ok']} | {r['margin_mean']:.3f} |")
    lines.append('')
    lines.append('## Различители (детально)')
    lines.append('')
    lines.append('| Модель | Термин | Правильный (cos) | Неправильный (cos) | OK |')
    lines.append('|---|---|---|---|---|')
    for r in results:
        for term, good, cg, bad, cb, ok in r['disc_rows']:
            lines.append(f"| {r['model']} | {term} | {good} ({cg:.3f}) | "
                         f"{bad} ({cb:.3f}) | {'✅' if ok else '❌'} |")
    out = REPORTS_DIR / 'm1_lexical.md'
    out.write_text('\n'.join(lines), encoding='utf-8')
    print(f"\nwritten {out}")


if __name__ == '__main__':
    main()
