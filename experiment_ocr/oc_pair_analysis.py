# oc_pair_analysis.py
# Парный анализ ошибок двух OCR-движков (по умолчанию deepseek vs paddle)
# на одном и том же наборе pdf_text из results/o1_results.jsonl.
#
# Используется phase_o2_eval.py (раздел в reports/ocr_compare.md)
# и make_word_report.py (раздел 4.7 Word-отчёта).
#
# «Слов-ошибок на документ» = замены + пропуски + вставки слов:
# пропущенные (есть в эталоне, нет в OCR) и выдуманные (есть в OCR,
# нет в эталоне) слова жадно выравниваются по Левенштейну ≤ 2 —
# выровненное считается заменой, невыровненное — пропуском/вставкой.

import io
import json
import re
from collections import Counter

from rapidfuzz.distance import Levenshtein

from oc_config import RESULTS_JSONL  # noqa: E402
from oc_metrics import bow_f1, coverage  # noqa: E402
from oc_textnorm import normalize, words  # noqa: E402

_CYR = re.compile(r'[а-яё]')
_LAT = re.compile(r'[a-z]')

BUCKETS = ('>=0.95', '0.80-0.95', '0.50-0.80', '<0.50')
TOP_PAIRS = 15  # сколько пар замен показывать в отчётах


def load_merged():
    """o1_results.jsonl: каждый doc_id записан несколькими строками,
    движки разнесены по строкам → объединяем outputs по doc_id."""
    recs = [json.loads(l) for l in io.open(RESULTS_JSONL, encoding='utf-8')]
    by_doc = {}
    for r in recs:
        d = by_doc.setdefault(r['doc_id'], {
            'doc_id': r['doc_id'], 'source': r['source'],
            'file_name': r['file_name'], 'gt_text': r['gt_text'], 'outputs': {}})
        d['outputs'].update(r['outputs'])
    return [d for d in by_doc.values() if d['source'] == 'pdf_text']


def _bucket(f1):
    if f1 >= 0.95:
        return BUCKETS[0]
    if f1 >= 0.80:
        return BUCKETS[1]
    if f1 >= 0.50:
        return BUCKETS[2]
    return BUCKETS[3]


def _med(v):
    v = sorted(v)
    return v[len(v) // 2] if v else 0.0


def _p90(v):
    v = sorted(v)
    return v[int(0.9 * len(v))] if v else 0.0


def paired_analysis(a='deepseek', b='paddle'):
    """Парный анализ a vs b на документах, где есть вывод обоих движков."""
    rows = []
    missed = {a: Counter(), b: Counter()}     # слова эталона, не извлечённые OCR
    invented = {a: Counter(), b: Counter()}   # шумовые слова OCR, которых нет в эталоне
    sub_pairs = {a: Counter(), b: Counter()}  # (слово_эталона, слово_ocr) → count
    perdoc_err = {a: [], b: []}               # слов-ошибок на документ
    loops = {a: 0, b: 0}                      # документов с coverage > 1.5
    dig = {a: [0, 0], b: [0, 0]}              # цифровые токены: [сохранено, потеряно]

    for r in load_merged():
        oa, ob = r['outputs'].get(a), r['outputs'].get(b)
        if not oa or not ob or oa.get('error') or ob.get('error'):
            continue
        gt = r['gt_text']
        if not normalize(gt):
            continue
        fa, fb = bow_f1(gt, oa['text']), bow_f1(gt, ob['text'])
        rows.append({'doc_id': r['doc_id'], 'file': r['file_name'],
                     'fa': fa, 'fb': fb})
        gtc = Counter(words(gt))
        for eng, out in ((a, oa), (b, ob)):
            hc = Counter(words(out['text']))
            m = list((gtc - hc).elements())
            iv = list((hc - gtc).elements())
            missed[eng].update(m)
            invented[eng].update(iv)
            used = [False] * len(iv)
            nsub = 0
            for w in m:
                best, bi = 3, -1
                for j, x in enumerate(iv):
                    if used[j] or abs(len(x) - len(w)) > 2:
                        continue
                    dd = Levenshtein.distance(w, x)
                    if dd < best:
                        best, bi = dd, j
                if bi >= 0:
                    used[bi] = True
                    nsub += 1
                    if w != iv[bi]:
                        sub_pairs[eng][(w, iv[bi])] += 1
            perdoc_err[eng].append(nsub + (len(m) - nsub) + (len(iv) - nsub))
            if coverage(gt, out['text']) > 1.5:
                loops[eng] += 1
            gd = Counter(w for w in gtc if any(ch.isdigit() for ch in w))
            if gd:
                inter = sum((gd & hc).values())
                dig[eng][0] += inter
                dig[eng][1] += sum(gd.values()) - inter

    n = len(rows)
    fas = [r['fa'] for r in rows]
    fbs = [r['fb'] for r in rows]
    deltas = [r['fa'] - r['fb'] for r in rows]
    eps = 0.01
    mixed = {e: sum(c for w, c in invented[e].items()
                    if _CYR.search(w) and _LAT.search(w)) for e in (a, b)}
    digit_loss = {e: (100 * dig[e][1] / (dig[e][0] + dig[e][1])
                      if dig[e][0] + dig[e][1] else 0.0) for e in (a, b)}

    return {
        'a': a, 'b': b, 'n': n,
        'f1_mean': {a: sum(fas) / n, b: sum(fbs) / n},
        'f1_med': {a: _med(fas), b: _med(fbs)},
        'delta_mean': sum(deltas) / n, 'delta_med': _med(deltas),
        'wins': sum(1 for d in deltas if d > eps),
        'ties': sum(1 for d in deltas if abs(d) <= eps),
        'losses': sum(1 for d in deltas if d < -eps),
        'buckets': {e: {k: sum(1 for r in rows
                               if _bucket(r['fa' if e == a else 'fb']) == k)
                        for k in BUCKETS} for e in (a, b)},
        'tail20': {a: sum(sorted(fas)[:20]) / 20, b: sum(sorted(fbs)[:20]) / 20},
        'err_doc': {e: {'mean': sum(perdoc_err[e]) / n, 'med': _med(perdoc_err[e]),
                        'p90': _p90(perdoc_err[e]), 'max': max(perdoc_err[e])}
                    for e in (a, b)},
        'missed_total': {e: sum(missed[e].values()) for e in (a, b)},
        'invented_total': {e: sum(invented[e].values()) for e in (a, b)},
        'mixed': mixed,
        'loops': loops,
        'digit_loss': digit_loss,
        'sub_pairs': {e: sub_pairs[e].most_common(TOP_PAIRS) for e in (a, b)},
        'a_fail': [(r['file'], r['fa'], r['fb']) for r in rows
                   if r['fa'] < 0.5 and r['fb'] >= 0.8],
        'b_fail': [(r['file'], r['fa'], r['fb']) for r in rows
                   if r['fb'] < 0.5 and r['fa'] >= 0.8],
        'rows': rows,
    }
