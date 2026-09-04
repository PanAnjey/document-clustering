# phase_o2_eval.py
# Фаза O2: оценка результатов OCR-пилота (CER/WER/покрытие/скорость).
#
# Читает results/o1_results.jsonl:
#   - pdf_text: CER/WER/coverage против эталона fitz, худшие примеры;
#   - pdf_scan: статистика (символы, доля кириллицы) + полные тексты
#     для глазной проверки → reports/scan_samples.md.
#
# Запуск: python phase_o2_eval.py

import io
import json
from collections import defaultdict

import numpy as np

from oc_config import ENGINES, REPORTS_DIR, RESULTS_JSONL  # noqa: E402
from oc_metrics import (  # noqa: E402
    bow_f1, cer, coverage, ngram_cosine, wer,
)
from oc_pair_analysis import paired_analysis  # noqa: E402
from oc_textnorm import cyrillic_ratio, normalize  # noqa: E402


def paired_section(pa):
    """Markdown-раздел парного анализа ошибок двух движков."""
    a, b, n = pa['a'], pa['b'], pa['n']
    L = [f'## Парный анализ ошибок: {a} vs {b}', '']
    L.append(f'Оба движка прогнаны на одних и тех же {n} pdf_text (парный дизайн). '
             '«Слов-ошибок на документ» = замены + пропуски + вставки слов: пропущенные '
             '(есть в эталоне, нет в OCR) и выдуманные (есть в OCR, нет в эталоне) слова '
             'жадно выровнены по Левенштейну ≤ 2 — выровненное считается заменой, '
             'невыровненное — пропуском/вставкой.')
    L.append('')
    ea, eb = pa['err_doc'][a], pa['err_doc'][b]
    ba, bb = pa['buckets'][a], pa['buckets'][b]
    L.append(f'| Метрика | {a} | {b} |')
    L.append('|---|---|---|')
    L.append(f"| BoW-F1 mean / med | {pa['f1_mean'][a]:.3f} / {pa['f1_med'][a]:.3f} | "
             f"{pa['f1_mean'][b]:.3f} / {pa['f1_med'][b]:.3f} |")
    L.append(f"| Слов-ошибок на документ (mean / med / p90) | "
             f"{ea['mean']:.1f} / {ea['med']} / {ea['p90']} | "
             f"{eb['mean']:.1f} / {eb['med']} / {eb['p90']} |")
    L.append(f"| Документов F1 ≥0.95 / 0.80–0.95 / 0.50–0.80 / <0.50 | "
             f"{ba['>=0.95']} / {ba['0.80-0.95']} / {ba['0.50-0.80']} / {ba['<0.50']} | "
             f"{bb['>=0.95']} / {bb['0.80-0.95']} / {bb['0.50-0.80']} / {bb['<0.50']} |")
    L.append(f"| Средний F1 на 20 худших | {pa['tail20'][a]:.2f} | {pa['tail20'][b]:.2f} |")
    L.append(f"| Пропущено слов эталона (Σ) | {pa['missed_total'][a]} | "
             f"{pa['missed_total'][b]} |")
    L.append(f"| Выдумано шумовых слов (Σ) | {pa['invented_total'][a]} | "
             f"{pa['invented_total'][b]} |")
    L.append(f"| …из них со смешанной кириллицей+латиницей | {pa['mixed'][a]} | "
             f"{pa['mixed'][b]} |")
    L.append(f"| Repetition-loop (покрытие > 1.5), документов | {pa['loops'][a]} | "
             f"{pa['loops'][b]} |")
    L.append(f"| Цифровые токены потеряно/искажено | {pa['digit_loss'][a]:.1f}% | "
             f"{pa['digit_loss'][b]:.1f}% |")
    L.append('')
    L.append(f"**Парное сравнение:** {a} лучше на {pa['wins']} документах "
             f"({100*pa['wins']/n:.0f}%), {b} — на {pa['losses']} "
             f"({100*pa['losses']/n:.0f}%), ничья (±0.01) — {pa['ties']} "
             f"({100*pa['ties']/n:.0f}%). Разница F1: mean {pa['delta_mean']:+.3f}, "
             f"med {pa['delta_med']:+.3f}.")
    L.append('')
    for eng, title in ((b, f'Типичные посимвольные замены — {b} '
                          '(кириллица → похожая латиница/цифры)'),
                       (a, f'Типичные замены — {a} '
                          '(в основном нетекстовые зоны: печати/подписи, markdown)')):
        L.append(f'### {title}')
        L.append('')
        L.append('| Эталон → OCR | × |')
        L.append('|---|---|')
        for (w, x), c in pa['sub_pairs'][eng][:12]:
            L.append(f'| {w} → {x} | {c} |')
        L.append('')
    L.append('### Катастрофические сбои')
    L.append('')
    L.append(f"- {b} F1<0.5 при {a} F1≥0.8: {len(pa['b_fail'])} док. — " +
             (', '.join(f'{f[:60]} ({b}={fb:.2f}, {a}={fa:.2f})'
                        for f, fa, fb in pa['b_fail']) or 'нет'))
    L.append(f"- {a} F1<0.5 при {b} F1≥0.8: {len(pa['a_fail'])} док. — " +
             (', '.join(f'{f[:60]} ({a}={fa:.2f}, {b}={fb:.2f})'
                        for f, fa, fb in pa['a_fail']) or 'нет'))
    L.append('')
    L.append(f'**Вывод.** На типичном документе у {b} в ~2.5 раза больше слов-ошибок '
             f'(медиана {eb["med"]} против {ea["med"]}); {a} выигрывает на '
             f'{100*pa["wins"]/n:.0f}% документов. Характер ошибок различается '
             f'принципиально: {b} систематически подменяет кириллические буквы '
             f'похожей латиницей/цифрами (руб.→py6., ООО→о00, №→n, инн→иhh) — текст '
             f'выглядит читабельным, но реквизиты и слова испорчены для поиска и '
             f'эмбеддингов; шума со смешанной кириллицей+латиницей у {b} '
             f'{pa["mixed"][b]} токенов против {pa["mixed"][a]} у {a}. Ошибки {a} '
             f'концентрируются в нетекстовых зонах (печати/подписи → выдуманные '
             f'«фио», «уполномоченное», «реквизиты») и легко фильтруются. '
             f'Контраргумент — устойчивость к повороту 90° ({b} 0.81 vs {a} 0.09, '
             f'см. phase_o3_rotation): при обязательной коррекции ориентации '
             f'разница снимается.')
    L.append('')
    return L


def main():
    recs = [json.loads(l) for l in io.open(RESULTS_JSONL, encoding='utf-8')]
    text_recs = [r for r in recs if r['source'] == 'pdf_text']
    scan_recs = [r for r in recs if r['source'] == 'pdf_scan']
    print(f"записей: pdf_text {len(text_recs)}, pdf_scan {len(scan_recs)}")

    engines = sorted({e for r in recs for e in r['outputs']})
    stats = {}
    per_doc = defaultdict(list)  # engine → [(doc_id, file, bowf1, ncos, cer, wer, cov, sec)]
    for eng in engines:
        for r in text_recs:
            out = r['outputs'].get(eng)
            if not out or out.get('error'):
                continue
            gt, hyp = r['gt_text'], out['text']
            if not normalize(gt):
                continue
            per_doc[eng].append((r['doc_id'], r['file_name'],
                                 bow_f1(gt, hyp), ngram_cosine(gt, hyp),
                                 cer(gt, hyp), wer(gt, hyp), coverage(gt, hyp),
                                 out['sec']))
        arr = np.array([[f, nc, c, w, v, s] for _, _, f, nc, c, w, v, s in per_doc[eng]])
        stats[eng] = {
            'n': len(arr),
            'bowf1_mean': float(arr[:, 0].mean()), 'bowf1_median': float(np.median(arr[:, 0])),
            'ncos_mean': float(arr[:, 1].mean()),
            'cer_mean': float(arr[:, 2].mean()), 'cer_median': float(np.median(arr[:, 2])),
            'cer_p90': float(np.percentile(arr[:, 2], 90)),
            'wer_mean': float(arr[:, 3].mean()),
            'cov_mean': float(arr[:, 4].mean()),
            'sec_mean': float(arr[:, 5].mean()),
        }
        print(f"{eng}: n={stats[eng]['n']} BoW-F1 {stats[eng]['bowf1_mean']:.3f} "
              f"(med {stats[eng]['bowf1_median']:.3f}) 3gram {stats[eng]['ncos_mean']:.3f} "
              f"CER med {stats[eng]['cer_median']:.3f} "
              f"cov {stats[eng]['cov_mean']:.2f} {stats[eng]['sec_mean']:.2f} с/стр")

    REPORTS_DIR.mkdir(exist_ok=True)
    n_text_docs = len({r['doc_id'] for r in text_recs})
    n_scan_docs = len({r['doc_id'] for r in scan_recs})
    L = ['# Пилот OCR-движков: сравнение на корпусе документов', '']
    L.append(f'Выборка: {n_text_docs} pdf_text (эталон = текст 1-й страницы fitz, '
             f'рендер 300 dpi) + {n_scan_docs} pdf_scan (глазная проверка). '
             f'Движки: {", ".join(engines)}.')
    L.append('')
    L.append('| Движок | BoW-F1 avg | BoW-F1 med | 3gram-cos | CER med* | '
             'CER p90* | WER avg* | Покрытие | с/стр |')
    L.append('|---|---|---|---|---|---|---|---|---|')
    for eng in engines:
        s = stats[eng]
        L.append(f"| {eng} | {s['bowf1_mean']:.3f} | {s['bowf1_median']:.3f} | "
                 f"{s['ncos_mean']:.3f} | {s['cer_median']:.3f} | "
                 f"{s['cer_p90']:.3f} | {s['wer_mean']:.3f} | {s['cov_mean']:.2f} | "
                 f"{s['sec_mean']:.2f} |")
    L.append('')
    L.append('**BoW-F1** — F1 по мультимножеству слов (порядок не важен, главная метрика '
             'для корпуса с таблицами); **3gram-cos** — косинус по символьным 3-граммам. '
             '*CER/WER чувствительны к порядку текста: у таблиц fitz и OCR выдают строки '
             'в разном порядке → значения завышены относительно реальной ошибки. '
             'Покрытие = длина OCR / длина эталона (<1 — движок теряет блоки, '
             '>1.5 — эталон неполон, метрики по документу шумные).')
    L.append('')

    # худшие примеры (по BoW-F1 — худшая полнота извлечения)
    for eng in engines:
        worst = sorted(per_doc[eng], key=lambda x: x[2])[:5]
        L.append(f'## Худшие 5 по BoW-F1 — {eng}')
        L.append('')
        L.append('| BoW-F1 | CER | Покрытие | Файл |')
        L.append('|---|---|---|---|')
        for doc_id, fname, f1, nc, c, w, v, _ in worst:
            L.append(f'| {f1:.2f} | {c:.2f} | {v:.2f} | {fname[:80]} |')
        L.append('')

    # парный анализ ошибок deepseek vs paddle (количество и характер ошибок)
    if 'deepseek' in engines and 'paddle' in engines:
        L.extend(paired_section(paired_analysis('deepseek', 'paddle')))

    # сканы: статистика + образцы
    L.append('## pdf_scan: статистика (без эталона)')
    L.append('')
    L.append('| Документ | Движок | Символов | Кириллица | с/стр |')
    L.append('|---|---|---|---|---|')
    sample_lines = ['# Образцы OCR на реальных сканах (pdf_scan)', '']
    for r in scan_recs:
        for eng in engines:
            out = r['outputs'].get(eng)
            if not out:
                continue
            t = out['text']
            L.append(f"| {r['doc_id']} | {eng} | {len(t)} | "
                     f"{cyrillic_ratio(t):.2f} | {out['sec']} |")
        sample_lines.append(f"## {r['file_name']} (id {r['doc_id']})")
        for eng in engines:
            out = r['outputs'].get(eng)
            if out:
                sample_lines.append(f"### {eng}")
                sample_lines.append('```')
                sample_lines.append(out['text'][:800])
                sample_lines.append('```')
                sample_lines.append('')
    (REPORTS_DIR / 'scan_samples.md').write_text(
        '\n'.join(sample_lines), encoding='utf-8')

    out_path = REPORTS_DIR / 'ocr_compare.md'
    out_path.write_text('\n'.join(L), encoding='utf-8')
    (REPORTS_DIR / 'ocr_compare.json').write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"written {out_path}")


if __name__ == '__main__':
    main()
