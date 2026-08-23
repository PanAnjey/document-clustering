# make_report.py
# Сводный отчёт сравнительного эксперимента embedding-моделей:
# reports/models_compare_summary.md = сводная таблица + детали M1/M4/M4b/M5.
#
# Запуск: python make_report.py

import io
import json
import time
from pathlib import Path

from em_config import MODELS, REPORTS_DIR  # noqa: E402


def load(name):
    p = REPORTS_DIR / name
    return json.loads(io.open(p, encoding='utf-8').read()) if p.exists() else []


def main():
    m1 = {r['model']: r for r in load('m1_lexical.json')}
    m4 = {r['model']: r for r in load('m4_eval.json')}
    m4b = {r['model']: r for r in load('m4b_intrinsic.json')}
    m5 = {r['model']: r for r in load('m5_ocr.json')}

    L = []
    L.append('# Сравнительный эксперимент embedding-моделей')
    L.append('')
    L.append(f'Дата: {time.strftime("%Y-%m-%d %H:%M")}. '
             f'Документный корпус: 46,501 XML (25 тем, метки V3 assign). '
             f'OCR-тест: 200 pdf_text (1-я страница, Tesseract rus+eng).')
    L.append('')
    L.append('| Модель | dim | M1 syn cos | M1 top-1 | M1 disc | LOO top-1* | '
             'kNN@10 | ARI | NMI | OCR cos | OCR id@1 | док/с (2×GPU) |')
    L.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for key in MODELS:
        a, b, c, d = m1.get(key, {}), m4.get(key, {}), m4b.get(key, {}), m5.get(key, {})
        def f(x, fmt='{:.3f}'):
            return fmt.format(x) if isinstance(x, float) else ('—' if x is None else str(x))
        L.append(
            f"| {key} | {MODELS[key]['dim']} "
            f"| {f(a.get('syn_cos_mean'))} "
            f"| {f(a.get('top1_acc'), '{:.0%}')} "
            f"| {a.get('disc_ok', '—')} "
            f"| {f(b.get('loo_top1'), '{:.1%}')} "
            f"| {f(c.get('knn_purity_10'))} "
            f"| {f(c.get('kmeans_ari'))} "
            f"| {f(c.get('kmeans_nmi'))} "
            f"| {f(d.get('cos_mean'))} "
            f"| {f(d.get('id_top1'), '{:.1%}')} "
            f"| {f(b.get('docs_per_s'), '{:.0f}')} |"
        )
    L.append('')
    L.append('\\* LOO top-1: метки построены в пространстве nomic_v15 → '
             'у baseline «домашнее» преимущество. Справедливые метрики — '
             'kNN purity и ARI/NMI (центроиды v1.5 не участвуют).')
    L.append('')

    # ── M6: миграция кластеров ───────────────────────────────────
    m6a_path = REPORTS_DIR / 'm6_migration_a.json'
    if m6a_path.exists():
        m6a = json.loads(io.open(m6a_path, encoding='utf-8').read())
        L.append('---')
        L.append('')
        L.append('# M6: миграция кластеров между моделями')
        L.append('')
        L.append('## Track A: центроидное присвоение тем (порог sim ≥ 0.8)')
        L.append('')
        L.append('| Модель | coverage (sim≥0.8) | отличается от разметки V3 |')
        L.append('|---|---|---|')
        for k, v in m6a['per_model'].items():
            L.append(f"| {k} | {v['coverage']:.1%} | {v['diff_vs_v3']:.1%} |")
        L.append('')
        L.append('**Важно:** coverage на пороге 0.8 НЕ сравнимо между моделями — '
                 'распределения косинусных близостей различаются по пространствам '
                 '(qwen3e «плотнее»: max-sim ~0.6-0.8). Порог 0.8 откалиброван '
                 'под nomic_v15; для других моделей нужна своя калибровка порога. '
                 'Высокий coverage nomic_v15 (99.6%) частично тавтологичен: выборка '
                 'отобрана как sim≥0.8 в её же пространстве. diff_vs_v3 у самой '
                 'v15 = 25.9% — центроиды пересчитаны на сбалансированной выборке '
                 '(cap 3000/тему), границы тем сдвинулись.')
        L.append('')
        L.append('## Парное согласие присвоений (на общих уверенно присвоенных)')
        L.append('')
        L.append('**УСТАРЕЛО, см. M6d:** каждая пара считалась на СВОЁМ ядре '
                 '(18K…46K док.) → пары несопоставимы; низкие % переездов у пар '
                 'с qwen3e — артефакт узкого ядра, а не сходства моделей.')
        L.append('')
        L.append('| Пара | n общих | ARI | NMI | % переехавших |')
        L.append('|---|---|---|---|---|')
        for p in m6a['pairs']:
            L.append(f"| {p['pair']} | {p['n_common']} | {p['ari']:.3f} "
                     f"| {p['nmi']:.3f} | {p['moved_frac']:.1%} |")
        L.append('')
        if m6a.get('migration_from_nomic_v15'):
            L.append('## Топ переходов тем: nomic_v15 → модель')
            for tgt, mig in m6a['migration_from_nomic_v15'].items():
                L.append('')
                L.append(f"### → {tgt} (общих {mig['n_common']}, "
                         f"переехало {mig['moved_total']})")
                L.append('')
                L.append('| из темы | в тему | документов |')
                L.append('|---|---|---|')
                for t in mig['top_transitions'][:10]:
                    L.append(f"| {t['from']} | {t['to']} | {t['n']} |")
        L.append('')
    m6d_path = REPORTS_DIR / 'm6d_common_core.json'
    if m6d_path.exists():
        m6d = json.loads(io.open(m6d_path, encoding='utf-8').read())
        L.append('## M6d: контроль — согласие на ОДНИХ И ТЕХ ЖЕ документах')
        L.append('')
        L.append(f"Выборка {m6d['n_sample']}; ядро всех 7 моделей "
                 f"{m6d['n_core_all']} ({m6d['n_core_all']/m6d['n_sample']:.1%}).")
        L.append('')
        L.append('| Пара | moved full | ARI full | moved core | ARI core |')
        L.append('|---|---|---|---|---|')
        for p in m6d['pairs']:
            L.append(f"| {p['pair']} | {p['full']['moved_frac']:.1%} "
                     f"| {p['full']['ari']:.3f} | {p['core_all']['moved_frac']:.1%} "
                     f"| {p['core_all']['ari']:.3f} |")
        L.append('')
        L.append('| Модель | diff vs V3: full | diff vs V3: core |')
        L.append('|---|---|---|')
        for k, v in m6d['diff_vs_v3_masks'].items():
            L.append(f"| {k} | {v['full']:.1%} | {v['core_all']:.1%} |")
        L.append('')
        L.append('**Вывод:** на общем ядре (36.6%) ВСЕ пары согласны (moved '
                 '0.3-5.4%, ARI 0.93-0.99), на полной выборке ЛЮБАЯ смена модели '
                 'переставляет 14.5-33.5% документов (в т.ч. nomic_v2↔qwen3e_06b: '
                 '15.7% вместо 0.5% на узком ядре). Устойчивость разметки '
                 'определяется уверенностью присвоения, а не выбором модели. '
                 'Расхождение с V3 на ядре 8.1-12.5% против 25.9-40.2% на всей '
                 'выборке.')
        L.append('')

    m6c_path = REPORTS_DIR / 'm6c_calibrated.json'
    if m6c_path.exists():
        m6c = json.loads(io.open(m6c_path, encoding='utf-8').read())
        L.append('## M6c: калиброванный unknown (цель: плотные кластеры, минимум unknown)')
        L.append('')
        L.append('| Модель | unknown @точность 90% | unknown @точность 95% | порог @acc90 |')
        L.append('|---|---|---|---|')
        for k, v in m6c.items():
            u90 = f"{v['unknown_at_acc90']:.1%}" if v['unknown_at_acc90'] is not None else '—'
            u95 = f"{v['unknown_at_acc95']:.1%}" if v['unknown_at_acc95'] is not None else '—'
            t90 = f"{v['t_at_acc90']:.3f}" if v['t_at_acc90'] is not None else '—'
            L.append(f"| {k} | {u90} | {u95} | {t90} |")
        L.append('')
        L.append('Порог калибруется под каждую модель: минимальный sim, при котором '
                 'LOO-точность присвоения ≥ 90%. Qwen3E — лучшие при 90% точности '
                 '(unknown ~44%); при 95% лучше sbert_ru/rubert_tiny2. '
                 'unknown — доля ТЕСТОВОЙ ВЫБОРКИ (46 501, сама отобрана как '
                 'уверенно размеченная), не прогноз для всего потока. '
                 'acc_at_t080 (точность на продакшен-пороге 0.8): qwen3e_4b 94.7%, '
                 'qwen3e_06b 93.6%, nomic_v2 83.2%, nomic_v15 74.2%, '
                 'rubert_tiny2 70.3%, sbert_ru 65.2%, e5_large 64.7%.')
        L.append('')
    m7_path = REPORTS_DIR / 'm7_mrl.json'
    if m7_path.exists():
        m7 = json.loads(io.open(m7_path, encoding='utf-8').read())
        L.append('## M7: MRL-усечение размерности (Qwen3-Embedding)')
        L.append('')
        L.append('| Модель | dim | LOO top-1 | kNN@10 | ARI | NMI | unknown@90 | unknown@95 |')
        L.append('|---|---|---|---|---|---|---|---|')
        order_dims = {'qwen3e_4b': ['2560', '1024', '512', '256'],
                      'qwen3e_06b': ['1024', '512', '256'],
                      'nomic_v2': ['512', '256', '128'],
                      'nomic_v15': ['512', '256', '128'],
                      'e5_large': ['512', '256', '128'],
                      'sbert_ru': ['512', '256', '128']}
        native = {'qwen3e_4b': 2560, 'qwen3e_06b': 1024, 'nomic_v2': 768,
                  'nomic_v15': 768, 'e5_large': 1024, 'sbert_ru': 1024}
        for m, dims in order_dims.items():
            if m not in m7:
                continue
            for d in dims:
                if d not in m7[m]:
                    continue
                r = m7[m][d]
                mark = '*' if int(d) == native[m] else ''
                L.append(f"| {m} | {d}{mark} | {r['loo_top1']:.1%} "
                         f"| {r['knn_purity_10']:.3f} | {r['kmeans_ari']:.3f} "
                         f"| {r['kmeans_nmi']:.3f} | {r['unknown_at_acc90']:.1%} "
                         f"| {r['unknown_at_acc95']:.1%} |")
        L.append('')
        L.append('Qwen3-Embedding поддерживает Matryoshka (MRL): вектор можно '
                 'урезать без переобучения. Потеря при усечении 4B 2560→512 — '
                 'всего 1 п.п. LOO (68.6→67.6%), при этом ARI и unknown@90 '
                 'немного УЛУЧШАЮТСЯ (0.311→0.362, 44.2→44.9%) — MRL-префикс '
                 'чуть «чище». У nomic_v2 (не MRL) усечение 768→512 дороже: '
                 'ARI 0.360→0.305, unknown@90 50.8→52.2%. Расчёт: '
                 'phase_m7_mrl.py (протокол = M4/M4b/M6c, embeddings берутся '
                 'готовые, инференс не нужен).')
        L.append('')

    m6b_path = REPORTS_DIR / 'm6_migration_b.json'
    if m6b_path.exists():
        m6b = json.loads(io.open(m6b_path, encoding='utf-8').read())
        L.append('## Track B: HDBSCAN-структура (min_cluster_size=95, '
                 'euclidean на сфере = cosine, eps=√(2·0.3))')
        L.append('')
        L.append('| Модель | кластеров | шум | ARI vs V3 | NMI vs V3 | время, с |')
        L.append('|---|---|---|---|---|---|')
        for k, v in m6b['hdbscan_stats'].items():
            L.append(f"| {k} | {v['clusters']} | {v['noise']:.1%} "
                     f"| {v['ari_vs_v3']:.3f} | {v['nmi_vs_v3']:.3f} | {v['time_s']} |")
        L.append('')
        L.append('| Пара | ARI | NMI | ARI (только кластеризованные в обеих) |')
        L.append('|---|---|---|---|')
        for p in m6b['hdbscan_pairs']:
            ac = p['ari_clustered_only']
            acs = f"{ac:.3f}" if isinstance(ac, float) else '—'
            L.append(f"| {p['pair']} | {p['ari']:.3f} | {p['nmi']:.3f} | {acs} |")
        L.append('')
        L.append('NB: HDBSCAN-labels между прогонами нумеруются произвольно — '
                 'сравнивать можно только ARI/NMI, не «% разошедшихся». '
                 'На этом корпусе (XML, 25 тем) HDBSCAN с продакшен-параметрами '
                 'находит 2-9 макрокластеров, а не темы: тематическая гранулярность '
                 'достигается центроидным присвоением (Track A), а не HDBSCAN.')
        L.append('')

    for name, title in [('m1_lexical.md', None), ('m4_eval.md', None),
                        ('m4b_intrinsic.md', None), ('m5_ocr.md', None)]:
        p = REPORTS_DIR / name
        if p.exists():
            L.append('')
            L.append('---')
            L.append('')
            L.extend(p.read_text(encoding='utf-8').splitlines())

    out = REPORTS_DIR / 'models_compare_summary.md'
    out.write_text('\n'.join(L), encoding='utf-8')
    print(f"written {out}")


if __name__ == '__main__':
    main()
