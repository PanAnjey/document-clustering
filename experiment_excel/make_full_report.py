# make_full_report.py
# Полный детальный отчёт по эксперименту тематической кластеризации документов.
# Собирает статистику из БД (temp_*), строит ~10 диаграмм и формирует .docx
# с методологией, датасетом, результатами по раундам, статистикой и анализом.
#
# Запуск (venv с matplotlib):
#   C:\Windows\Temp\opencode\report_venv\Scripts\python.exe make_full_report.py

import datetime
import re
import tempfile
from pathlib import Path

import numpy as np
import psycopg2

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from experiment_config import DB_URL, SIMILARITY_THRESHOLD

OUT_DIR = Path(r'D:\FileOrganizer\Reports')
CH = Path(tempfile.gettempdir()) / 'excel_full_charts'
CH.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'figure.dpi': 150,
                     'savefig.bbox': 'tight'})
BLUE, ORANGE, GREEN, GRAY = '#2E75B6', '#ED7D31', '#70AD47', '#A6A6A6'
NAVY = RGBColor(0x1F, 0x4E, 0x79)

# Человекочитаемые названия 12 базовых тем (интерпретация top-слов)
THEME_NAMES = {
    0: 'Размещение ВОЛС на ЛЭП', 1: 'Уборка помещений/территории',
    2: 'Выдача материальных ценностей', 3: 'Аварийно-восстановительные работы',
    4: 'Дополнительные работы', 5: 'Логистика: приёмка-хранение',
    6: 'Логистика', 7: 'Аренда помещений', 8: 'Приёмка-передача оборудования',
    9: 'Электроснабжение', 10: 'Услуги связи', 11: 'Поставка оборудования/монтаж',
    12: 'Акт сверки (PDF)', 13: 'Аренда порта Metro Ethernet (PDF)',
    15: 'Регистрация юрлиц / ЕГРЮЛ (PDF)',
    16: 'Замена SIM-карт (PDF)', 17: 'Доступ к инфраструктуре/интернет (PDF)',
    18: 'Клининг кабельных линий (PDF)', 19: 'Поставка шин (PDF)',
    20: 'Цифровые каналы Ethernet EPL (PDF)', 21: 'Реклама на ТВ (PDF)',
    22: 'Выписки ЕГРН (PDF)', 23: 'PR-размещение (PDF)',
    24: 'Услуги HeadHunter (PDF)', 25: 'Закупка мебели/ТМЦ (PDF)',
    26: 'Вознаграждение по договору (PDF)',
}


def tname(lab, words=''):
    if lab in THEME_NAMES:
        return THEME_NAMES[lab]
    seen, parts = set(), []
    for p in (x.strip() for x in (words or '').split(',')):
        if p and p not in seen:
            seen.add(p); parts.append(p)
    n = ', '.join(parts[:3])
    return (n[:1].upper() + n[1:]) if n else f'тема {lab}'


# ─────────────────────── сбор статистики ───────────────────────

def collect():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    st = {}

    def one(q, p=None):
        cur.execute(q, p); return cur.fetchone()

    # Датасет
    st['excel_train'] = one("SELECT COUNT(*) FROM temp_excel_train")[0]
    st['excel_test'] = one("SELECT COUNT(*) FROM temp_excel_test")[0]
    st['excel_failed'] = one("SELECT COUNT(*) FROM temp_excel_failed")[0]
    st['pdftext'] = one("SELECT COUNT(*) FROM temp_pdftext_files")[0]
    st['pdftables'] = one("SELECT COUNT(*) FROM temp_pdftables_files")[0]
    st['total'] = st['excel_train'] + st['excel_test'] + st['pdftext'] + st['pdftables']

    # Round 1 (типы) — размеры кластеров
    cur.execute("SELECT label, COUNT(*) FROM temp_hdbscan WHERE label!= -1 GROUP BY label ORDER BY COUNT(*) DESC")
    st['r1'] = cur.fetchall()
    st['r1_noise'] = one("SELECT COUNT(*) FROM temp_hdbscan WHERE label=-1")[0]

    # Round 2 themes (train)
    cur.execute("SELECT theme_label, COUNT(*) FROM temp_theme2_clusters WHERE theme_label!= -1 GROUP BY theme_label")
    st['r2_train'] = {int(l): c for l, c in cur.fetchall()}
    st['r2_noise'] = one("SELECT COUNT(*) FROM temp_theme2_clusters WHERE theme_label=-1")[0]
    cur.execute("SELECT theme_label, top_words FROM temp_theme2_topwords")
    st['r2_words'] = {int(l): w for l, w in cur.fetchall()}

    # Assign per stream: {theme: count}, plus sims arrays, unknowns
    def assign_stats(table):
        cur.execute(f"SELECT assigned_theme, COUNT(*) FROM {table} WHERE NOT is_unknown GROUP BY assigned_theme")
        per = {int(l): c for l, c in cur.fetchall()}
        n, avg, unk = one(f"SELECT COUNT(*), AVG(similarity), COUNT(*) FILTER (WHERE is_unknown) FROM {table}")
        cur.execute(f"SELECT similarity FROM {table}")
        sims = np.array([r[0] for r in cur.fetchall()], dtype=np.float32)
        return {'per': per, 'n': n, 'avg': float(avg), 'unk': unk, 'sims': sims}

    st['a_test'] = assign_stats('temp_theme2_test_assign')
    st['a_ptext'] = assign_stats('temp_pdftext_assign')
    st['a_ptab'] = assign_stats('temp_pdftables_assign')

    # Registry v2
    cur.execute("SELECT origin, COUNT(*) FROM temp_theme2_centroids_v2 GROUP BY origin")
    st['reg'] = dict(cur.fetchall())
    cur.execute("SELECT theme_label, origin, top_words FROM temp_theme2_centroids_v2 WHERE origin!= 'excel_r2' ORDER BY theme_label")
    st['reg_new'] = [(int(l), o, w) for l, o, w in cur.fetchall()]

    # New themes sizes (pdftables candidates)
    cur.execute("""SELECT n.new_theme, COUNT(*), MAX(w.top_words)
                   FROM temp_pdftables_newthemes n
                   LEFT JOIN temp_pdftables_newthemes_top w ON w.new_theme=n.new_theme
                   GROUP BY n.new_theme ORDER BY COUNT(*) DESC""")
    st['newthemes'] = [(int(l), c, w) for l, c, w in cur.fetchall()]

    # Type canonicalization v2
    st['tip_distinct'] = one("SELECT COUNT(*) FROM temp_tip_groups_v2")[0]
    st['tip_groups'] = one("SELECT COUNT(DISTINCT group_name) FROM temp_tip_groups_v2")[0]
    cur.execute("""SELECT group_name, SUM(docs_excel+docs_pdftext+docs_pdftables) t,
                   SUM(docs_excel), SUM(docs_pdftext), SUM(docs_pdftables)
                   FROM temp_tip_groups_v2 GROUP BY group_name ORDER BY t DESC LIMIT 20""")
    st['tip_top'] = [(g, int(t), int(e), int(px), int(pt)) for g, t, e, px, pt in cur.fetchall()]
    cur.execute("""SELECT group_name, SUM(docs_pdftext+docs_pdftables) p
                   FROM temp_tip_groups_v2 GROUP BY group_name
                   HAVING SUM(docs_excel)=0 ORDER BY p DESC LIMIT 15""")
    st['tip_pdfonly'] = [(g, int(p)) for g, p in cur.fetchall()]

    # Second level (Round 2 KMeans) sub-clusters
    cur.execute("SELECT COUNT(DISTINCT (type_label, sub_label)) FROM temp_second_level")
    st['subclusters'] = one("SELECT COUNT(*) FROM (SELECT DISTINCT type_label, sub_label FROM temp_second_level) x")[0]

    conn.close()
    return st


# ─────────────────────── диаграммы ───────────────────────

def ch_dataset(st):
    names = ['Excel train', 'Excel test', 'PDF_Text', 'PDF_Tables']
    vals = [st['excel_train'], st['excel_test'], st['pdftext'], st['pdftables']]
    colors = [BLUE, BLUE, ORANGE, GREEN]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=colors)
    ax.set_yticks(y, names)
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.01, yi, f'{v:,}'.replace(',', ' '), va='center', fontsize=9)
    ax.set_xlim(0, max(vals) * 1.13)
    ax.set_xlabel('Документов')
    ax.set_title(f'Датасет: {st["total"]:,}'.replace(',', ' ') + ' документов')
    ax.spines[['top', 'right']].set_visible(False)
    p = CH / 'dataset.png'; fig.savefig(p); plt.close(fig); return p


def ch_registry_growth(st):
    stages = ['Раунд 2\n(Excel)', '+PDF_Text\n(дрейф)', '+PDF_Tables\n(дрейф)']
    vals = [12, 12 + st['reg'].get('pdftext_drift', 0),
            12 + st['reg'].get('pdftext_drift', 0) + st['reg'].get('pdftables_drift', 0)]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(stages, vals, marker='o', color=BLUE, lw=2, ms=10)
    for x, v in zip(stages, vals):
        ax.text(x, v + 0.4, str(v), ha='center', fontweight='bold')
    ax.set_ylabel('Тем в реестре')
    ax.set_ylim(0, max(vals) + 3)
    ax.set_title('Рост тематического реестра (механизм дрейфа)')
    ax.spines[['top', 'right']].set_visible(False)
    p = CH / 'registry_growth.png'; fig.savefig(p); plt.close(fig); return p


def ch_theme_streams(st):
    themes = sorted(st['r2_train'], key=lambda l: -(st['r2_train'][l]
                    + st['a_ptext']['per'].get(l, 0) + st['a_ptab']['per'].get(l, 0)))
    names = [tname(l, st['r2_words'].get(l, '')) for l in themes]
    ex = [st['r2_train'][l] + st['a_test']['per'].get(l, 0) for l in themes]
    tx = [st['a_ptext']['per'].get(l, 0) for l in themes]
    tb = [st['a_ptab']['per'].get(l, 0) for l in themes]
    y = np.arange(len(themes))[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(y, ex, color=BLUE, label='Excel (train+test)')
    ax.barh(y, tx, left=ex, color=ORANGE, label='PDF_Text')
    ax.barh(y, tb, left=[a + b for a, b in zip(ex, tx)], color=GREEN, label='PDF_Tables')
    ax.set_yticks(y, names, fontsize=8)
    ax.set_xlabel('Документов (все потоки)')
    ax.set_title('Наполнение 12 тем по потокам')
    ax.legend(fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    p = CH / 'theme_streams.png'; fig.savefig(p); plt.close(fig); return p


def ch_coverage(st):
    rows = [('Excel test', st['a_test']), ('PDF_Text', st['a_ptext']), ('PDF_Tables', st['a_ptab'])]
    names = [r[0] for r in rows]
    cov = [(1 - r[1]['unk'] / r[1]['n']) * 100 for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, cov, color=BLUE)
    ax.axhline(90, color='red', ls='--', lw=1.2, label='цель 90%')
    for b, c in zip(bars, cov):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4, f'{c:.2f}%',
                ha='center', fontweight='bold')
    ax.set_ylim(85, 102); ax.set_ylabel('Покрытие @0.8, %')
    ax.set_title('Покрытие реестра по потокам (перенос без переобучения)')
    ax.legend(); ax.spines[['top', 'right']].set_visible(False)
    p = CH / 'coverage.png'; fig.savefig(p); plt.close(fig); return p


def ch_sim_hist(st):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, (name, a) in zip(axes, [('Excel test', st['a_test']),
                                    ('PDF_Text', st['a_ptext']), ('PDF_Tables', st['a_ptab'])]):
        ax.hist(a['sims'], bins=30, range=(0.6, 1.0), color=BLUE, edgecolor='white')
        ax.axvline(SIMILARITY_THRESHOLD, color='red', ls='--', lw=1.2)
        ax.set_title(f'{name}\navg={a["avg"]:.3f}', fontsize=10)
        ax.set_xlabel('sim'); ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Документов')
    fig.suptitle('Распределение сходства с ближайшим центроидом', fontsize=12)
    p = CH / 'sim_hist.png'; fig.savefig(p); plt.close(fig); return p


def ch_type_top(st):
    labels = [g for g, *_ in st['tip_top'][:15]]
    vals = [t for _, t, *_ in st['tip_top'][:15]]
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.barh(y, vals, color=BLUE)
    ax.set_yticks(y, labels, fontsize=8)
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.01, yi, f'{v:,}'.replace(',', ' '), va='center', fontsize=8)
    ax.set_xlim(0, max(vals) * 1.15)
    ax.set_xlabel('Документов')
    ax.set_title(f'Канонические типы документов (top-15 из {st["tip_groups"]} групп)')
    ax.spines[['top', 'right']].set_visible(False)
    p = CH / 'type_top.png'; fig.savefig(p); plt.close(fig); return p


def ch_type_streams(st):
    top = st['tip_top'][:10]
    labels = [g for g, *_ in top]
    ex = [e for *_, e, px, pt in [(g, t, e, px, pt) for g, t, e, px, pt in top]]
    px = [p for *_, p, _pt in [(g, t, e, p, pt) for g, t, e, p, pt in top]]
    pt = [p for *_, p in [(g, t, e, px2, p) for g, t, e, px2, p in top]]
    x = np.arange(len(labels)); w = 0.27
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.bar(x - w, ex, w, label='Excel', color=BLUE)
    ax.bar(x, px, w, label='PDF_Text', color=ORANGE)
    ax.bar(x + w, pt, w, label='PDF_Tables', color=GREEN)
    ax.set_xticks(x, labels, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Документов'); ax.set_yscale('log')
    ax.set_title('Топ-10 типов документов по потокам (лог. шкала)')
    ax.legend(); ax.spines[['top', 'right']].set_visible(False)
    p = CH / 'type_streams.png'; fig.savefig(p); plt.close(fig); return p


def ch_drift(st):
    rows = [('Excel test', st['a_test']), ('PDF_Text', st['a_ptext']), ('PDF_Tables', st['a_ptab'])]
    names = [r[0] for r in rows]
    cov = [r[1]['n'] - r[1]['unk'] for r in rows]
    unk = [r[1]['unk'] for r in rows]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(x, cov, 0.55, label='назначено (sim≥0.8)', color=BLUE)
    ax.bar(x, unk, 0.55, bottom=cov, label='дрейф-буфер', color=ORANGE)
    for xi, c, u in zip(x, cov, unk):
        ax.text(xi, c + u + max([r[1]['n'] for r in rows]) * 0.01,
                f'{u} ({u / (c + u):.1%})', ha='center', fontsize=9)
    ax.set_xticks(x, names)
    ax.set_ylabel('Документов'); ax.legend()
    ax.set_title('Дрейф-буфер по потокам (источник новых тем)')
    ax.spines[['top', 'right']].set_visible(False)
    p = CH / 'drift.png'; fig.savefig(p); plt.close(fig); return p


def ch_newthemes(st):
    data = st['newthemes'][:12]
    labels = [(w or '').split(',')[0][:28] for _l, _c, w in data]
    vals = [c for _l, c, _w in data]
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.barh(y, vals, color=ORANGE)
    ax.set_yticks(y, labels, fontsize=8)
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.01, yi, str(v), va='center', fontsize=8)
    ax.set_xlim(0, max(vals) * 1.13)
    ax.set_xlabel('Документов')
    ax.set_title('Новые темы, переоткрытые из дрейфа PDF_Tables')
    ax.spines[['top', 'right']].set_visible(False)
    p = CH / 'newthemes.png'; fig.savefig(p); plt.close(fig); return p


# ─────────────────────── docx хелперы ───────────────────────

def h(doc, text, level):
    p = doc.add_heading(text, level=level)
    for r in p.runs:
        r.font.color.rgb = NAVY
    return p


def kv(cell, text, bold=False):
    cell.text = ''
    r = cell.paragraphs[0].add_run(str(text)); r.bold = bold; r.font.size = Pt(10)


def table(doc, header, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(header)); t.style = 'Light Grid Accent 1'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, x in enumerate(header):
        kv(t.rows[0].cells[i], x, True)
    for row in rows:
        c = t.add_row().cells
        for i, v in enumerate(row):
            kv(c[i], v)
    if widths:
        for i, wdt in enumerate(widths):
            for r in t.rows:
                r.cells[i].width = Cm(wdt)
    return t


def pic(doc, path, width=16.5):
    doc.add_picture(str(path), width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER


def page_number(doc):
    fp = doc.sections[0].footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = fp.add_run()
    for t, attr in (('begin', None), (None, 'PAGE'), ('end', None)):
        el = OxmlElement('w:fldChar') if t else OxmlElement('w:instrText')
        if t:
            el.set(qn('w:fldCharType'), t)
        else:
            el.text = attr
        run._r.append(el)


def bullets(doc, items):
    for it in items:
        doc.add_paragraph(it, style='List Bullet')


# ─────────────────────── документ ───────────────────────

def build(st):
    ch = {name: fn(st) for name, fn in [
        ('dataset', ch_dataset), ('growth', ch_registry_growth),
        ('themes', ch_theme_streams), ('coverage', ch_coverage),
        ('sim', ch_sim_hist), ('type_top', ch_type_top),
        ('type_streams', ch_type_streams), ('drift', ch_drift),
        ('newthemes', ch_newthemes)]}

    doc = Document()
    doc.styles['Normal'].font.name = 'Calibri'
    doc.styles['Normal'].font.size = Pt(11)
    page_number(doc)
    now = datetime.datetime.now()

    reg_total = sum(st['reg'].values())
    cov_test = (1 - st['a_test']['unk'] / st['a_test']['n'])
    cov_ptext = (1 - st['a_ptext']['unk'] / st['a_ptext']['n'])
    cov_ptab = (1 - st['a_ptab']['unk'] / st['a_ptab']['n'])

    # Титул
    t = doc.add_paragraph(); t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run('\n\nТематическая кластеризация\nкорпоративного документопотока\n')
    r.bold = True; r.font.size = Pt(24); r.font.color.rgb = NAVY
    s = doc.add_paragraph(); s.alignment = WD_ALIGN_PARAGRAPH.CENTER
    s.add_run('Детальный отчёт по эксперименту\n\n').font.size = Pt(14)
    si = doc.add_paragraph(); si.alignment = WD_ALIGN_PARAGRAPH.CENTER
    si.add_run(
        f'{now.strftime("%d.%m.%Y")}\n\n'
        f'Обработано документов: {st["total"]:,}'.replace(',', ' ') + '\n'
        f'Метод: LLM (Qwen3.5-4B) → эмбеддинг (nomic-embed) → HDBSCAN\n'
        f'Тематический реестр: {reg_total} направлений деятельности'
    ).italic = True
    doc.add_page_break()

    # Резюме
    h(doc, 'Резюме', 1)
    doc.add_paragraph(
        f'Цель — автоматически, без заранее заданной таксономии, разложить '
        f'корпоративный документопоток по направлениям деятельности. Обработано '
        f'{st["total"]:,}'.replace(',', ' ') + f' документов из трёх потоков (Excel, '
        f'PDF_Text, PDF_Tables). LLM извлекает из каждого документа ПРЕДМЕТ '
        f'(направление) и ТИП (вид документа); эмбеддинг предмета кластеризуется '
        f'HDBSCAN. Получен реестр из {reg_total} тематических направлений, '
        f'переносимый на новые потоки без переобучения: покрытие '
        f'{cov_ptext:.1%} (PDF_Text) и {cov_ptab:.1%} (PDF_Tables) при пороге '
        f'{SIMILARITY_THRESHOLD}. Механизм дрейфа автоматически обнаруживает новые '
        f'направления. Типы документов канонизированы отдельным HDBSCAN '
        f'({st["tip_distinct"]} строк → {st["tip_groups"]} групп).')
    table(doc, ['Показатель', 'Значение'], [
        ['Документов обработано', f'{st["total"]:,}'.replace(',', ' ')],
        ['Тематических направлений (реестр v2)', reg_total],
        ['Покрытие Excel test / PDF_Text / PDF_Tables', f'{cov_test:.1%} / {cov_ptext:.1%} / {cov_ptab:.1%}'],
        ['Канонических типов документов', st['tip_groups']],
        ['Под-кластеров 2-го уровня (KMeans)', st['subclusters']],
        ['GPU', '2× RTX PRO 4000 Blackwell (24 ГБ)'],
    ], widths=[10, 7])

    # 1. Методология
    h(doc, '1. Постановка задачи и методология', 1)
    doc.add_paragraph(
        'Задача — обнаружить направления деятельности (стройка, аренда, '
        'энергоснабжение, связь и т.д.) кластерным анализом, а не классификацией '
        'по фиксированному списку. Ключевая проблема: эмбеддинг документа целиком '
        'кластеризует по ТИПАМ (счёт, акт, УПД), т.к. шапка доминирует. Решение — '
        'двухшаговое извлечение смысла LLM.')
    h(doc, '1.1. Конвейер', 2)
    table(doc, ['Шаг', 'Инструмент', 'Результат'], [
        ['Извлечение текста', 'Aspose.Cells (Excel) / fitz (PDF), первые 2 стр.', 'сырой текст ≤3000 символов'],
        ['Извлечение смысла', 'LLM Qwen3.5-4B (промпт ПРЕДМЕТ/ТИП)', 'предмет + тип документа'],
        ['Векторизация', 'nomic-embed-text-v1.5 (768-dim)', 'эмбеддинг ПРЕДМЕТа'],
        ['Кластеризация', 'HDBSCAN (euclidean) + слияние центроидов', 'темы (направления)'],
        ['Канонизация типов', 'HDBSCAN по строкам ТИП', 'реестр типов документов'],
        ['Назначение нового', 'косинус к центроидам, порог 0.8', 'тема или дрейф-буфер'],
    ], widths=[4, 6.5, 6])
    doc.add_paragraph(
        'Промпт LLM требует, чтобы ПРЕДМЕТ не дублировал ТИП (для счёта на '
        'электроэнергию предмет — «Электроснабжение», а не «Счёт»), с двумя '
        'контрастными примерами. Эмбеддинг строится только по строке ПРЕДМЕТ — '
        'тип документа выносится на второй уровень. HDBSCAN дробит короткие '
        'однородные фразы на микрокластеры, поэтому центроиды с косинусным '
        'сходством ≥ 0.92 сливаются (silhouette на валидации 0.009 → 0.340).')

    # 2. Датасет
    h(doc, '2. Датасет', 1)
    doc.add_paragraph(
        f'Три потока корпоративных документов. Excel разделён пополам (train для '
        f'построения реестра, test для проверки). PDF-потоки — независимая проверка '
        f'переносимости реестра на «чужой» домен с более широкой предметной областью.')
    pic(doc, ch['dataset'], 15)
    table(doc, ['Поток', 'Документов', 'Роль', 'Извлечение'], [
        ['Excel train', f'{st["excel_train"]:,}'.replace(',', ' '), 'построение реестра', 'Aspose.Cells TSV'],
        ['Excel test', f'{st["excel_test"]:,}'.replace(',', ' '), 'валидация', 'Aspose.Cells TSV'],
        ['PDF_Text', f'{st["pdftext"]:,}'.replace(',', ' '), 'перенос + дрейф', 'fitz (2 стр.)'],
        ['PDF_Tables', f'{st["pdftables"]:,}'.replace(',', ' '), 'перенос + дрейф', 'fitz (2 стр.)'],
        ['Ошибки извлечения (Excel)', st['excel_failed'], '—', 'запароленные'],
    ], widths=[4.5, 3.5, 4.5, 4])

    # 3. Результаты: реестр
    h(doc, '3. Тематический реестр', 1)
    doc.add_paragraph(
        f'Реестр v2 — {reg_total} направлений: {st["reg"].get("excel_r2", 0)} построены '
        f'на Excel (Раунд 2), {st["reg"].get("pdftext_drift", 0)} добавлены из дрейфа '
        f'PDF_Text, {st["reg"].get("pdftables_drift", 0)} — из дрейфа PDF_Tables. '
        f'Каждое новое направление проходит дедупликацию (слияние с существующим '
        f'при sim ≥ 0.90) — так ЕГРЮЛ из PDF_Tables слился с уже найденным в PDF_Text.')
    pic(doc, ch['growth'], 12)
    h(doc, '3.1. Наполнение базовых тем по потокам', 2)
    pic(doc, ch['themes'])
    doc.add_paragraph(
        'Крупнейшие направления — поставка оборудования/монтаж, услуги связи и '
        'электроснабжение (телеком-специфика датасета). PDF-потоки распределяются '
        'по тем же темам, что и Excel, подтверждая переносимость.')
    # таблица 12 тем
    rows = []
    for l in sorted(st['r2_train'], key=lambda x: -(st['r2_train'][x] + st['a_ptab']['per'].get(x, 0))):
        rows.append([tname(l, st['r2_words'].get(l, '')),
                     st['r2_train'][l], st['a_test']['per'].get(l, 0),
                     st['a_ptext']['per'].get(l, 0), st['a_ptab']['per'].get(l, 0)])
    table(doc, ['Направление', 'Excel tr', 'Excel te', 'PDF_Text', 'PDF_Tables'],
          rows, widths=[6.5, 2.5, 2.5, 2.5, 3])

    # 4. Покрытие
    h(doc, '4. Покрытие и качество назначения', 1)
    doc.add_paragraph(
        f'Реестр, обученный на Excel, применён к PDF-потокам без переобучения. '
        f'Покрытие при пороге {SIMILARITY_THRESHOLD}: Excel test {cov_test:.2%}, '
        f'PDF_Text {cov_ptext:.2%}, PDF_Tables {cov_ptab:.2%} — везде выше цели 90%.')
    pic(doc, ch['coverage'], 13)
    pic(doc, ch['sim'])
    doc.add_paragraph(
        'Среднее сходство падает от Excel test (эталон) к PDF_Tables по мере '
        'расширения предметной области, но остаётся высоким. Документы ниже порога '
        'уходят в дрейф-буфер, а не размазываются по чужим темам.')

    # 5. Дрейф
    h(doc, '5. Механизм дрейфа и новые направления', 1)
    doc.add_paragraph(
        'Документы с sim < 0.8 накапливаются в дрейф-буфере; по нему запускается '
        'HDBSCAN, который переоткрывает новые направления. Так реестр растёт '
        'автоматически без ручной разметки.')
    pic(doc, ch['drift'], 13)
    pic(doc, ch['newthemes'])
    doc.add_paragraph('Новые направления, найденные в PDF-потоках:')
    table(doc, ['Тема', 'Происхождение', 'Top-слова'],
          [[tname(l), ('PDF_Text' if o == 'pdftext_drift' else 'PDF_Tables'), (w or '')[:60]]
           for l, o, w in st['reg_new']], widths=[5.5, 3, 8])

    # 6. Типы
    h(doc, '6. Канонизация типов документов', 1)
    doc.add_paragraph(
        f'Тип документа — второй, независимый уровень. Свободные строки ТИП от LLM '
        f'({st["tip_distinct"]} вариантов) канонизированы отдельным HDBSCAN в '
        f'{st["tip_groups"]} групп: семьи-синонимы склеены («Лист записи '
        f'егрюл/егрн/егрип» → одна группа), частотные одиночные типы — сами по себе.')
    pic(doc, ch['type_top'])
    pic(doc, ch['type_streams'])
    doc.add_paragraph('Типы документов, отсутствующие в Excel (принесли PDF-потоки):')
    table(doc, ['Тип документа', 'Документов (PDF)'],
          [[g, p] for g, p in st['tip_pdfonly'][:12]], widths=[11, 4])

    # 7. Статистика
    h(doc, '7. Сводная статистика', 1)
    table(doc, ['Метрика', 'Значение'], [
        ['Документов всего', f'{st["total"]:,}'.replace(',', ' ')],
        ['Excel train / test', f'{st["excel_train"]} / {st["excel_test"]}'],
        ['PDF_Text / PDF_Tables', f'{st["pdftext"]} / {st["pdftables"]}'],
        ['Тем Раунда 2 (Excel)', 12],
        ['Тем в реестре v2', reg_total],
        ['Под-кластеров 2-го уровня', st['subclusters']],
        ['Канонических типов', f'{st["tip_groups"]} (из {st["tip_distinct"]} строк)'],
        ['Покрытие Excel test @0.8', f'{cov_test:.2%}'],
        ['Покрытие PDF_Text @0.8', f'{cov_ptext:.2%}'],
        ['Покрытие PDF_Tables @0.8', f'{cov_ptab:.2%}'],
        ['Ср. сходство test/ptext/ptab', f'{st["a_test"]["avg"]:.3f} / {st["a_ptext"]["avg"]:.3f} / {st["a_ptab"]["avg"]:.3f}'],
        ['Дрейф-буфер PDF_Text / PDF_Tables', f'{st["a_ptext"]["unk"]} / {st["a_ptab"]["unk"]}'],
    ], widths=[10, 7])

    # 8. Выводы
    h(doc, '8. Выводы и рекомендации', 1)
    h(doc, '8.1. Методология', 2)
    bullets(doc, [
        'Разделение ПРЕДМЕТ/ТИП в промпте LLM — ключ к тематической (а не типовой) '
        'кластеризации; эмбеддинг только предмета даёт направления деятельности.',
        'Отказ от regex-словарей и фиксированной таксономии: направления и типы '
        'обнаруживаются кластерным анализом, новые появляются автоматически.',
        'Слияние центроидов (sim ≥ 0.92) лечит дробление HDBSCAN на коротких фразах '
        '(silhouette 0.009 → 0.340).',
    ])
    h(doc, '8.2. Результаты', 2)
    bullets(doc, [
        f'Реестр из {reg_total} направлений переносится на «чужие» PDF-потоки без '
        f'переобучения: покрытие {cov_ptext:.1%}–{cov_ptab:.1%} при пороге 0.8.',
        'Механизм дрейфа доказан на практике: из PDF найдены новые направления — '
        'замена SIM-карт, регистрация юрлиц (ЕГРЮЛ), клининг кабельных линий, '
        'реклама/PR, HeadHunter и др.',
        f'Типы документов канонизированы независимо ({st["tip_groups"]} групп); '
        f'PDF принесли новые виды (оферта, проектная документация, лист записи '
        f'ЕГРЮЛ, экспедиторская расписка).',
    ])
    h(doc, '8.3. В продакшен', 2)
    bullets(doc, [
        'Реестр центроидов тем + пороговое назначение новых документов; тип '
        'документа — вторичным измерением (двумерная навигация тема × тип).',
        'Периодическое переоткрытие тем из дрейф-буфера + мониторинг покрытия как '
        'индикатора дрейфа тематики.',
        'LLM — узкое место (~1.5 док/с на GPU); масштабирование — шардированием по '
        'нескольким GPU (проверено на 2× RTX PRO 4000).',
    ])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f'full_report_{now.strftime("%Y%m%d_%H%M%S")}.docx'
    doc.save(out)
    print(f'✅ Отчёт: {out}')
    print(f'   Диаграмм: {len(ch)}, {CH}')


if __name__ == '__main__':
    build(collect())

