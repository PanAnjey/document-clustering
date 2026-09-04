# make_word_report.py
# Генерация детального отчёта OCR-пилота в формате MS Word (.docx):
# стили, таблицы с подсветкой лучших значений, графики matplotlib,
# оглавление, нумерация страниц.
#
# Запуск (venv_ocr): python make_word_report.py

import io
import json
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from oc_config import REPORTS_DIR, RESULTS_JSONL  # noqa: E402
from oc_metrics import bow_f1, cer, coverage, ngram_cosine, wer  # noqa: E402
from oc_pair_analysis import paired_analysis  # noqa: E402
from oc_textnorm import cyrillic_ratio, normalize  # noqa: E402

plt.rcParams['font.family'] = 'DejaVu Sans'

CHARTS_DIR = REPORTS_DIR / 'charts'
OUT_DOCX = REPORTS_DIR / 'Отчёт_пилот_OCR_движков_2026-07-24_v2.docx'

DARK_BLUE = RGBColor(0x1F, 0x38, 0x64)
GREEN = RGBColor(0x2E, 0x7D, 0x32)
GRAY = RGBColor(0x60, 0x60, 0x60)

ORDER = ['qwen3vl', 'nanonets', 'deepseek', 'qwen3vl4b', 'finereader',
         'deepseek2', 'gemma4', 'paddle', 'tesseract']
DISPLAY = {
    'qwen3vl': 'Qwen3-VL-8B',
    'nanonets': 'Nanonets-OCR2-3B',
    'deepseek': 'DeepSeek-OCR',
    'deepseek2': 'DeepSeek-OCR-2',
    'qwen3vl4b': 'Qwen3-VL-4B',
    'finereader': 'FineReader 16 (ABBYY)',
    'gemma4': 'Gemma-4-E4B',
    'paddle': 'PaddleOCR',
    'tesseract': 'Tesseract (реф.)',
}
COLORS = {
    'qwen3vl': '#6a1b9a', 'nanonets': '#2e7d32', 'deepseek': '#ef6c00',
    'deepseek2': '#bc5100', 'qwen3vl4b': '#ab47bc', 'finereader': '#c62828',
    'gemma4': '#1a73e8', 'paddle': '#1565c0', 'tesseract': '#8c8c8c',
}
ENGINE_DESCR = {
    'qwen3vl': ('VLM (Qwen3-VL-8B-Instruct, энкодер SigLIP-2)',
                'GPU, bf16, ~17 GB VRAM', 'html/markdown'),
    'nanonets': ('VLM (Qwen2.5-VL-3B)', 'GPU, bf16, ~6.5 GB VRAM',
                 'markdown/html (таблицы сохраняются)'),
    'deepseek': ('VLM (3B MoE, DeepEncoder)', 'GPU, bf16, ~7 GB VRAM',
                 'markdown'),
    'deepseek2': ('VLM (3B MoE, энкодер Visual Causal Flow)',
                  'GPU, bf16, ~7 GB VRAM', 'markdown'),
    'qwen3vl4b': ('VLM (Qwen3-VL-4B-Instruct, repetition_penalty 1.15)',
                  'GPU, bf16, ~9 GB VRAM', 'html/markdown'),
    'finereader': ('классический (ABBYY FineReader 16, CNN-распознавание)',
                   'CPU, ~1.3 с/стр (Hot Folder)', 'plain text'),
    'gemma4': ('VLM (google/gemma-4-E4B-it, ~7.6B dense)',
               'GPU, bf16, ~16 GB VRAM', 'plain text'),
    'paddle': ('классический (PP-OCRv5 det+rec)', 'CPU',
               'plain text'),
    'tesseract': ('классический (Tesseract 5, rus+eng)', 'CPU',
                  'plain text'),
}




# ── Данные ───────────────────────────────────────────────────────

def load_data():
    recs = [json.loads(l) for l in io.open(RESULTS_JSONL, encoding='utf-8')]
    text_recs = [r for r in recs if r['source'] == 'pdf_text']
    scan_recs = [r for r in recs if r['source'] == 'pdf_scan']
    per_doc = defaultdict(dict)  # engine → doc_id → метрики
    for eng in ORDER:
        for r in text_recs:
            out = r['outputs'].get(eng)
            if not out or out.get('error'):
                continue
            gt, hyp = r['gt_text'], out['text']
            if not normalize(gt):
                continue
            per_doc[eng][r['doc_id']] = {
                'file': r['file_name'],
                'f1': bow_f1(gt, hyp), 'ncos': ngram_cosine(gt, hyp),
                'cer': cer(gt, hyp), 'wer': wer(gt, hyp),
                'cov': coverage(gt, hyp), 'sec': out['sec'],
            }
    stats = {}
    for eng in ORDER:
        arr = list(per_doc[eng].values())
        stats[eng] = {
            'n': len(arr),
            'f1_mean': float(np.mean([a['f1'] for a in arr])),
            'f1_median': float(np.median([a['f1'] for a in arr])),
            'ncos': float(np.mean([a['ncos'] for a in arr])),
            'cer_med': float(np.median([a['cer'] for a in arr])),
            'wer_mean': float(np.mean([a['wer'] for a in arr])),
            'cov': float(np.mean([a['cov'] for a in arr])),
            'sec': float(np.mean([a['sec'] for a in arr])),
            'tail5': float(np.mean(sorted([a['f1'] for a in arr])[:20])),  # 20 худших
        }
    return text_recs, scan_recs, per_doc, stats


# ── Графики ──────────────────────────────────────────────────────

def chart_quality(stats):
    labels = [DISPLAY[e] for e in ORDER]
    x = np.arange(len(labels))
    width = 0.27
    fig, ax = plt.subplots(figsize=(12.5, 4.5))
    series = [('BoW-F1 (среднее)', [stats[e]['f1_mean'] for e in ORDER]),
              ('BoW-F1 (медиана)', [stats[e]['f1_median'] for e in ORDER]),
              ('3gram-cos', [stats[e]['ncos'] for e in ORDER])]
    for i, (name, vals) in enumerate(series):
        bars = ax.bar(x + (i - 1) * width, vals, width, label=name,
                      color=[COLORS[e] for e in ORDER], alpha=0.55 + 0.2 * i)
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008,
                    f"{b.get_height():.2f}", ha='center', va='bottom', fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title('Полнота извлечения текста (200 pdf_text, эталон fitz)', fontsize=12)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    out = CHARTS_DIR / 'chart_quality.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def chart_tail(per_doc):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for eng in ORDER:
        f1s = sorted([d['f1'] for d in per_doc[eng].values()], reverse=True)
        ax.plot(range(1, len(f1s) + 1), f1s, label=DISPLAY[eng],
                color=COLORS[eng], linewidth=2)
    ax.set_xlabel('документы, отсортированные по убыванию BoW-F1')
    ax.set_ylabel('BoW-F1')
    ax.set_ylim(0, 1.02)
    ax.set_title('Распределение качества по документам: «хвост» ошибок', fontsize=12)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = CHARTS_DIR / 'chart_tail.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def chart_speed(stats):
    labels = [DISPLAY[e] for e in ORDER]
    secs = [stats[e]['sec'] for e in ORDER]
    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(labels, secs, color=[COLORS[e] for e in ORDER], alpha=0.85)
    ax.set_yscale('log')
    ax.set_ylabel('секунд на страницу (лог. шкала)')
    for b, s, e in zip(bars, secs, ORDER):
        ax.text(b.get_x() + b.get_width() / 2, s * 1.12,
                f"{s:.1f} с/стр\n{ENGINE_DESCR[e][1]}", ha='center', va='bottom',
                fontsize=8)
    ax.set_ylim(0.5, 60)
    ax.set_title('Скорость инференса (1 страница 300 dpi)', fontsize=12)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    out = CHARTS_DIR / 'chart_speed.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def chart_pair_scatter(pa):
    fig, ax = plt.subplots(figsize=(7.5, 6))
    fa = [r['fa'] for r in pa['rows']]
    fb = [r['fb'] for r in pa['rows']]
    ax.scatter(fb, fa, s=14, alpha=0.45, color=COLORS['deepseek'], edgecolors='none')
    ax.plot([0, 1], [0, 1], ls='--', color='#888888', lw=1)
    ax.set_xlabel('BoW-F1 PaddleOCR')
    ax.set_ylabel('BoW-F1 DeepSeek-OCR')
    ax.set_title('Парное сравнение на 200 документах: выше диагонали — DeepSeek лучше',
                 fontsize=11)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = CHARTS_DIR / 'chart_pair_scatter.png'
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# ── Хелперы docx ─────────────────────────────────────────────────

def shade_cell(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:fill'), fill)
    tcPr.append(shd)


def set_cell(cell, text, bold=False, color=None, align='center', size=9):
    cell.text = ''
    p = cell.paragraphs[0]
    p.alignment = {'center': WD_ALIGN_PARAGRAPH.CENTER,
                   'left': WD_ALIGN_PARAGRAPH.LEFT}[align]
    run = p.add_run(text)
    run.font.size = Pt(size)
    run.font.bold = bold
    if color is not None:
        run.font.color.rgb = color


def add_table(doc, headers, rows, best_cols=None):
    best_cols = best_cols or {}
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    try:
        table.style = 'Light Shading Accent 1'
    except KeyError:
        table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(headers):
        set_cell(table.rows[0].cells[j], h, bold=True)
    best = {}
    for j, mode in best_cols.items():
        vals = [r[j] for r in rows]
        best[j] = max(vals) if mode == 'max' else min(vals)
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            is_best = j in best and v == best[j]
            text = v if isinstance(v, str) else str(v)
            set_cell(table.rows[i + 1].cells[j], text, bold=is_best,
                     color=GREEN if is_best else None,
                     align='left' if j == 0 else 'center')
    return table


def add_note(doc, text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.size = Pt(8.5)
    run.font.italic = True
    run.font.color.rgb = GRAY
    return p


def add_body(doc, text, bold_prefix=None):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if bold_prefix:
        run = p.add_run(bold_prefix)
        run.font.bold = True
    p.add_run(text)
    return p


def add_bullet(doc, text, bold_prefix=None):
    p = doc.add_paragraph(style='List Bullet')
    if bold_prefix:
        run = p.add_run(bold_prefix)
        run.font.bold = True
    p.add_run(text)
    return p


def add_toc(doc):
    para = doc.add_paragraph()
    run = para.add_run()
    fld1 = OxmlElement('w:fldChar')
    fld1.set(qn('w:fldCharType'), 'begin')
    instr = OxmlElement('w:instrText')
    instr.set(qn('xml:space'), 'preserve')
    instr.text = 'TOC \\o "1-2" \\h \\z \\u'
    fld2 = OxmlElement('w:fldChar')
    fld2.set(qn('w:fldCharType'), 'separate')
    t = OxmlElement('w:t')
    t.text = 'Оглавление обновляется в Word: выделить всё (Ctrl+A) → F9'
    fld3 = OxmlElement('w:fldChar')
    fld3.set(qn('w:fldCharType'), 'end')
    for el in (fld1, instr, fld2, t, fld3):
        run._r.append(el)


def add_page_number(footer_para):
    run = footer_para.add_run()
    fld1 = OxmlElement('w:fldChar')
    fld1.set(qn('w:fldCharType'), 'begin')
    instr = OxmlElement('w:instrText')
    instr.text = 'PAGE'
    fld2 = OxmlElement('w:fldChar')
    fld2.set(qn('w:fldCharType'), 'end')
    for el in (fld1, instr, fld2):
        run._r.append(el)


def setup_styles(doc):
    normal = doc.styles['Normal']
    normal.font.name = 'Calibri'
    normal.font.size = Pt(10.5)
    for name, size in [('Heading 1', 16), ('Heading 2', 13), ('Heading 3', 11.5)]:
        st = doc.styles[name]
        st.font.name = 'Calibri'
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = DARK_BLUE
    core = doc.core_properties
    core.title = 'Пилот OCR-движков'
    core.author = 'experiment_ocr'
    core.comments = 'Автоматически сгенерировано make_word_report.py'


def title_page(doc):
    for _ in range(6):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run('Сравнительный анализ OCR-движков\nдля распознавания русскоязычных '
                    'деловых документов')
    run.font.size = Pt(24)
    run.font.bold = True
    run.font.color.rgb = DARK_BLUE
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run('Пилот: Tesseract, PaddleOCR, DeepSeek-OCR / OCR-2, '
                    'Nanonets-OCR2-3B, Qwen3-VL-8B / 4B, Gemma-4-E4B\n'
                    'Точка отсчёта качества: ABBYY FineReader 16 '
                    '(классический эталон на кириллице)\n'
                    '(включая анализ «хвоста» ошибок и стоимости инференса)')
    run.font.size = Pt(13)
    run.font.color.rgb = GRAY
    for _ in range(3):
        doc.add_paragraph()
    for line in ['Дата: 24–27 июля 2026 г.',
                 'Выборка: 200 pdf_text (эталон fitz) + 10 pdf_scan, рендер 300 dpi',
                 'Платформа: 2 × NVIDIA RTX PRO 4000 (24 GB), CPU для классических движков',
                 'Проект: FileOrganizer / experiment_ocr']:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(line)
        run.font.size = Pt(11)
    doc.add_page_break()


# ── Документ ─────────────────────────────────────────────────────

def main():
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    text_recs, scan_recs, per_doc, stats = load_data()
    n_text = len({r['doc_id'] for r in text_recs})
    n_scan = len({r['doc_id'] for r in scan_recs})

    print('charts...')
    ch_q = chart_quality(stats)
    ch_t = chart_tail(per_doc)
    ch_s = chart_speed(stats)

    doc = Document()
    setup_styles(doc)
    title_page(doc)

    doc.add_heading('Оглавление', level=1)
    add_toc(doc)
    doc.add_page_break()

    # ── Термины и определения ────────────────────────────────────
    doc.add_heading('Термины и определения', level=1)
    glossary = [
        ('OCR (Optical Character Recognition)',
         'Оптическое распознавание символов: извлечение машиночитаемого текста '
         'из изображения документа.',
         'общеупотребительный термин'),
        ('VLM (Vision-Language Model)',
         'Мультимодальная модель: визуальный энкодер + языковая модель. '
         'Анализирует документ целиком (layout, таблицы, контекст) и выдаёт '
         'структурированный текст (markdown/html).',
         'обзор концепций VLM-OCR см. в статье ai-manual.ru (2026)'),
        ('Визуальный энкодер / ViT',
         'Компонент VLM, преобразующий изображение в последовательность '
         'векторных представлений для языковой модели. Базовая архитектура — '
         'Vision Transformer.',
         'Dosovitskiy et al., «An Image is Worth 16x16 Words», ICLR 2021, '
         'arXiv:2010.11929'),
        ('SigLIP-2',
         'Многоязычный визуальный энкодер (контрастивное обучение с сигмоидным '
         'лоссом); применён в Qwen3-VL. В экспериментах проекта по ориентации '
         'показал лучший результат (100% на классе 180°).',
         'Zhai et al., «Sigmoid Loss for Language Image Pre-Training», ICCV '
         '2023, arXiv:2303.15343; Tschannen et al., «SigLIP 2», 2025'),
        ('DeepEncoder',
         'Специализированный визуальный энкодер DeepSeek-OCR: гибрид '
         'SAM-base + CLIP-large с 16-кратной свёрточной компрессией '
         'визуальных токенов (высокое разрешение при малом числе токенов).',
         'DeepSeek-AI, «DeepSeek-OCR: Contexts Optical Compression», 2025'),
        ('Эталон (ground truth)',
         'Референсный текст, с которым сравнивается результат OCR. В пилоте — '
         'встроенный текстовый слой PDF (извлечён fitz), фактически '
         'безошибочный для цифровых документов.',
         'проектная методика'),
        ('fitz / PyMuPDF',
         'Python-библиотека для работы с PDF: извлечение встроенного текста '
         'и рендер страниц в изображения.',
         'pymupdf.readthedocs.io'),
        ('Парный дизайн',
         'Все движки оцениваются на одном и том же зафиксированном наборе '
         'документов (results/docs_set.json), что исключает влияние выборки '
         'на сравнение.',
         'проектная методика'),
        ('BoW-F1',
         'F1-меру по мультимножеству слов (bag-of-words): пересечение слов '
         'OCR и эталона без учёта их порядка. Главная метрика пилота, т.к. '
         'у таблиц порядок текста у fitz и OCR различается.',
         'Salton & McGill, «Introduction to Modern Information Retrieval», '
         '1983; van Rijsbergen, «Information Retrieval», 1979 (F-мера)'),
        ('3gram-cos',
         'Косинусная близость мультимножеств символьных 3-грамм (подстрок '
         'длиной 3). Чувствительна к посимвольным ошибкам, толерантна '
         'к порядку текста.',
         'Cavnar & Trenkle, «N-Gram-Based Text Categorization», SDAIR-94'),
        ('CER (Character Error Rate)',
         'Доля ошибочных символов: расстояние Левенштейна между OCR и '
         'эталоном, делённое на длину эталона. На табличных документах '
         'завышен из-за различий в порядке текста (см. 3.3).',
         'Levenshtein, «Binary codes capable of correcting deletions, '
         'insertions, and reversals», 1966'),
        ('WER (Word Error Rate)',
         'Доля ошибочных слов: расстояние Левенштейна на уровне слов. '
         'Та же оговорка про порядок, что и у CER.',
         'стандарт метрик распознавания речи (NIST); Levenshtein, 1966'),
        ('Покрытие (coverage)',
         'Отношение длины OCR-текста к длине эталона. <1 — движок теряет '
         'блоки; ≫1 — VLM добавляет разметку или зацикливается.',
         'проектная метрика'),
        ('«Хвост» (tail)',
         'Левая часть распределения качества по документам: подмножество '
         'документов с аномально низким BoW-F1. Операционализируется двумя '
         'способами: графиком распределения (раздел 4.2) и средним BoW-F1 '
         'на 20 худших документах выборки. Критичен для кластеризации: '
         'документ с мусорным текстом «уезжает» в чужую тему.',
         'проектный термин'),
        ('Repetition-loop',
         'Сбой авторегрессионной генерации VLM: бесконечное повторение '
         'фрагмента текста (внешний признак — покрытие ≫1).',
         'проектный термин'),
        ('bf16 (bfloat16)',
         '16-битный формат с плавающей точкой (Brain Floating Point): '
         'диапазон fp32 при половинном объёме памяти. Используется для '
         'инференса VLM на GPU.',
         'Kalamkar et al., «A Study of BFLOAT16 for Deep Learning», 2019, '
         'arXiv:1905.12322'),
        ('LoRA',
         'Метод дообучения больших моделей через низкоранговые адаптеры '
         '(замороженные веса + обучаемые матрицы малого ранга).',
         'Hu et al., «LoRA: Low-Rank Adaptation of Large Language Models», '
         'ICLR 2022, arXiv:2106.09685'),
        ('ToUnicode (CMap)',
         'Таблица PDF, сопоставляющая коды глифов шрифта символам Unicode. '
         'Её отсутствие/битость (типично для CAD-чертежей с ГОСТ/SHX-'
         'шрифтами) делает извлечённый текст мусорным при визуально '
         'нормальном документе.',
         'ISO 32000-1 (PDF Reference), раздел ToUnicode CMaps'),
        ('GPU·ч',
         'Графический процессор-час: стоимость вычислений как произведение '
         'времени на число занятых GPU.',
         'общеупотребительная единица'),
        ('Visual Causal Flow',
         'Визуальный энкодер DeepSeek-OCR-2 (2026): «каузальный поток» '
         'визуального кодирования, имитирующий порядок чтения человека; '
         'динамическое разрешение (0-6)×768² + 1024².',
         'DeepSeek-AI, «DeepSeek-OCR 2: Visual Causal Flow», 2026, '
         'arXiv:2601.20552'),
        ('repetition_penalty',
         'Штраф в greedy/sampling-декодинге, понижающий логиты уже '
         'сгенерированных токенов; подавляет repetition-loop VLM '
         '(у Qwen3-VL-4B значение 1.15 полностью лечит HTML/imgur-'
         'галлюцинацию на бланках).',
         'параметр генерации HF transformers'),
    ]
    add_table(doc, ['Термин', 'Определение', 'Источник'],
              [[t, d, s] for t, d, s in glossary])
    add_note(doc, 'Ссылки arXiv: arxiv.org/abs/<идентификатор>. Первоисточники '
                  'архитектур конкретных движков — Hugging Face model cards: '
                  'deepseek-ai/DeepSeek-OCR, nanonets/Nanonets-OCR2-3B, '
                  'PaddlePaddle/PaddleOCR, tesseract-ocr/tesseract.')
    doc.add_page_break()

    # ── 1. Резюме ────────────────────────────────────────────────
    doc.add_heading('1. Резюме', level=1)
    box = doc.add_table(rows=1, cols=1)
    box.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = box.rows[0].cells[0]
    shade_cell(cell, 'E8F0E3')
    cell.text = ''
    p = cell.paragraphs[0]
    run = p.add_run('Рекомендация: Qwen3-VL-8B — максимум качества; '
                    'DeepSeek-OCR v1 — оптимум скорости в 3B-классе')
    run.font.bold = True
    run.font.size = Pt(12)
    run.font.color.rgb = GREEN
    p2 = cell.add_paragraph()
    p2.add_run(
        'Qwen3-VL-8B — лучшее качество OCR на русскоязычном корпусе: BoW-F1 '
        '0.931 (среднее) / 0.979 (медиана), 75% документов с F1≥0.95, в 3 раза '
        'меньше слов-ошибок на типичном документе против DeepSeek (медиана 2 '
        'против 6), потери цифр 16.5% против 25.4%. Цена: в 3.7 раза медленнее '
        '(17.1 против 4.6 с/стр) и repetition-loop чаще (11 против 2 документов; '
        'ловится фильтром покрытия). Поддерживает схему «одна VLM на всё» '
        '(OCR + саммаризация + ориентация-LoRA). DeepSeek-OCR v1 — лучший '
        'выбор 3B-класса (F1 0.901, 4.6 с/стр): Nanonets-OCR2-3B (равное '
        'качество, в 2.3 раза медленнее), Qwen3-VL-4B (равное качество, в 3 '
        'раза медленнее и шумнее) и DeepSeek-OCR-2 (на RU хуже v1 и медленнее) '
        'его не подвинули.')
    p3 = cell.add_paragraph()
    run = p3.add_run('Альтернативы: Nanonets-OCR2-3B — медиана 0.957, '
                     'устойчив к повороту 90° среди VLM; PaddleOCR — если '
                     'доступен только CPU (F1 0.823, 6.2 с/стр).')
    run.font.size = Pt(9.5)
    p4 = cell.add_paragraph()
    run = p4.add_run('Точка отсчёта качества: ABBYY FineReader 16 — классический '
                     'эталон распознавания на кириллице; использована только '
                     'для калибровки сравнения (см. 4.8.4): DeepSeek-OCR стоит '
                     'ровно на её уровне, Qwen3-VL-8B её превзошёл. Как '
                     'кандидат не рассматривалась.')
    run.font.size = Pt(9.5)
    run.font.italic = True
    doc.add_paragraph()
    add_body(doc,
             f'Пилот проведён для выбора OCR-движка проектируемого пайплайна '
             f'кластеризации деловых документов: значительная часть корпуса — сканы '
             f'(счета, акты, УПД, таблицы). Сравнивались 4 движка на {n_text} '
             f'текстовых PDF (эталон — встроенный текст первой страницы) и '
             f'{n_scan} реальных сканах. Ключевое требование — качество кириллицы '
             f'и полнота извлечения табличного содержимого.')

    # ── 2. Постановка задачи ─────────────────────────────────────
    doc.add_heading('2. Постановка задачи', level=1)
    add_body(doc,
             'Проектируемый пайплайн будет обрабатывать смешанный корпус '
             'русскоязычных деловых документов, где сканы составляют значительную '
             'долю (в текущем корпусе — более 14 тыс. документов категории '
             'pdf_scan, плюс табличные PDF). OCR — первый этап обработки сканов, '
             'его качество напрямую определяет качество последующей векторизации '
             'и кластеризации: ошибки распознавания сдвигают документ в '
             'векторном пространстве.')
    doc.add_heading('2.1. Кандидаты', level=2)
    add_body(doc,
             'В качестве точки отсчёта взят Tesseract (текущий движок проекта, '
             'rus+eng). Классическая альтернатива — PaddleOCR (PP-OCRv5, '
             'кириллическая модель). Отдельный класс — vision-language модели '
             '(VLM-OCR), анализирующие документ целиком и возвращающие '
             'структурированный текст: DeepSeek-OCR и Nanonets-OCR2-3B. '
             'Позже (27.07) добавлены: Qwen3-VL-8B и Qwen3-VL-4B (универсальные '
             'VLM с сильным OCR, кандидаты на схему «одна VLM на всё»), '
             'DeepSeek-OCR-2 (релиз 27.01.2026, новый энкодер Visual Causal '
             'Flow) — кандидат на замену DeepSeek-OCR в 3B-классе. Отдельно '
             'от списка кандидатов: ABBYY FineReader 16 — не кандидат, а '
             'точка отсчёта качества распознавания (классический эталон на '
             'кириллице; использована только для калибровки сравнения, '
             'см. 4.8.4).')
    doc.add_heading('2.2. Требования к OCR', level=2)
    for t in ['полнота извлечения кириллического текста (минимум замен символов);',
              'сохранение содержимого таблиц (доминирующий тип документов корпуса);',
              'устойчивость на сложных/стилизованных документах (печати, гарнитуры, '
              'подписи) — «хвост» ошибок важнее среднего;',
              'приемлемая скорость инференса на имеющемся оборудовании (2 × 24 GB GPU);',
              'структурированный вывод — плюс для downstream-задач.']:
        add_bullet(doc, t)

    # ── 3. Методология ───────────────────────────────────────────
    doc.add_heading('3. Методология', level=1)
    doc.add_heading('3.1. Движки', level=2)
    rows = [[DISPLAY[e], ENGINE_DESCR[e][0], ENGINE_DESCR[e][1], ENGINE_DESCR[e][2]]
            for e in ORDER]
    add_table(doc, ['Движок', 'Тип', 'Инференс', 'Вывод'], rows)
    add_note(doc, 'Nanonets-OCR2-3B — файнтюн Qwen2.5-VL-3B на документах; '
                  'DeepSeek-OCR — 3B MoE VLM (DeepEncoder + MoE-декодер), режим '
                  '«Free OCR», gundam base 1024 / crop 640; DeepSeek-OCR-2 — '
                  'тот же 3B MoE-класс, новый энкодер Visual Causal Flow '
                  '(base 1024 / tiles 768 — разрешение зашито, иные base_size '
                  'ломают кастомный код); Qwen3-VL-8B/4B — универсальные VLM '
                  'с промптом v2 «Transcribe the entire document…» (короткий '
                  'промпт в стиле nanonets → извлекается только таблица, '
                  'F1 0.36), для 4B обязателен repetition_penalty 1.15 '
                  '(иначе HTML/imgur-галлюцинация на бланках); PaddleOCR — '
                  'PP-OCRv5_server_det + eslav_PP-OCRv5_mobile_rec (кириллица); '
                  'Tesseract — rus+eng, psm 3.')
    doc.add_heading('3.2. Протокол', level=2)
    add_body(doc,
             f'Выборка: {n_text} документов категории pdf_text (случайные, из корпуса) '
             f'+ {n_scan} реальных сканов pdf_scan. Первая страница каждого документа '
             f'рендерится в PNG (300 dpi) и прогоняется через все движки. '
             f'Эталоном для pdf_text служит встроенный текст той же страницы '
             f'(извлечён fitz) — он фактически безошибочен для цифровых PDF. '
             f'Реальные сканы эталона не имеют и оцениваются визуально '
             f'(образцы — reports/scan_samples.md). Markdown/html-разметка VLM '
             f'перед подсчётом метрик срезается нормализатором (единые правила '
             f'для всех движков).')
    doc.add_heading('3.3. Метрики', level=2)
    add_bullet(doc,
               'F1 по мультимножеству слов (пересечение слов OCR и эталона). '
               'Не зависит от порядка текста — главная метрика для корпуса '
               'с таблицами.', bold_prefix='BoW-F1 — ')
    add_bullet(doc,
               'косинусная близость по мультимножествам символьных 3-грамм; '
               'чувствительна к посимвольным ошибкам, толерантна к порядку.',
               bold_prefix='3gram-cos — ')
    add_bullet(doc,
               'доля ошибочных символов/слов (расстояние Левенштейна). '
               'ВАЖНО: на табличных документах завышены — fitz и OCR выдают '
               'строки в разном порядке, поэтому CER ~0.3 соответствует '
               'визуально идеальному OCR. Приведены справочно.',
               bold_prefix='CER/WER — ')
    add_bullet(doc,
               'длина OCR / длина эталона. <1 — движок теряет блоки; '
               '≫1 — VLM добавляет разметку либо зацикливается (повторы).',
               bold_prefix='Покрытие — ')
    add_bullet(doc, 'среднее время обработки одной страницы.',
               bold_prefix='с/стр — ')

    # ── 4. Результаты ────────────────────────────────────────────
    doc.add_heading('4. Результаты', level=1)
    doc.add_heading('4.1. Сводная таблица', level=2)
    rows = []
    for e in ORDER:
        s = stats[e]
        rows.append([DISPLAY[e],
                     f"{s['f1_mean']:.3f}", f"{s['f1_median']:.3f}",
                     f"{s['ncos']:.3f}", f"{s['cer_med']:.3f}",
                     f"{s['wer_mean']:.3f}", f"{s['cov']:.2f}",
                     f"{s['sec']:.2f}"])
    add_table(doc, ['Движок', 'BoW-F1 avg', 'BoW-F1 med', '3gram-cos', 'CER med*',
                    'WER avg*', 'Покрытие', 'с/стр'], rows,
              best_cols={1: 'max', 2: 'max', 3: 'max', 4: 'min', 5: 'min', 7: 'min'})
    add_note(doc, '* CER/WER завышены порядком текста (см. 3.3). '
                  'Зелёным — лучшие значения в столбце.')
    doc.add_picture(str(ch_q), width=Cm(16.5))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading('4.2. Среднее vs медиана: «хвост» ошибок', level=2)
    add_body(doc,
             'Медианы BoW-F1 у классических движков (0.864–0.870) близки к '
             'VLM 3B-класса (0.943–0.957), но средние заметно ниже '
             '(0.758/0.823): у Tesseract и PaddleOCR тяжёлый «хвост» провалов '
             'на сложных документах — систематические кириллические замены '
             '(№→ne, ООО→000, д→n, «марта»→«mapina»), мусор вместо гарнитурных '
             'подписей и печатей. Qwen3-VL-8B оторвался от всех и по медиане '
             '(0.979), и по среднему (0.931). Средний BoW-F1 на 20 худших '
             'документах: ' +
             ', '.join(f"{DISPLAY[e]} — {stats[e]['tail5']:.2f}" for e in ORDER) +
             '. Лучший хвост у DeepSeek-OCR v1 и Qwen3-VL-8B (0.55/0.54); '
             'у Qwen3-VL-4B и DeepSeek-OCR-2 хвосты тяжелее (0.49/0.50). '
             'Для кластеризации критичен именно хвост: документ с мусорным '
             'текстом «уезжает» в чужую тему (терминология — см. раздел '
             '«Термины и определения»).')
    doc.add_picture(str(ch_t), width=Cm(16.5))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading('4.3. Скорость инференса', level=2)
    add_body(doc,
             'Среднее время: Tesseract — 1.1 с/стр (CPU), DeepSeek-OCR — '
             '8.2 с/стр (GPU, медиана 4.6), DeepSeek-OCR-2 — 11.8 с/стр '
             '(медиана 6.3), PaddleOCR — 6.2 с/стр (CPU), Nanonets-OCR2-3B — '
             '18.6 с/стр (GPU), Qwen3-VL-4B — 30.7 с/стр, Qwen3-VL-8B — '
             '39.6 с/стр (медианы 14.6/17.1; средние раздуты документами-'
             'зацикливаниями, генерация до 4096 токенов). Разница скорости '
             'VLM — прежде всего архитектурная: у DeepSeek MoE-декодер с '
             '~570M активных параметров на токен против dense 4B/8B у Qwen3-VL. '
             'Для текущего корпуса pdf_scan (14 тыс. страниц) это означает '
             '≈32 GPU·ч на DeepSeek v1, ≈46 GPU·ч на DeepSeek-OCR-2, '
             '≈72 GPU·ч на Nanonets или ≈67–154 GPU·ч на Qwen3-VL (по '
             'медиане/среднему; на двух GPU — вдвое быстрее; батч-инференс '
             'через vLLM может дать кратное ускорение, не тестировался). '
             'Классические движки масштабируются по процессам CPU.')
    doc.add_picture(str(ch_s), width=Cm(16.5))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading('4.4. Худшие случаи', level=2)
    for eng in ORDER:
        worst = sorted(per_doc[eng].values(), key=lambda d: d['f1'])[:5]
        rows = [[f"{d['f1']:.2f}", f"{d['cer']:.2f}", f"{d['cov']:.2f}", d['file'][:70]]
                for d in worst]
        doc.add_heading(f'4.4.{ORDER.index(eng) + 1}. {DISPLAY[eng]}', level=3)
        add_table(doc, ['BoW-F1', 'CER', 'Покрытие', 'Файл'], rows)
    add_body(doc,
             'Характерные провалы VLM — repetition-loop: на 1–3 документах модель '
             'зацикливается (покрытие 13–23×, F1≈0). В проде лечится тривиальным '
             'фильтром по покрытию. У PaddleOCR один случай пустого вывода '
             '(счёт, который Tesseract прочитал). У VLM-движков документов '
             'с пустым выводом нет.')
    add_body(doc,
             'Отдельный кейс — документ dc06052e («25-10485G9L9L18-ВЭС.pdf», '
             'чертёж БС): фигурирует в худших у Tesseract и PaddleOCR (F1 0.05/0.07), '
             'но это ЛОЖНЫЙ провал. Его встроенный текстовый слой битый '
             '(CAD-чертёж с ГОСТ/SHX-шрифтом без корректной ToUnicode-таблицы): '
             'fitz извлекает мусор вида «Ǌаǰоǫая Ǻтанция 10485 ǉǴтǓǹǛуǬоǰǫоноǫо», '
             'тогда как визуально документ — нормальный русский текст, и движки '
             'прочитали его правильно. На этапе классификации PDF такие документы '
             'проходят фильтр ≥100 букв (символы Ǌǰǫ — формально латинские буквы '
             'Unicode) и попадают в pdf_text, хотя семантического текста из слоя '
             'извлечь нельзя. Доля в выборке — 1 из 200 (0.5%), в категориях '
             'технической документации может быть выше. Вывод для пайплайна: '
             'нужен детектор битой кодировки текстового слоя (доля букв вне '
             'базовой кириллицы/латиницы > ~30%) с перенаправлением таких '
             'документов в OCR-ветку.',
             bold_prefix='Важная оговорка про «худшие случаи». ')

    doc.add_heading('4.5. Реальные сканы (pdf_scan)', level=2)
    scan_stats = defaultdict(list)
    for r in scan_recs:
        for e in ORDER:
            out = r['outputs'].get(e)
            if out:
                scan_stats[e].append((len(out['text']),
                                      cyrillic_ratio(out['text']), out['sec']))
    rows = []
    for e in ORDER:
        arr = scan_stats[e]
        rows.append([DISPLAY[e], str(len(arr)),
                     f"{np.mean([a[0] for a in arr]):.0f}",
                     f"{np.mean([a[1] for a in arr]):.2f}",
                     f"{np.mean([a[2] for a in arr]):.1f}"])
    add_table(doc, ['Движок', 'Документов', 'Символов (avg)', 'Кириллица (avg)',
                    'с/стр'], rows)
    add_note(doc, 'Эталона нет — оценка визуальная (reports/scan_samples.md): '
                  'все движки извлекают читаемый текст, VLM аккуратнее на '
                  'сложной вёрстке; у Tesseract заметны посимвольные ошибки '
                  'в названиях и числах.')

    doc.add_heading('4.6. Визуальные энкодеры VLM и устойчивость к повороту скана', level=2)
    add_body(doc,
         'Два VLM-движка построены на принципиально разных визуальных энкодерах. '
         'Nanonets-OCR2-3B — файнтюн Qwen2.5-VL-3B (подтверждено конфигом: '
         'Qwen2_5_VLForConditionalGeneration, ViT depth 32 / hidden 1280, '
         'оконное внимание, динамическое разрешение). DeepSeek-OCR использует '
         'специализированный DeepEncoder — гибрид SAM-base + CLIP-large с '
         '16× свёрточным компрессором, спроектированный под документы '
         '(высокое разрешение при малом числе визуальных токенов).')
    add_body(doc,
         'Проект имеет собственный опыт сравнения визуальных энкодеров на задаче '
         'определения ориентации документов (TRAIN/SESSION_CONTENT.md): '
         'zero-shot Qwen2.5-VL-7B — 81.9% accuracy; Qwen2.5-VL-7B + LoRA — 97.7%, '
         'но класс 180° — только 81.7% (главная слабость энкодера: различие '
         '0°/180° требует чтения мелкого текста); Qwen3-VL-8B с энкодером '
         'SigLIP-2 + LoRA — 99.8%, класс 180° — 100%. Это подтверждает: '
         'энкодер Qwen2.5-VL имеет архитектурную слабость на повёрнутых '
         'документах.')
    doc.add_heading('4.6.1. Эмпирический тест: поворот 90°/180° (20 документов)', level=3)
    rows = [
        ['Nanonets-OCR2-3B', '0.958', '0.781', '0.129', '18/20', '0/20'],
        ['DeepSeek-OCR', '0.903', '0.094', '0.026', '0/20', '0/20'],
        ['PaddleOCR', '0.853', '0.805', '0.024', '19/20', '0/20'],
        ['Tesseract (реф.)', '0.782', '0.783', '0.028', '17/20', '0/20'],
    ]
    add_table(doc, ['Движок', 'F1 при 0°', 'F1 при 90°', 'F1 при 180°',
                    '90°: F1>0.5', '180°: F1>0.5'], rows,
              best_cols={1: 'max', 2: 'max', 3: 'max'})
    add_note(doc, 'Те же 20 документов из парного набора, повёрнутые на 90° и 180° '
                  '(PIL), BoW-F1 против эталона fitz. Файл: results/rotation_test.jsonl.')
    add_body(doc,
         'Три вывода. Первый: при 180° не справляется НИ ОДИН движок '
         '(F1 ≤ 0.13) — коррекция ориентации перед OCR обязательна независимо '
         'от выбранной модели. Второй: при 90° DeepSeek-OCR деградирует '
         'полностью (0.094, 0/20) — его crop-пайплайн не переносит поворот, '
         'тогда как Nanonets, PaddleOCR и Tesseract держатся (0.78–0.81). '
         'Третий: опасение про энкодер Qwen2.5-VL частично подтверждается '
         'опытом ориентации (класс 180°), однако в OCR-задаче Nanonets '
         'оказался наиболее устойчивым VLM к 90° — различие энкодеров '
         'проявляется слабее, чем само отсутствие коррекции ориентации.')
    add_body(doc,
         'Архитектурное следствие: в проектируемом пайплайне обязателен '
         'отдельный шаг определения ориентации ДО OCR. Готовые решения уже '
         'есть в проекте: обученная Qwen3-VL+LoRA (99.8%) либо лёгкие '
         'ViT-Large / SigLIP2 / EfficientNet (миллисекунды на документ на GPU). '
         'Схема: orientation-модель → поворот в 0° → OCR.',
         bold_prefix='Вывод. ')
    add_note(doc, 'Новые движки (Qwen3-VL-8B/4B, DeepSeek-OCR-2) в тесте '
                  'поворота не участвовали — отдельный следующий шаг. Косвенно '
                  'для Qwen3-VL ожидается устойчивость не хуже Nanonets '
                  '(энкодер SigLIP-2: 100% на классе 180° в экспериментах '
                  'по ориентации).')

    # ── 4.7. Парный анализ ошибок: DeepSeek-OCR vs PaddleOCR ─────
    pa = paired_analysis('deepseek', 'paddle')
    doc.add_heading('4.7. Парный анализ ошибок: DeepSeek-OCR vs PaddleOCR', level=2)
    add_body(doc,
             f'Оба движка прогнаны на одних и тех же {pa["n"]} pdf_text (парный дизайн). '
             '«Слов-ошибок на документ» = замены + пропуски + вставки слов: '
             'пропущенные (есть в эталоне, нет в OCR) и выдуманные (есть в OCR, нет в эталоне) '
             'слова жадно выравниваются по Левенштейну ≤ 2 — выровненное считается заменой, '
             'невыровненное — пропуском/вставкой.')
    a, b = pa['a'], pa['b']
    ea, eb = pa['err_doc'][a], pa['err_doc'][b]
    ba, bb = pa['buckets'][a], pa['buckets'][b]
    rows = [
        ['BoW-F1 mean / med', f"{pa['f1_mean'][a]:.3f} / {pa['f1_med'][a]:.3f}",
         f"{pa['f1_mean'][b]:.3f} / {pa['f1_med'][b]:.3f}"],
        ['Слов-ошибок на документ (mean / med / p90)',
         f"{ea['mean']:.1f} / {ea['med']} / {ea['p90']}",
         f"{eb['mean']:.1f} / {eb['med']} / {eb['p90']}"],
        ['Документов F1 ≥0.95 / 0.80–0.95 / 0.50–0.80 / <0.50',
         f"{ba['>=0.95']} / {ba['0.80-0.95']} / {ba['0.50-0.80']} / {ba['<0.50']}",
         f"{bb['>=0.95']} / {bb['0.80-0.95']} / {bb['0.50-0.80']} / {bb['<0.50']}"],
        ['Средний F1 на 20 худших', f"{pa['tail20'][a]:.2f}", f"{pa['tail20'][b]:.2f}"],
        ['Пропущено слов эталона (Σ)', str(pa['missed_total'][a]), str(pa['missed_total'][b])],
        ['Выдумано шумовых слов (Σ)', str(pa['invented_total'][a]), str(pa['invented_total'][b])],
        ['…из них со смешанной кириллицей+латиницей', str(pa['mixed'][a]), str(pa['mixed'][b])],
        ['Repetition-loop (покрытие > 1.5), документов', str(pa['loops'][a]), str(pa['loops'][b])],
        ['Цифровые токены потеряно/искажено', f"{pa['digit_loss'][a]:.1f}%", f"{pa['digit_loss'][b]:.1f}%"],
    ]
    add_table(doc, ['Метрика', 'DeepSeek-OCR', 'PaddleOCR'], rows)
    add_note(doc, f'Парное сравнение: DeepSeek-OCR лучше на {pa["wins"]} документах '
                  f'({100*pa["wins"]/pa["n"]:.0f}%), PaddleOCR — на {pa["losses"]} '
                  f'({100*pa["losses"]/pa["n"]:.0f}%), ничья — {pa["ties"]} '
                  f'({100*pa["ties"]/pa["n"]:.0f}%). Разница F1: mean {pa["delta_mean"]:+.3f}, '
                  f'med {pa["delta_med"]:+.3f}.')
    doc.add_picture(str(chart_pair_scatter(pa)), width=Cm(15))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    doc.add_heading('4.7.1. Характер ошибок PaddleOCR', level=3)
    add_body(doc,
             'Систематические посимвольные подмены кириллицы на похожую латиницу/цифры: '
             'текст визуально «почти правильный», но реквизиты и слова испорчены для поиска '
             'и эмбеддингов. Типичные замены (топ-12):')
    rows = [[f'{w} → {x}', str(c)] for (w, x), c in pa['sub_pairs'][b][:12]]
    add_table(doc, ['Эталон → OCR PaddleOCR', '×'], rows)

    doc.add_heading('4.7.2. Характер ошибок DeepSeek-OCR', level=3)
    add_body(doc,
             'Ошибки концентрируются в нетекстовых зонах (печати/подписи) и в markdown-разметке '
             'таблиц. Систематических подмен букв почти нет; в зоне М.П. и подписей встречаются '
             'галлюцинации вроде «фио», «уполномоченное», «реквизиты». Типичные замены (топ-12):')
    rows = [[f'{w} → {x}', str(c)] for (w, x), c in pa['sub_pairs'][a][:12]]
    add_table(doc, ['Эталон → OCR DeepSeek-OCR', '×'], rows)

    doc.add_heading('4.7.3. Катастрофические сбои', level=3)
    add_body(doc,
             f'PaddleOCR F1<0.5 при DeepSeek-OCR F1≥0.8: {len(pa["b_fail"])} документ — '
             + (', '.join(f'{f[:60]} (Paddle={fb:.2f}, DS={fa:.2f})'
                          for f, fa, fb in pa['b_fail']) or 'нет') + '.')
    add_body(doc,
             f'DeepSeek-OCR F1<0.5 при PaddleOCR F1≥0.8: {len(pa["a_fail"])} документ — '
             + (', '.join(f'{f[:60]} (DS={fa:.2f}, Paddle={fb:.2f})'
                          for f, fa, fb in pa['a_fail']) or 'нет') + '.')
    add_body(doc,
             f'Вывод. На типичном документе у PaddleOCR в ~2.5 раза больше слов-ошибок '
             f'(медиана {eb["med"]} против {ea["med"]}). DeepSeek-OCR выигрывает на '
             f'{100*pa["wins"]/pa["n"]:.0f}% документов. Главный недостаток PaddleOCR — '
             f'систематические подмены букв (ООО→о00/000, руб.→py6., №→n, инн→иhh): '
             f'шума со смешанной кириллицей+латиницей у PaddleOCR '
             f'{pa["mixed"][b]} токенов против {pa["mixed"][a]} у DeepSeek-OCR. '
             f'Контраргумент — устойчивость к повороту 90° (см. 4.6): PaddleOCR F1=0.81, '
             f'DeepSeek-OCR F1=0.09 — поэтому в пайплайн обязательно нужен шаг '
             f'коррекции ориентации перед OCR (при нём разница снимается).')

    # ── 4.8. Парные сравнения новых движков (27.07) ─────────────
    doc.add_heading('4.8. Парные сравнения новых движков (Qwen3-VL-8B/4B, '
                    'DeepSeek-OCR-2, FineReader 16)', level=2)
    add_body(doc,
             'Все новые движки прогнаны на том же зафиксированном наборе из 200 '
             'документов (парный дизайн, промпт v2 для Qwen3-VL, для 4B — '
             'repetition_penalty 1.15). Ниже — парные сравнения новых движков '
             'против DeepSeek-OCR v1 (чемпиона 3B-класса по итогам 24.07), '
             'а также сверка с точкой отсчёта качества — ABBYY FineReader 16.')

    def pair_block(idx, a, b, title, conclusion):
        p = paired_analysis(a, b)
        ea_, eb_ = p['err_doc'][a], p['err_doc'][b]
        ba_, bb_ = p['buckets'][a], p['buckets'][b]
        doc.add_heading(f'4.8.{idx}. {title}', level=3)
        rows = [
            ['BoW-F1 mean / med',
             f"{p['f1_mean'][a]:.3f} / {p['f1_med'][a]:.3f}",
             f"{p['f1_mean'][b]:.3f} / {p['f1_med'][b]:.3f}"],
            ['Слов-ошибок на документ (med / p90)',
             f"{ea_['med']} / {ea_['p90']}", f"{eb_['med']} / {eb_['p90']}"],
            ['Документов F1 ≥0.95 / <0.50',
             f"{ba_['>=0.95']} / {ba_['<0.50']}",
             f"{bb_['>=0.95']} / {bb_['<0.50']}"],
            ['Средний F1 на 20 худших',
             f"{p['tail20'][a]:.2f}", f"{p['tail20'][b]:.2f}"],
            ['Пропущено слов эталона (Σ)',
             str(p['missed_total'][a]), str(p['missed_total'][b])],
            ['Выдумано шумовых слов (Σ)',
             str(p['invented_total'][a]), str(p['invented_total'][b])],
            ['Смешанная кириллица+латиница',
             str(p['mixed'][a]), str(p['mixed'][b])],
            ['Repetition-loop (cov>1.5), документов',
             str(p['loops'][a]), str(p['loops'][b])],
            ['Цифровые токены потеряно',
             f"{p['digit_loss'][a]:.1f}%", f"{p['digit_loss'][b]:.1f}%"],
        ]
        add_table(doc, ['Метрика', DISPLAY[a], DISPLAY[b]], rows)
        add_note(doc, f"Парное сравнение: {DISPLAY[a]} лучше на {p['wins']} "
                      f"({100*p['wins']/p['n']:.0f}%), {DISPLAY[b]} — на "
                      f"{p['losses']} ({100*p['losses']/p['n']:.0f}%), ничья — "
                      f"{p['ties']} ({100*p['ties']/p['n']:.0f}%). Разница F1: "
                      f"mean {p['delta_mean']:+.3f}, med {p['delta_med']:+.3f}.")
        add_body(doc, conclusion, bold_prefix='Вывод. ')
        return p

    pa_q8 = pair_block(
        1, 'qwen3vl', 'deepseek',
        'Qwen3-VL-8B vs DeepSeek-OCR v1',
        'Qwen3-VL-8B уверенно лучше: выигрыш на 62% документов, в 3 раза меньше '
        'слов-ошибок на типичном документе (медиана 2 против 6), цифры и '
        'реквизиты читает заметно точнее (потери 16.5% против 25.4%), 75% '
        'документов с F1≥0.95 против 48%. Ошибки — в основном безобидные '
        'варианты тире. Минусы: в 3.7 раза медленнее (17.1 vs 4.6 с/стр) и '
        'repetition-loop чаще (11 против 2): на бланках возможна HTML/SVG-'
        'галлюцинация base64 до max tokens (2 документа, F1=0.00) — ловится '
        'фильтром покрытия >1.5.')
    doc.add_picture(str(chart_pair_scatter(pa_q8)), width=Cm(15))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

    pair_block(
        2, 'qwen3vl4b', 'deepseek',
        'Qwen3-VL-4B vs DeepSeek-OCR v1',
        'Статистическая ничья по качеству (разница F1 −0.0002), но 4B в 3 раза '
        'медленнее (14.6 vs 4.6 с/стр: dense 4B против MoE ~570M активных), '
        'шумнее (mixed-script 103 против 33, выдуманных слов 2983 против 1775) '
        'и чаще зацикливается (8 против 2 — даже с repetition_penalty 1.15, '
        'без него на бланках imgur-галлюцинация до F1=0.00). «Золотой '
        'серединой» 4B не является: качество в семействе Qwen3-VL круто растёт '
        'с размером — 4B даёт ровно 3B-класс, а не середину между 3B и 8B.')

    pair_block(
        3, 'deepseek2', 'deepseek',
        'DeepSeek-OCR-2 vs DeepSeek-OCR v1',
        'Апгрейд не состоялся: v1 лучше на 45% документов против 32% '
        '(разница −0.0135). Регрессии v2 на RU-документах: markdown-мусор '
        '(«всего→**всего», «руб.→руб.**» — шумовых слов в 2.2 раза больше), '
        'посимвольные подмены кириллицы («Толстобров→Толстодроб», «сети→селу», '
        'укр. «І» — mixed-script в 4.5 раза больше), тяжелее хвост (p90 ошибок '
        '59 против 43), медленнее на 37% (6.3 vs 4.6 с/стр). Плюс только в '
         'цифрах (22.9% против 25.4%). EN-бенчмарки (olmOCR-bench 76.3 vs 75.7) '
         'на RU-корпус не перенеслись. В 3B-классе остаётся DeepSeek-OCR v1.')

    pair_block(
        4, 'finereader', 'deepseek',
        'Точка отсчёта: FineReader 16 (ABBYY) vs DeepSeek-OCR v1',
        'Мёртвая ничья по качеству (40/42%, разница −0.003): FineReader даёт '
        'лучшую полноту извлечения из всех движков (1889 пропущенных слов '
        'против 3019), самый чистый текст (9 смешанных токенов), лучшие цифры '
        '(18.8% против 25.4%), нулевые катастрофические сбои и лучший хвост '
        '(0.56) — и всё это в 3.5 раза быстрее на CPU (1.3 с/стр против 4.6 '
        'на GPU). Характерные ошибки — регистровые подмены на стилизованных '
        'гарнитурах («сотовой→сотобой», «в→б/6») и пунктуация. Вывод по '
        'калибровке: классический эталон подтверждает уровень DeepSeek-OCR '
        '(паритет), а Qwen3-VL-8B превосходит эталон (см. 4.8.1).')

    pair_block(
        5, 'gemma4', 'deepseek',
        'Gemma-4-E4B-it (google) vs DeepSeek-OCR v1',
        'DeepSeek лучше по всем позициям: gemma4 проигрывает 55% документов '
        '(разница −0.022), пропускает больше слов (4070 против 3019), шумит '
        'сильнее (3305 против 1775), цифры — худший результат среди VLM '
        '(потери 30.2% против 25.4%; характерная ошибка «0→01»), лупы 10 '
        'против 2, тяжелее хвост (p90 ошибок 70 против 43) и в 4.7 раза '
        'медленнее (21.4 vs 4.6 с/стр). Против Qwen3-VL-8B проигрывает 79% '
        'документов. Плюс — аккуратный plain-text вывод без '
        'галлюцинаций-катастроф. Крупные варианты gemma-4 (12B/26B-A4B/31B) '
        'в 24 GB VRAM без квантизации не помещаются. Вывод: конкурентом '
        'не является.')

    # ── 5. Выводы ────────────────────────────────────────────────
    doc.add_heading('5. Выводы и рекомендации', level=1)
    doc.add_heading('5.1. Итоговый рейтинг', level=2)
    ranking = [
        ('1. Qwen3-VL-8B — максимум качества; единственный кандидат на схему '
         '«одна VLM на всё». ',
         'Лучшие все метрики качества: F1 0.931/0.979, 3gram 0.938, 75% '
         'документов с F1≥0.95, слов-ошибок медиана 2, потери цифр 16.5%. '
         'На него же обучена ориентация-LoRA (99.83%), он же может '
         'саммаризировать. Минусы: в 3.7 раза медленнее DeepSeek (17.1 с/стр), '
         'repetition-loop на 11 документах (2 катастрофы F1=0.00 — фильтр '
         'покрытия обязателен), ~17 GB VRAM, чувствительность к промпту '
         '(нужен «Transcribe the entire document…», иначе извлекает только '
         'таблицу).'),
        ('2. DeepSeek-OCR v1 — оптимум скорости/качества в 3B-классе; '
         'рекомендуется как основной OCR при ограниченном GPU-бюджете. ',
         'F1 0.901/0.943, лучший «хвост» (0.55), 4.6 с/стр (MoE ~570M '
         'активных параметров), ~7 GB VRAM. Защитил титул 3B-класса против '
         'трёх претендентов (Nanonets, Qwen3-VL-4B, DeepSeek-OCR-2). Минус: '
         'полностью деградирует при повороте 90° (F1 0.09) — обязателен шаг '
         'коррекции ориентации (он нужен в любом случае: 180° не читает ни '
         'один движок).'),
        ('3. Nanonets-OCR2-3B — запасной VLM 3B-класса. ',
         'Равное DeepSeek качество (F1 0.906/0.957), наиболее устойчив к '
         'повороту 90° среди VLM (0.78), но в 2.3 раза медленнее DeepSeek '
         '(18.6 с/стр) и слабее на «хвосте» (0.48).'),
        ('4. Qwen3-VL-4B, DeepSeek-OCR-2 и Gemma-4-E4B — не рекомендуются. ',
         '4B: качество ровно 3B-класс (0.900), но в 3 раза медленнее DeepSeek '
         'и шумнее; без repetition_penalty 1.15 — imgur-галлюцинации на '
         'бланках. OCR-2: на RU хуже v1 (markdown-мусор ×2.2, подмены '
         'кириллицы ×4.5, хвост тяжелее) и медленнее на 37%; EN-бенчмарки '
         'на наш корпус не перенеслись. Gemma-4-E4B: ниже DeepSeek по всем '
         'метрикам (0.879/0.922, цифры 30.2%, лупы 10) и в 4.7 раза медленнее '
         '(21.4 с/стр).'),
        ('5. PaddleOCR — вариант без GPU (бесплатный). ',
         'F1 0.823/0.864 на CPU (6.2 с/стр); слабый хвост на стилизованных '
         'документах. Подходит как fallback и для дешёвого массового прохода.'),
        ('6. Tesseract — исходный референс. ',
         'Самый быстрый (1.1 с/стр), но худшее среднее качество (F1 0.758) '
         'из-за систематических кириллических замен; на сканах корпуса '
         'использовать не рекомендуется.'),
    ]
    for prefix, text in ranking:
        add_body(doc, text, bold_prefix=prefix)
    add_body(doc,
             'ABBYY FineReader 16 участвовала в пилоте только как точка '
             'отсчёта качества распознавания (классический эталон на '
             'кириллице, см. 4.8.4) и в рейтинг кандидатов не входит.',
             bold_prefix='Точка отсчёта. ')
    doc.add_heading('5.2. Ограничения и риски', level=2)
    for t in [
        'VLM-модели зацикливаются (repetition-loop): DeepSeek — 2 документа '
        'из 200, Qwen3-VL-8B — 11, Qwen3-VL-4B — 8 (даже с rep-penalty), '
        'DeepSeek-OCR-2 — 6; крайняя форма — HTML/SVG-галлюцинация base64 '
        '(Qwen3-VL на бланках, F1=0.00). В проде обязателен фильтр по '
        'покрытию (>1.5) с перенаправлением на запасной движок;',
        'Qwen3-VL критично чувствителен к промпту: короткий промпт в стиле '
        'nanonets → извлекается только таблица, без шапки и тела письма '
        '(F1 0.36). Нужен «Transcribe the entire document… include ALL '
        'elements». Для 4B обязателен repetition_penalty=1.15;',
        'оценка скорости VLM — HF generate без батчирования; vLLM-инференс '
        'может дать кратное ускорение (не тестировалось);',
        'эталон fitz для pdf_text неидеален: порядок строк в таблицах '
        'отличается от визуального (учтено выбором порядко-независимых метрик);',
        'Nanonets и Qwen3-VL чувствительны к разрешению: рендер 300 dpi без '
        'даунскейла не помещается в 24 GB (в пилоте — даунскейл до 1600 px);',
        'DeepSeek-OCR и OCR-2 закреплены за transformers 4.46.x (несовместимы '
        'с ≥4.48), Nanonets — за ≥4.56, Qwen3-VL — за ≥4.57: два отдельных '
        'окружения (venv_dsocr / venv_ocr);',
        'ни один движок не читает повёрнутые сканы (180° — у всех F1 ≤ 0.13; '
        '90° — DeepSeek 0.09): коррекция ориентации перед OCR обязательна '
        '(готовые модели проекта: Qwen3-VL+LoRA 99.8% либо лёгкие '
        'PP-LCNet/EfficientNet/ViT-Large). Qwen3-VL-8B/4B и DeepSeek-OCR-2 '
        'в тесте поворота не участвовали;',
        'DeepSeek-OCR-2: разрешение входа зашито (base 1024 + тайлы 768), '
        'иные base_size ломают кастомный код энкодера.',
    ]:
        add_bullet(doc, t)

    # ── Приложения ───────────────────────────────────────────────
    doc.add_heading('Приложение А. Технические детали', level=1)
    tech = [
        ('Каталог эксперимента', 'experiment_ocr/ (engines/oc_*.py, phase_o1/o2, '
         'результаты results/o1_results.jsonl)'),
        ('Окружения', 'venv_ocr (tesseract/paddle/nanonets/qwen3vl, transformers '
         '4.57.6, paddlepaddle 3.2.0); venv_dsocr (deepseek/deepseek2, '
         'transformers 4.46.3); venv_gemma (gemma4, transformers 5.14.1 — '
         'gemma4 требует ≥5.x) — все в C:\\Windows\\Temp\\opencode (ASCII-путь: '
         'venv в кириллическом пути неработоспособен); 27.07 venv_ocr и '
         'venv_dsocr пересозданы после повреждения site-packages'),
        ('PaddleOCR', 'paddlepaddle==3.2.0 обязателен: 3.3.x падает на CPU '
         '(баг oneDNN, SO 79884564); use_textline_orientation=True — native crash'),
        ('DeepSeek-OCR / OCR-2', 'transformers==4.46.3 (≥4.48 несовместим: '
         'LlamaFlashAttention2, DynamicCache.get_max_length); infer() возвращает '
         'текст только с eval_mode=True; у OCR-2 разрешение зашито '
         '(base 1024 / tiles 768, иные base_size — UnboundLocalError)'),
        ('Qwen3-VL-8B/4B', 'transformers ≥4.57 (venv_ocr обновлён до 4.57.6); '
         'промпт v2 «Transcribe the entire document…»; 4B — repetition_penalty '
         '1.15; даунскейл входа до 1600 px'),
        ('Nanonets-OCR2-3B', 'transformers ≥4.56; даунскейл входа до 1600 px '
         '(300 dpi → OOM на 24 GB)'),
        ('Модели', 'D:\\MODELS\\Transformers (deepseek-ai_DeepSeek-OCR ~6.7 GB, '
         'deepseek-ai_DeepSeek-OCR-2 ~6.7 GB, nanonets_Nanonets-OCR2-3B ~6.2 GB, '
         'Qwen3-VL-8B-Instruct ~17.5 GB, Qwen3-VL-4B-Instruct ~8.5 GB, '
         'google_gemma-4-E4B-it ~15.2 GB); paddle-кэш %USERPROFILE%\\.paddlex'),
        ('FineReader 16', 'десктоп C:\\Program Files\\ABBYY FineReader 16 '
         '(16.0.14.6564); автоматизация — задача Hot Folder «ocr_experiment» '
         '(Run once, fr_in → fr_out, Text UTF-8, [F].txt), создана '
         'UI-автоматизацией pywinauto (AWL-контролы не видны в UIA — '
         'навигация по скриншотам и координатам); прогон '
         'phase_o1c_finereader.py, 210 файлов за 280 с'),
        ('Прогоны', 'phase_o1_run.py (классические + deepseek/deepseek2, '
         '1 процесс); phase_o1b_qwen3vl.py (qwen3vl/qwen3vl4b, шард 2×GPU '
         '60/40 → слияние в o1_results.jsonl); phase_o1c_finereader.py '
         '(FineReader, Hot Folder); новые движки добавлены 27.07'),
        ('Тест поворота', 'phase_o3_rotation.py: 20 документов × 90°/180° × '
         '4 движка → results/rotation_test.jsonl (см. 4.6)'),
        ('Воспроизводимость', 'отчёт сгенерирован make_word_report.py из '
         'results/o1_results.jsonl; сводка — reports/ocr_compare.md; парные '
         'анализы — oc_pair_analysis.py; детально по новым движкам — '
         'reports/qwen3vl_vs_deepseek.md'),
    ]
    add_table(doc, ['Параметр', 'Значение'], [[k, v] for k, v in tech])

    footer = doc.sections[0].footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_page_number(footer)

    try:
        doc.save(OUT_DOCX)
        print(f'written {OUT_DOCX}')
    except PermissionError:
        alt = OUT_DOCX.with_name(OUT_DOCX.name.replace('_v2', '_v3'))
        doc.save(alt)
        print(f'ФАЙЛ {OUT_DOCX.name} ОТКРЫТ В WORD → отчёт сохранён в {alt}')


if __name__ == '__main__':
    main()
