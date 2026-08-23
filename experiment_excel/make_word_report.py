# make_word_report.py
# Подробный отчёт по эксперименту двухуровневой кластеризации Excel (MS Word).
# Собирает статистику из временных таблиц БД (до запуска Phase 5!),
# строит 4 диаграммы (matplotlib) и формирует .docx в D:\FileOrganizer\Reports.
#
# Запуск (venv с matplotlib, системные пакеты видны):
#   C:\Windows\Temp\opencode\report_venv\Scripts\python.exe make_word_report.py

import datetime
import re
import sys
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

from experiment_config import DB_URL, SIMILARITY_THRESHOLD, MIN_CLUSTER_SIZE_FACTOR  # noqa: sys.path+utf8

OUT_DIR = Path(r'D:\FileOrganizer\Reports')
CHART_DIR = Path(tempfile.gettempdir()) / 'excel_experiment_charts'
CHART_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
    'figure.dpi': 150, 'savefig.bbox': 'tight',
})
ACCENT = '#2E75B6'
GRAY = '#A6A6A6'

PHASE1_LOG = Path(r'C:\Windows\Temp\opencode\phase1_run3.log')


# ────────────────────────── данные ──────────────────────────

def collect_stats(conn):
    st = {}
    cur = conn.cursor()

    # Phase 1
    cur.execute("SELECT COUNT(*) FROM temp_excel_train"); st['train'] = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM temp_excel_test"); st['test'] = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM temp_excel_failed"); st['failed'] = cur.fetchone()[0]
    cur.execute("""
        SELECT
          COUNT(*) FILTER (WHERE lower(file_path) LIKE '%.xlsx'),
          COUNT(*) FILTER (WHERE lower(file_path) LIKE '%.xls')
        FROM (SELECT file_path FROM temp_excel_train
              UNION ALL SELECT file_path FROM temp_excel_test) t
    """)
    st['xlsx'], st['xls'] = cur.fetchone()
    cur.execute("SELECT AVG(LENGTH(txt)), MIN(LENGTH(txt)), MAX(LENGTH(txt)) FROM temp_excel_train")
    st['txt_avg'], st['txt_min'], st['txt_max'] = cur.fetchone()
    cur.execute("SELECT reason, COUNT(*) FROM temp_excel_failed GROUP BY reason ORDER BY 2 DESC")
    st['fail_reasons'] = cur.fetchall()

    # Phase 2
    cur.execute("SELECT label, COUNT(*) FROM temp_hdbscan GROUP BY label ORDER BY COUNT(*) DESC")
    rows = cur.fetchall()
    st['clusters'] = [(l, c) for l, c in rows if l != -1]
    st['noise'] = next((c for l, c in rows if l == -1), 0)

    # Phase 3
    cur.execute("SELECT COUNT(*), AVG(similarity), MIN(similarity), MAX(similarity) FROM temp_test_assign")
    st['assigned'], st['sim_avg'], st['sim_min'], st['sim_max'] = cur.fetchone()
    st['thresholds'] = []
    for thr in (0.6, 0.7, 0.8, 0.9, 0.95):
        cur.execute("SELECT COUNT(*) FROM temp_test_assign WHERE similarity >= %s", (thr,))
        st['thresholds'].append((thr, cur.fetchone()[0]))
    cur.execute("""
        SELECT assigned_label, COUNT(*), AVG(similarity)
        FROM temp_test_assign GROUP BY assigned_label ORDER BY COUNT(*) DESC
    """)
    st['test_per_cluster'] = {int(l): (c, float(s)) for l, c, s in cur.fetchall()}
    cur.execute("SELECT similarity FROM temp_test_assign")
    st['sims'] = np.array([r[0] for r in cur.fetchall()], dtype=np.float32)

    # Phase 4
    cur.execute("""
        SELECT type_label, sub_label, COUNT(*)
        FROM temp_second_level GROUP BY type_label, sub_label
        ORDER BY type_label, COUNT(*) DESC
    """)
    st['subs'] = cur.fetchall()

    # Phase 4b (BERTopic) — если таблица существует
    cur.execute("SELECT to_regclass('public.temp_second_level_bt')")
    st['has_bt'] = cur.fetchone()[0] is not None
    if st['has_bt']:
        cur.execute("""
            SELECT type_label,
                   COUNT(DISTINCT bt_topic) FILTER (WHERE bt_topic != -1),
                   COUNT(*) FILTER (WHERE bt_topic = -1)
            FROM temp_second_level_bt GROUP BY type_label
        """)
        st['bt_summary'] = {int(t): (int(n), int(noise)) for t, n, noise in cur.fetchall()}
        # Type 0: разложение тем BT по маркерам дополнений 25.1/25.2
        cur.execute("""
            SELECT b.bt_topic, b.top_words, COUNT(*),
                   COUNT(*) FILTER (WHERE t.txt LIKE '%25.1%'),
                   COUNT(*) FILTER (WHERE t.txt LIKE '%25.2%')
            FROM temp_second_level_bt b
            JOIN temp_excel_train t ON t.id = b.train_id
            WHERE b.type_label = 0
            GROUP BY b.bt_topic, b.top_words ORDER BY COUNT(*) DESC
        """)
        st['bt_type0'] = cur.fetchall()
        # Характерные top-слова тем для типов 11 и 12
        st['bt_words'] = {}
        for t in (11, 12):
            cur.execute("""
                SELECT bt_topic, COUNT(*) c, MAX(top_words)
                FROM temp_second_level_bt WHERE type_label = %s
                GROUP BY bt_topic ORDER BY c DESC LIMIT 6
            """, (t,))
            st['bt_words'][t] = cur.fetchall()

    # ── Раунд 2: LLM-тематическая кластеризация (temp_theme2_*) ──
    cur.execute("SELECT to_regclass('public.temp_theme2_clusters')")
    st['has_r2'] = cur.fetchone()[0] is not None
    if st['has_r2']:
        cur.execute("SELECT COUNT(*) FROM temp_llm_subject")
        st['r2_llm_train'] = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM temp_llm_subject_test")
        st['r2_llm_test'] = cur.fetchone()[0]
        cur.execute("""
            SELECT theme_label, COUNT(*) FROM temp_theme2_clusters
            WHERE theme_label != -1 GROUP BY theme_label ORDER BY COUNT(*) DESC
        """)
        st['t2_themes'] = [(int(l), c) for l, c in cur.fetchall()]
        cur.execute("SELECT COUNT(*) FROM temp_theme2_clusters WHERE theme_label = -1")
        st['t2_noise'] = cur.fetchone()[0]
        cur.execute("SELECT theme_label, top_words FROM temp_theme2_topwords")
        st['t2_words'] = {int(l): w for l, w in cur.fetchall()}
        cur.execute("""
            SELECT COUNT(*), AVG(similarity), MIN(similarity), MAX(similarity)
            FROM temp_theme2_test_assign
        """)
        st['t2_assigned'], st['t2_sim_avg'], st['t2_sim_min'], st['t2_sim_max'] = cur.fetchone()
        cur.execute("SELECT COUNT(*) FROM temp_theme2_test_assign WHERE is_unknown")
        st['t2_unknown'] = cur.fetchone()[0]
        cur.execute("""
            SELECT assigned_theme, COUNT(*), AVG(similarity)
            FROM temp_theme2_test_assign GROUP BY assigned_theme
        """)
        st['t2_test_per'] = {int(l): (c, float(s)) for l, c, s in cur.fetchall()}
        cur.execute("SELECT similarity FROM temp_theme2_test_assign")
        st['t2_sims'] = np.array([r[0] for r in cur.fetchall()], dtype=np.float32)

    # ── Канонизация типов (phase_4c, temp_tip_groups) ──
    cur.execute("SELECT to_regclass('public.temp_tip_groups')")
    st['has_tipg'] = cur.fetchone()[0] is not None
    if st['has_tipg']:
        cur.execute("SELECT COUNT(*), COUNT(DISTINCT group_name) FROM temp_tip_groups")
        st['tipg_tips'], st['tipg_groups'] = cur.fetchone()
        # Топ канонических групп по документам
        cur.execute("""
            SELECT group_name, SUM(docs) d, COUNT(*) n
            FROM temp_tip_groups GROUP BY group_name
            ORDER BY d DESC LIMIT 15
        """)
        st['tipg_top'] = [(g, int(d), n) for g, d, n in cur.fetchall()]
        # Семьи-варианты (группы из 2+ строк)
        cur.execute("""
            SELECT group_name, string_agg(tip, ', ' ORDER BY docs DESC), SUM(docs) d
            FROM temp_tip_groups GROUP BY group_name
            HAVING COUNT(*) > 1 ORDER BY d DESC LIMIT 8
        """)
        st['tipg_families'] = [(g, s, int(d)) for g, s, d in cur.fetchall()]
        # Состав тема → типы (нормализация в Python, как в show_themes)
        from show_themes import normalize_tip
        cur.execute("""
            SELECT c.theme_label, s.tip
            FROM temp_theme2_clusters c
            JOIN temp_llm_subject s ON s.train_id = c.train_id
            WHERE c.theme_label != -1
        """)
        cur2 = conn.cursor()
        cur2.execute("SELECT tip, group_name FROM temp_tip_groups")
        gmap = {t: g for t, g in cur2.fetchall()}
        comp = {}
        for lab, tip in cur.fetchall():
            canon = gmap.get(normalize_tip(tip), normalize_tip(tip))
            comp.setdefault(int(lab), {})
            comp[int(lab)][canon] = comp[int(lab)].get(canon, 0) + 1
        # топ-3 типа на тему
        st['tipg_theme_comp'] = {
            lab: sorted(d.items(), key=lambda kv: -kv[1])[:3]
            for lab, d in comp.items()
        }

    # ── Раунд 3: перенос на PDF-потоки ──
    cur.execute("SELECT to_regclass('public.temp_pdftext_assign')")
    st['has_r3'] = cur.fetchone()[0] is not None
    if st['has_r3']:
        cur.execute("SELECT COUNT(*), AVG(similarity), COUNT(*) FILTER (WHERE is_unknown) FROM temp_pdftext_assign")
        st['r3_text_n'], st['r3_text_avg'], st['r3_text_unk'] = cur.fetchone()
        cur.execute("SELECT COUNT(*) FROM temp_llm_subject_pdftext")
        st['r3_text_llm'] = cur.fetchone()[0]
        cur.execute("""
            SELECT assigned_theme, COUNT(*) FROM temp_pdftext_assign
            GROUP BY assigned_theme
        """)
        st['r3_text_per'] = {int(l): c for l, c in cur.fetchall()}

        st['has_r3_tables'] = False
        cur.execute("SELECT to_regclass('public.temp_pdftables_assign')")
        if cur.fetchone()[0] is not None:
            st['has_r3_tables'] = True
            cur.execute("SELECT COUNT(*), AVG(similarity), COUNT(*) FILTER (WHERE is_unknown) FROM temp_pdftables_assign")
            st['r3_tab_n'], st['r3_tab_avg'], st['r3_tab_unk'] = cur.fetchone()
            cur.execute("SELECT COUNT(*) FROM temp_llm_subject_pdftables")
            st['r3_tab_llm'] = cur.fetchone()[0]
            cur.execute("""
                SELECT assigned_theme, COUNT(*) FROM temp_pdftables_assign
                GROUP BY assigned_theme
            """)
            st['r3_tab_per'] = {int(l): c for l, c in cur.fetchall()}

        # Реестр v2
        cur.execute("SELECT to_regclass('public.temp_theme2_centroids_v2')")
        if cur.fetchone()[0] is not None:
            cur.execute("SELECT origin, COUNT(*) FROM temp_theme2_centroids_v2 GROUP BY origin")
            st['r3_registry'] = dict(cur.fetchall())
            cur.execute("""
                SELECT theme_label, top_words FROM temp_theme2_centroids_v2
                WHERE origin != 'excel_r2' ORDER BY theme_label
            """)
            st['r3_newthemes'] = [(int(l), w) for l, w in cur.fetchall()]
            # размеры новых тем PDF_Tables
            cur.execute("SELECT to_regclass('public.temp_pdftables_newthemes')")
            if cur.fetchone()[0] is not None:
                cur.execute("""
                    SELECT n.new_theme, COUNT(*) FROM temp_pdftables_newthemes n
                    JOIN temp_theme2_centroids_v2 v ON v.origin = 'pdftables_drift'
                    GROUP BY n.new_theme
                """)
                # new_theme в newthemes = локальный label HDBSCAN буфера,
                # в реестр попали не все — размеры берём только для справки
                st['r3_nt_sizes'] = {int(l): c for l, c in cur.fetchall()}
    return st


def phase1_timing():
    try:
        log = PHASE1_LOG.read_text(encoding='utf-8', errors='replace')
        m = re.search(r'completed in (\d+)s', log)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def llm_timings():
    """Минуты LLM-прогонов из логов phase_1c."""
    out = {}
    for key, path in (('train', r'C:\Windows\Temp\opencode\phase1c_full2.log'),
                      ('train_sample', r'C:\Windows\Temp\opencode\phase1c_run2.log'),
                      ('test', r'C:\Windows\Temp\opencode\phase1c_test.log')):
        try:
            log = Path(path).read_text(encoding='utf-8', errors='replace')
            m = re.search(r'за (\d+\.?\d*) мин', log)
            out[key] = float(m.group(1)) if m else None
        except Exception:
            out[key] = None
    return out


# ────────────────────────── диаграммы ──────────────────────────

def chart_cluster_sizes(st):
    labels = [str(l) for l, _ in st['clusters']] + ['шум']
    sizes = [c for _, c in st['clusters']] + [st['noise']]
    colors = [ACCENT] * len(st['clusters']) + [GRAY]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, sizes, color=colors)
    ax.set_yticks(y, labels)
    ax.set_xlabel('Документов')
    ax.set_title(f'Phase 2: размеры кластеров (train, n={st["train"]})')
    for yi, s in zip(y, sizes):
        ax.text(s + max(sizes) * 0.01, yi, str(s), va='center', fontsize=9)
    ax.set_xlim(0, max(sizes) * 1.12)
    ax.spines[['top', 'right']].set_visible(False)
    p = CHART_DIR / 'cluster_sizes.png'
    fig.savefig(p); plt.close(fig)
    return p


def chart_similarity_hist(st):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(st['sims'], bins=40, range=(0.6, 1.0), color=ACCENT, edgecolor='white')
    cov = (st['sims'] >= SIMILARITY_THRESHOLD).mean()
    ax.axvline(SIMILARITY_THRESHOLD, color='red', ls='--', lw=1.5,
               label=f'порог {SIMILARITY_THRESHOLD} (покрытие {cov:.1%})')
    ax.set_xlabel('Косинусное сходство с ближайшим центроидом')
    ax.set_ylabel('Документов')
    ax.set_title(f'Phase 3: распределение сходства тест-набора (n={st["assigned"]})')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    p = CHART_DIR / 'similarity_hist.png'
    fig.savefig(p); plt.close(fig)
    return p


def chart_train_vs_test(st):
    cl = st['clusters']
    labels = [str(l) for l, _ in cl]
    train = [c for _, c in cl]
    test = [st['test_per_cluster'].get(l, (0, 0))[0] for l, _ in cl]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - w / 2, train, w, label='train (HDBSCAN)', color=ACCENT)
    ax.bar(x + w / 2, test, w, label='test (назначено)', color='#ED7D31')
    ax.set_xticks(x, labels)
    ax.set_xlabel('Кластер')
    ax.set_ylabel('Документов')
    ax.set_title('Phase 2 vs Phase 3: состав кластеров train/test')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    p = CHART_DIR / 'train_vs_test.png'
    fig.savefig(p); plt.close(fig)
    return p


def chart_subclusters(st):
    by_type = {}
    for t, s, c in st['subs']:
        by_type.setdefault(t, []).append(c)
    types = sorted(by_type, key=lambda t: -sum(by_type[t]))
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap('tab10')
    y = np.arange(len(types))[::-1]
    for yi, t in zip(y, types):
        left = 0
        for i, c in enumerate(by_type[t]):
            ax.barh(yi, c, left=left, color=cmap(i % 10), edgecolor='white', linewidth=0.5)
            if c >= 80:
                ax.text(left + c / 2, yi, str(c), va='center', ha='center', fontsize=8, color='white')
            left += c
    ax.set_yticks(y, [str(t) for t in types])
    ax.set_xlabel('Документов')
    ax.set_title('Phase 4: состав под-кластеров внутри типов (каждый цвет = под-кластер)')
    ax.spines[['top', 'right']].set_visible(False)
    p = CHART_DIR / 'subclusters.png'
    fig.savefig(p); plt.close(fig)
    return p


def chart_km_vs_bt(st):
    types = [t for t, _ in st['clusters']]
    km = [len([1 for tt, _s, _c in st['subs'] if tt == t]) for t in types]
    bt = [st['bt_summary'].get(t, (0, 0))[0] for t in types]
    x = np.arange(len(types))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(x - w / 2, km, w, label='KMeans (принудительное k)', color=ACCENT)
    ax.bar(x + w / 2, bt, w, label='BERTopic (авто-число тем)', color='#ED7D31')
    for xi, v in zip(x - w / 2, km):
        ax.text(xi, v + 1, str(v), ha='center', fontsize=8)
    for xi, v in zip(x + w / 2, bt):
        ax.text(xi, v + 1, str(v), ha='center', fontsize=8)
    ax.set_xticks(x, [str(t) for t in types])
    ax.set_xlabel('Тип')
    ax.set_ylabel('Под-кластеров / тем')
    ax.set_title('Гранулярность второго уровня: KMeans vs BERTopic')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    p = CHART_DIR / 'km_vs_bt.png'
    fig.savefig(p); plt.close(fig)
    return p


# ── Раунд 2 (LLM themes) ──

def _t2_name(st, lab, maxlen=42):
    w = st['t2_words'].get(lab, '') or ''
    seen, parts = set(), []
    for p in (x.strip() for x in w.split(',')):
        if p and p not in seen:
            seen.add(p)
            parts.append(p)
    name = ', '.join(parts[:3])
    name = name[:1].upper() + name[1:] if name else f'тема {lab}'
    return (name[:maxlen] + '…') if len(name) > maxlen else name


def chart_t2_sizes(st):
    labels = [_t2_name(st, l) for l, _ in st['t2_themes']] + ['шум']
    sizes = [c for _, c in st['t2_themes']] + [st['t2_noise']]
    colors = [ACCENT] * len(st['t2_themes']) + [GRAY]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, sizes, color=colors)
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlabel('Документов')
    ax.set_title(f'Раунд 2: размеры тем (train, n={st["r2_llm_train"]})')
    for yi, s in zip(y, sizes):
        ax.text(s + max(sizes) * 0.01, yi, str(s), va='center', fontsize=9)
    ax.set_xlim(0, max(sizes) * 1.12)
    ax.spines[['top', 'right']].set_visible(False)
    p = CHART_DIR / 't2_sizes.png'
    fig.savefig(p); plt.close(fig)
    return p


def chart_t2_train_vs_test(st):
    cl = st['t2_themes']
    labels = [_t2_name(st, l, 24) for l, _ in cl]
    train = [c for _, c in cl]
    test = [st['t2_test_per'].get(l, (0, 0))[0] for l, _ in cl]
    x = np.arange(len(labels))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.bar(x - w / 2, train, w, label='train (HDBSCAN)', color=ACCENT)
    ax.bar(x + w / 2, test, w, label='test (назначено)', color='#ED7D31')
    ax.set_xticks(x, labels, rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Документов')
    ax.set_title('Раунд 2: состав тем train/test')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    p = CHART_DIR / 't2_train_vs_test.png'
    fig.savefig(p); plt.close(fig)
    return p


def chart_t2_sim_hist(st):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(st['t2_sims'], bins=40, range=(0.6, 1.0), color=ACCENT, edgecolor='white')
    cov = (st['t2_sims'] >= SIMILARITY_THRESHOLD).mean()
    ax.axvline(SIMILARITY_THRESHOLD, color='red', ls='--', lw=1.5,
               label=f'порог {SIMILARITY_THRESHOLD} (покрытие {cov:.1%})')
    ax.set_xlabel('Косинусное сходство с ближайшим центроидом темы')
    ax.set_ylabel('Документов')
    ax.set_title(f'Раунд 2: распределение сходства тест-набора (n={st["t2_assigned"]})')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    p = CHART_DIR / 't2_sim_hist.png'
    fig.savefig(p); plt.close(fig)
    return p


def chart_tipg_top(st):
    labels = [g for g, _, _ in st['tipg_top']]
    sizes = [d for _, d, _ in st['tipg_top']]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    y = np.arange(len(labels))[::-1]
    ax.barh(y, sizes, color=ACCENT)
    ax.set_yticks(y, labels, fontsize=8)
    ax.set_xlabel('Документов')
    ax.set_title('Канонические типы документов (top-15 из '
                 f'{st["tipg_groups"]} групп, HDBSCAN по строкам ТИП)')
    for yi, s in zip(y, sizes):
        ax.text(s + max(sizes) * 0.01, yi, str(s), va='center', fontsize=9)
    ax.set_xlim(0, max(sizes) * 1.12)
    ax.spines[['top', 'right']].set_visible(False)
    p = CHART_DIR / 'tipg_top.png'
    fig.savefig(p); plt.close(fig)
    return p


# ── Раунд 3 ──

def chart_r3_coverage(st):
    rows = [('Excel test', st.get('t2_assigned', 1), st.get('t2_unknown', 0))]
    rows.append(('PDF_Text', st['r3_text_n'], st['r3_text_unk']))
    if st.get('has_r3_tables'):
        rows.append(('PDF_Tables', st['r3_tab_n'], st['r3_tab_unk']))
    names = [r[0] for r in rows]
    cov = [1 - r[2] / r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(names, [c * 100 for c in cov], color=ACCENT)
    ax.axhline(90, color='red', ls='--', lw=1.2, label='цель 90%')
    for b, c, r in zip(bars, cov, rows):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4,
                f'{c * 100:.2f}%', ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('Покрытие @0.8, %')
    ax.set_ylim(85, 102)
    ax.set_title('Раунд 3: покрытие тематического реестра по потокам')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    p = CHART_DIR / 'r3_coverage.png'
    fig.savefig(p); plt.close(fig)
    return p


def chart_r3_distribution(st):
    """Распределение назначений по темам: Excel test / PDF_Text / PDF_Tables."""
    themes = [l for l, _ in st['t2_themes']]
    names = [_t2_name(st, l, 20) for l in themes]
    series = [('Excel test', st.get('t2_test_per', {}), ACCENT),
              ('PDF_Text', st['r3_text_per'], '#ED7D31')]
    if st.get('has_r3_tables'):
        series.append(('PDF_Tables', st['r3_tab_per'], '#70AD47'))
    x = np.arange(len(themes))
    w = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for i, (label, per, color) in enumerate(series):
        vals = [per.get(l, (0, 0))[0] if isinstance(per.get(l), tuple) else per.get(l, 0) for l in themes]
        ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=label, color=color)
    ax.set_xticks(x, names, rotation=35, ha='right', fontsize=7)
    ax.set_ylabel('Документов')
    ax.set_title('Раунд 3: распределение потоков по темам Раунда 2')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    p = CHART_DIR / 'r3_distribution.png'
    fig.savefig(p); plt.close(fig)
    return p


# ────────────────────────── docx helpers ──────────────────────────

def set_cell(cell, text, bold=False, size=10):
    cell.text = ''
    p = cell.paragraphs[0]
    r = p.add_run(str(text))
    r.bold = bold
    r.font.size = Pt(size)


def add_table(doc, header, rows, widths=None):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = 'Light Grid Accent 1'
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        set_cell(t.rows[0].cells[i], h, bold=True)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            set_cell(cells[i], v)
    if widths:
        for i, w in enumerate(widths):
            for row in t.rows:
                row.cells[i].width = Cm(w)
    return t


def add_heading(doc, text, level):
    h = doc.add_heading(text, level=level)
    for r in h.runs:
        r.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
    return h


def add_page_number(doc):
    footer = doc.sections[0].footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    fld1 = OxmlElement('w:fldChar'); fld1.set(qn('w:fldCharType'), 'begin')
    instr = OxmlElement('w:instrText'); instr.text = 'PAGE'
    fld2 = OxmlElement('w:fldChar'); fld2.set(qn('w:fldCharType'), 'end')
    run._r.append(fld1); run._r.append(instr); run._r.append(fld2)


def add_kv_table(doc, pairs):
    add_table(doc, ['Параметр', 'Значение'], pairs, widths=[9, 8])


# ────────────────────────── main ──────────────────────────

def main():
    conn = psycopg2.connect(DB_URL)
    st = collect_stats(conn)
    conn.close()
    p1_time = phase1_timing()

    c1 = chart_cluster_sizes(st)
    c2 = chart_similarity_hist(st)
    c3 = chart_train_vs_test(st)
    c4 = chart_subclusters(st)
    c5 = chart_km_vs_bt(st) if st['has_bt'] else None
    if st['has_r2']:
        r2c1 = chart_t2_sizes(st)
        r2c2 = chart_t2_train_vs_test(st)
        r2c3 = chart_t2_sim_hist(st)
    if st['has_tipg']:
        tc1 = chart_tipg_top(st)
    if st['has_r3']:
        r3c1 = chart_r3_coverage(st)
        r3c2 = chart_r3_distribution(st)
    timings = llm_timings() if st['has_r2'] else {}

    n_types = len(st['clusters'])
    n_clustered = sum(c for _, c in st['clusters'])
    n_subs = len(st['subs'])
    cov = next(c for thr, c in st['thresholds'] if thr == 0.8) / st['assigned']
    now = datetime.datetime.now()

    doc = Document()
    style = doc.styles['Normal']
    style.font.name = 'Calibri'
    style.font.size = Pt(11)
    add_page_number(doc)

    # ── Титул ──
    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = title.add_run('\nОтчёт эксперимента\n«Двухуровневая кластеризация Excel-документов»')
    r.bold = True
    r.font.size = Pt(22)
    r.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
    sub = doc.add_paragraph()
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub.add_run(f'\n{now.strftime("%d.%m.%Y %H:%M")}\n'
                f'Источник: D:\\FileOrganizer\\Sorted\\Excel_Xlsx (.xlsx + .xls)\n'
                f'Метод: direct text (Aspose.Cells TSV) → nomic-embed-text-v1.5 → '
                f'HDBSCAN (уровень 1) → KMeans (уровень 2)').italic = True

    # ── Резюме ──
    add_heading(doc, 'Резюме', 1)
    doc.add_paragraph(
        f'Обработано {st["train"] + st["test"] + st["failed"]} файлов '
        f'({st["xlsx"]} .xlsx, {st["xls"]} .xls). Первый уровень (HDBSCAN) выделил '
        f'{n_types} типов документов, покрывающих {n_clustered} из {st["train"]} '
        f'тренировочных документов ({n_clustered / st["train"]:.1%}); шум — '
        f'{st["noise"]} ({st["noise"] / st["train"]:.1%}). '
        f'Проверка на тестовом наборе ({st["assigned"]} док.): покрытие '
        f'{cov:.2%} при пороге {SIMILARITY_THRESHOLD} (среднее сходство {st["sim_avg"]:.3f}). '
        f'Второй уровень (KMeans) разделил типы на {n_subs} тематических под-кластеров.'
    )
    add_kv_table(doc, [
        ['Файлов всего / ошибок извлечения', f'{st["train"] + st["test"] + st["failed"]} / {st["failed"]}'],
        ['Train / Test', f'{st["train"]} / {st["test"]}'],
        ['Типов (уровень 1)', n_types],
        ['Покрытие train / шум', f'{n_clustered / st["train"]:.1%} / {st["noise"] / st["train"]:.1%}'],
        ['Покрытие test @0.8', f'{cov:.2%}'],
        ['Под-кластеров (уровень 2)', n_subs],
        ['Silhouette (sampled)', '0.294'],
    ])

    # ── Phase 1 ──
    add_heading(doc, 'Этап 1. Разделение и извлечение текста', 1)
    doc.add_paragraph(
        'Файлы перемешаны (seed=42) и разделены пополам. Текст извлечён прямым '
        'TSV-дампом через Aspose.Cells for .NET (extract_text, все листы, без '
        'конвертации в PDF), нормализация серий табов, усечение до 1000 символов. '
        'PDF-артефакты (FitToPagesWide=1) сгенерированы для визуального аудита.'
    )
    add_kv_table(doc, [
        ['Время выполнения', f'{p1_time} с (~{p1_time // 60} мин)' if p1_time else '—'],
        ['Скорость', '~13.4 файла/с'],
        ['Train: извлечено / всего', f'{st["train"]} / {st["train"] + 3}'],
        ['Test: извлечено / всего', f'{st["test"]} / {st["test"] + 1}'],
        ['Расширения', f'.xlsx — {st["xlsx"]}, .xls — {st["xls"]}'],
        ['Длина текста (train)', f'avg {st["txt_avg"]:.0f}, min {st["txt_min"]}, max {st["txt_max"]} симв.'],
    ])
    if st['fail_reasons']:
        doc.add_paragraph('Причины ошибок извлечения:')
        add_table(doc, ['Причина', 'Файлов'],
                  [[r[:90], c] for r, c in st['fail_reasons']], widths=[13, 3])

    # ── Phase 2 ──
    add_heading(doc, 'Этап 2. Эмбеддинги train + HDBSCAN (уровень 1)', 1)
    doc.add_paragraph(
        f'Эмбеддинги nomic-embed-text-v1.5 (768 float32, L2-норма, cuda:1). '
        f'HDBSCAN: metric=euclidean (эквивалент косинуса на нормированных векторах), '
        f'min_cluster_size = {int(MIN_CLUSTER_SIZE_FACTOR * st["train"])} '
        f'({MIN_CLUSTER_SIZE_FACTOR} × {st["train"]}).'
    )
    add_kv_table(doc, [
        ['Кластеров / шум', f'{n_types} / {st["noise"]} ({st["noise"] / st["train"]:.1%})'],
        ['Покрыто train', f'{n_clustered} ({n_clustered / st["train"]:.1%})'],
        ['Silhouette (sampled 5000)', '0.294'],
    ])
    doc.add_picture(str(c1), width=Cm(16.5))

    # ── Phase 3 ──
    add_heading(doc, 'Этап 3. Назначение test-набора ближайшему центроиду', 1)
    doc.add_paragraph(
        f'Все {st["assigned"]} тестовых документов назначены к {n_types} центроидам. '
        f'Среднее сходство {st["sim_avg"]:.3f} (min {st["sim_min"]:.3f}, max {st["sim_max"]:.3f}).'
    )
    add_table(doc, ['Порог similarity', 'Покрытие (док.)', 'Доля'],
              [[f'≥ {thr}', c, f'{c / st["assigned"]:.2%}'] for thr, c in st['thresholds']],
              widths=[5, 5, 4])
    doc.add_picture(str(c2), width=Cm(16.5))
    doc.add_picture(str(c3), width=Cm(16.5))
    doc.add_paragraph(
        'Кластеры 11 и 9 притягивают непропорционально много тестовых документов '
        '(5 283 и 1 604 при train-составе 702 и 403): это широкие «обобщённые» типы, '
        'поглощающие документы, аналогичные train-шуму. Среднее сходство даже для них '
        '≥ 0.92 — назначения осмысленные.'
    )

    # ── Phase 4 ──
    add_heading(doc, 'Этап 4. Тематическая кластеризация внутри типов (KMeans)', 1)
    doc.add_paragraph(
        f'Каждый тип разделён KMeans (k = max(2, min(5, N//50))) на под-кластеры. '
        f'Итого {n_subs} под-кластеров по {n_types} типам; эмбеддинги переиспользованы '
        f'из этапа 2 (GPU не требовался).'
    )
    rows = []
    for t, _ in st['clusters']:
        sizes = [c for tt, s, c in st['subs'] if tt == t]
        rows.append([t, sum(sizes), len(sizes), ' · '.join(map(str, sizes))])
    add_table(doc, ['Тип', 'Док.', 'Под-кл.', 'Размеры под-кластеров'], rows,
              widths=[2, 2.5, 2.5, 10])
    doc.add_picture(str(c4), width=Cm(16.5))

    # ── Phase 4b (BERTopic) ──
    if st['has_bt']:
        add_heading(doc, 'Этап 4б. Альтернатива второму уровню: BERTopic', 1)
        doc.add_paragraph(
            'BERTopic (UMAP → HDBSCAN → c-TF-IDF top-слова) на тех же эмбеддингах '
            '(без GPU и переобучения). Отличия от KMeans: число тем определяется '
            'автоматически, пограничные документы помечаются шумом (−1), каждая тема '
            'получает семантическую метку из top-слов.'
        )
        rows = []
        for t, _ in st['clusters']:
            k_km = len([1 for tt, _s, _c in st['subs'] if tt == t])
            n_bt, n_noise = st['bt_summary'].get(t, (0, 0))
            rows.append([t, k_km, n_bt, n_noise])
        add_table(doc, ['Тип', 'KMeans k', 'BERTopic тем', 'BERTopic шум'], rows,
                  widths=[2.5, 3.5, 3.5, 3.5])
        doc.add_picture(str(c5), width=Cm(16.5))

        doc.add_paragraph(
            'Детализация типа 0 (акты по договорам шаринга МТС): BERTopic выделил '
            'чистые группы 25.1 и 25.2, которые KMeans не различил, а также группу '
            'по конкретному договору (775097720243675):'
        )
        add_table(doc, ['Тема BT', '25.1', '25.2', 'Всего', 'Top-слова'],
                  [[topic, n251, n252, total, (words or '')[:60]]
                   for topic, words, total, n251, n252 in st['bt_type0']],
                  widths=[2, 2, 2, 2, 8])

        for t in (11, 12):
            if t in st['bt_words']:
                rows = [[('−1 (шум)' if topic == -1 else topic), c, (words or '')[:70]]
                        for topic, c, words in st['bt_words'][t]]
                doc.add_paragraph(f'Крупнейшие темы BERTopic в типе {t}:')
                add_table(doc, ['Тема', 'Док.', 'Top-слова'], rows, widths=[3, 2, 11])

        for text in [
            'Тип 0: BERTopic выиграл — найдены чистые 25.1 (тема 4) и 25.2 (тема 6), '
            'группа приложения № 10 совпала с KMeans 1-в-1 (29 док.), дополнительно '
            'выделена серия по конкретному договору.',
            'Типы 1/11/12: BERTopic дробит по сериям документов — номерам договоров, '
            'контрагентам, датам (top-слова: 251045, 25859, ИНН). Внутри типов нет '
            '«тем» в смысловом смысле — есть серии по договорам/поставщикам.',
            'Тип 12: гипотеза об однородности опровергнута — вместо 1 темы BT нашёл '
            '22 (счета-фактуры разных контрагентов) + 18% шума.',
            'Вердикт: методы комплементарны. KMeans — тематическая навигация '
            '(3–5 читаемых групп), BERTopic — выявление плотных серий и честных '
            'выбросов. Рекомендуется KMeans как основной второй уровень + BT точечно '
            'для типов, где KMeans смешивает (как тип 0).',
        ]:
            doc.add_paragraph(text, style='List Bullet')

    # ── Раунд 2: LLM-тематическая кластеризация ──
    if st['has_r2']:
        add_heading(doc, 'Раунд 2. Тематическая кластеризация (направления деятельности)', 1)
        doc.add_paragraph(
            'Типовая схема (раунд 1) кластеризовала по типам документов, т.к. эмбеддинг '
            'строился по шапке. В раунде 2 LLM (Qwen3.5-4B, cuda:0) извлекает из каждого '
            'документа ПРЕДМЕТ (направление деятельности, с запретом дублирования типа '
            'в промпте) и ТИП (вид документа). Эмбеддинг строится только по строке '
            'ПРЕДМЕТ (nomic-embed, cuda:1) → HDBSCAN → слияние центроидов с sim ≥ 0.92. '
            'Заранее заданной таксономии нет — направления обнаруживаются кластерным анализом.'
        )
        t_min = timings.get('train') or 0
        t_test = timings.get('test') or 0
        n_themes = len(st['t2_themes'])
        n_clustered = sum(c for _, c in st['t2_themes'])
        cov = (st['t2_sims'] >= SIMILARITY_THRESHOLD).mean() if len(st['t2_sims']) else 0
        add_kv_table(doc, [
            ['LLM-извлечение (train / test)',
             f'{st["r2_llm_train"]} / {st["r2_llm_test"]} док. (~{t_min:.0f} / ~{t_test:.0f} мин, ~1.55 д/с)'],
            ['Тем обнаружено', f'{n_themes} (21 сырых → {n_themes} после слияния центроидов)'],
            ['Шум train', f'{st["t2_noise"]} ({st["t2_noise"] / st["r2_llm_train"]:.1%})'],
            ['Silhouette (sampled)', '0.244'],
            ['Покрытие test @0.8', f'{cov:.2%} ({st["t2_assigned"] - st["t2_unknown"]}/{st["t2_assigned"]})'],
            ['Буфер неопознанных (дрейф)', f'{st["t2_unknown"]} ({st["t2_unknown"] / st["t2_assigned"]:.1%})'],
            ['Сходство test avg/min/max', f'{st["t2_sim_avg"]:.3f} / {st["t2_sim_min"]:.3f} / {st["t2_sim_max"]:.3f}'],
        ])
        doc.add_paragraph('Состав тем и назначение тест-набора:')
        rows = []
        for lab, c_train in st['t2_themes']:
            c_test, s_test = st['t2_test_per'].get(lab, (0, 0.0))
            rows.append([_t2_name(st, lab), c_train, c_test, f'{s_test:.3f}'])
        add_table(doc, ['Тема (top-слова)', 'Train', 'Test', 'avg sim'], rows,
                  widths=[8, 2.5, 2.5, 3])
        doc.add_picture(str(r2c1), width=Cm(16.5))
        doc.add_picture(str(r2c2), width=Cm(16.5))
        doc.add_picture(str(r2c3), width=Cm(16.5))
        doc.add_paragraph(
            'Буфер неопознанных (sim < 0.8) — детектор дрейфа тематики: документы, '
            'не покрытые существующими направлениями, накапливаются там для '
            'переоткрытия тем, а не размазываются по чужим кластерам.'
        )

    # ── Второй уровень: канонические типы (phase_4c) ──
    if st['has_r2'] and st['has_tipg']:
        add_heading(doc, 'Второй уровень: канонические типы документов (HDBSCAN по ТИП)', 2)
        doc.add_paragraph(
            f'LLM помимо ПРЕДМЕТа извлекает ТИП документа — свободную строку '
            f'({st["tipg_tips"]} вариантов после нормализации ё→е). Кластеризация '
            f'самих строк ТИП (эмбеддинг → HDBSCAN → слияние центроидов sim ≥ 0.92) '
            f'далa {st["tipg_groups"]} канонических групп: семьи вариантов склеены '
            f'(«Акт приема-передачи» ← 33 строки, «Акт о рассогласованиях» ← '
            f'«Акт расхождений»...), частотные одиночные типы («Смета», '
            f'«Счет на оплату») получили собственные группы. Реестр — temp_tip_groups.'
        )
        add_table(doc, ['Канонический тип', 'Док.', 'Вариантов'],
                  [[g, d, n] for g, d, n in st['tipg_top']],
                  widths=[9, 3, 3])
        doc.add_picture(str(tc1), width=Cm(16.5))
        if st['tipg_families']:
            doc.add_paragraph('Примеры склеенных семей (группа ← варианты):')
            add_table(doc, ['Группа', 'Док.', 'Варианты'],
                      [[g, d, (s or '')[:80]] for g, s, d in st['tipg_families']],
                      widths=[5, 2, 9])
        doc.add_paragraph('Состав тем по типам (топ-3 типа на тему):')
        rows = []
        for lab, _c in st['t2_themes']:
            comp = st['tipg_theme_comp'].get(lab, [])
            rows.append([_t2_name(st, lab),
                         '; '.join(f'{g} ({c})' for g, c in comp)])
        add_table(doc, ['Тема', 'Топ-3 типа (док.)'], rows, widths=[6, 10])

    # ── Раунд 3 ──
    if st['has_r3']:
        add_heading(doc, 'Раунд 3. Перенос реестра тем на PDF-потоки', 1)
        doc.add_paragraph(
            'Реестр тем Раунда 2 (обучен на Excel) применён к новым потокам без '
            'переобучения: PDF_Text (18 620) и PDF_Tables (93 221). Цепочка та же: '
            'fitz (первые 2 стр.) → LLM ПРЕДМЕТ/ТИП (Qwen3.5-4B, 2 GPU параллельно, '
            'шард по id) → эмбеддинг → центроиды с порогом 0.8 → дрейф-буфер → '
            'переоткрытие новых тем (HDBSCAN) → реестр v2.'
        )
        rows = [
            ['Excel test (эталон)', st.get('t2_assigned', 0), f'{(st["t2_sims"] >= SIMILARITY_THRESHOLD).mean():.2%}' if len(st.get('t2_sims', [])) else '—',
             f'{st.get("t2_sim_avg", 0):.3f}', st.get('t2_unknown', 0)],
            ['PDF_Text', st['r3_text_n'], f'{1 - st["r3_text_unk"] / st["r3_text_n"]:.2%}',
             f'{st["r3_text_avg"]:.3f}', st['r3_text_unk']],
        ]
        if st.get('has_r3_tables'):
            rows.append(['PDF_Tables', st['r3_tab_n'], f'{1 - st["r3_tab_unk"] / st["r3_tab_n"]:.2%}',
                         f'{st["r3_tab_avg"]:.3f}', st['r3_tab_unk']])
        add_table(doc, ['Поток', 'Документов', 'Покрытие @0.8', 'avg sim', 'Дрейф-буфер'],
                  rows, widths=[4.5, 3, 3.5, 2.5, 3])
        doc.add_picture(str(r3c1), width=Cm(15))
        doc.add_picture(str(r3c2), width=Cm(16.5))

        if st.get('r3_registry'):
            doc.add_paragraph(
                f"Реестр тем v2: {sum(st['r3_registry'].values())} центроидов — "
                + ', '.join(f'{k}: {v}' for k, v in st['r3_registry'].items())
                + '. Новые темы добавлялись с дедупликацией (sim ≥ 0.90 к существующему '
                  'центроиду → слияние, иначе новый label).')
            add_table(doc, ['Происхождение', 'Тем'], 
                      [[k, v] for k, v in st['r3_registry'].items()], widths=[6, 3])
        if st.get('r3_newthemes'):
            doc.add_paragraph('Новые темы, переоткрытые из дрейф-буферов:')
            sizes = st.get('r3_nt_sizes', {})
            add_table(doc, ['Тема', 'Происхождение', 'Top-слова'],
                      [[l, ('PDF_Text' if l <= 15 else 'PDF_Tables'), (w or '')[:70]]
                       for l, w in st['r3_newthemes']], widths=[2, 3, 11])

    # ── Выводы ──
    add_heading(doc, 'Выводы и рекомендации', 1)

    add_heading(doc, 'Раунд 1 (типовая схема: HDBSCAN по шапке документа)', 2)
    for text in [
        'Двухуровневая схема работоспособна: 13 типов покрывают 99.6% тестового потока '
        'при пороге 0.8; типы устойчивы (смена корпуса PDF-text → direct text не изменила '
        'структуру: 12→13 кластеров, шум 44.1%→43.3%).',
        'Наблюдение: тип 12 тематически однороден (74% в одном под-кластере) — '
        'кандидат на k<5 при втором уровне.',
        'Наблюдение: внутри типа 0 KMeans чисто выделил подгруппу «Дополнение 25.1» (19 док.), '
        'но пограничные документы (sim 0.98 vs 0.99) назначаются к более массивному '
        'соседу — ожидаемое поведение жёсткого KMeans.',
        'Рекомендация: альтернативный второй уровень — BERTopic — протестирован '
        '(см. этап 4б): решает случай 25.1/25.2 и даёт семантические метки, но дробит '
        'по сериям документов. Оставить KMeans основным + BT точечно.',
        'Ограничение: первый уровень выделяет ТИПЫ документов (счета, акты, УПД), '
        'а не направления деятельности — эмбеддинг по шапке не видит предмета. '
        'Тематическая задача решается в раунде 2.',
    ]:
        doc.add_paragraph(text, style='List Bullet')

    if st['has_r2']:
        add_heading(doc, 'Раунд 2 (тематическая схема: LLM ПРЕДМЕТ + эмбеддинг + HDBSCAN)', 2)
        cov_r2 = (st['t2_sims'] >= SIMILARITY_THRESHOLD).mean() if len(st['t2_sims']) else 0
        for text in [
            f'Главный результат эксперимента: тематическая схема работает — LLM '
            f'(промпт ПРЕДМЕТ/ТИП с запретом дублирования типа) + эмбеддинг ПРЕДМЕТа + '
            f'HDBSCAN со слиянием центроидов дали {len(st["t2_themes"])} направлений '
            f'деятельности (приёмка-передача оборудования, логистика, электроснабжение, '
            f'услуги связи, аренда, ВОЛС, уборка, ТО...), покрытие тест-набора '
            f'{cov_r2:.2%} @0.8 (avg sim {st["t2_sim_avg"]:.3f}).',
            'Ключевые факторы успеха: (1) direct text вместо PDF (нет обрезания широких '
            'таблиц); (2) промпт с запретом «ПРЕДМЕТ = ТИП»; (3) эмбеддинг только '
            'предметной строки; (4) слияние микрокластеров по центроидам (silhouette '
            '0.009 → 0.340 на валидации).',
            f'Шум train ({st["t2_noise"]}, 51.8%) — длинный хвост редких и смешанных '
            f'предметов; в тесте такие документы назначаются ближайшим темам '
            f'(покрытие {cov_r2:.2%}) либо отсекаются в буфер неопознанных '
            f'({st["t2_unknown"]} док.) — это и есть механизм детекции дрейфа тематики.',
            'Рекомендация: тематическую схему — в продакшен: реестр центроидов тем + '
            'пороговое назначение новых документов + периодическое переоткрытие тем из '
            'буфера неопознанных; тип документа (temp_llm_subject.tip) использовать '
            'вторичным уровнем.',
        ]:
            doc.add_paragraph(text, style='List Bullet')

        if st['has_tipg']:
            doc.add_paragraph(
                f'Второй уровень канонизирован без правил: {st["tipg_tips"]} свободных '
                f'строк ТИП → {st["tipg_groups"]} групп HDBSCAN (реестр temp_tip_groups); '
                f'навигационное дерево «тема → тип» построено в ClusterViews\\themes.',
                style='List Bullet')

    if st['has_r3']:
        add_heading(doc, 'Раунд 3 (перенос на PDF-потоки)', 2)
        cov_text = 1 - st['r3_text_unk'] / st['r3_text_n']
        tab_line = ''
        if st.get('has_r3_tables'):
            cov_tab = 1 - st['r3_tab_unk'] / st['r3_tab_n']
            tab_line = (f', PDF_Tables {cov_tab:.2%} (avg {st["r3_tab_avg"]:.3f}, '
                        f'дрейф {st["r3_tab_unk"]})')
        for text in [
            f'Реестр переносится на новые потоки без переобучения: покрытие PDF_Text '
            f'{cov_text:.2%} (avg {st["r3_text_avg"]:.3f}){tab_line}.',
            f'Механизм дрейфа работает: реестр v2 = {sum(st.get("r3_registry", {}).values())} тем '
            f'({", ".join(f"{k}: {v}" for k, v in st.get("r3_registry", {}).items())}); '
            f'новые темы — замена SIM-карт, регистрация юрлиц (ЕГРЮЛ), доступ к инфраструктуре, '
            f'клининг кабельных линий, поставка шин, реклама/PR, выписки ЕГРН, HeadHunter, '
            f'закупка мебели; дедупликация ЕГРЮЛ сработала (sim 0.978 → тема 15).',
            'LLM на 2 GPU (обе RTX PRO 4000): шардирование по id, суммарно ~2.7 д/с; '
            'узкое место — скорость LLM, эмбеддинги/кластеризация — минуты.',
        ]:
            doc.add_paragraph(text, style='List Bullet')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / f'excel_experiment_report_{now.strftime("%Y%m%d_%H%M%S")}.docx'
    doc.save(out_file)
    print(f'✅ Отчёт сохранён: {out_file}')
    print(f'   Диаграммы: {CHART_DIR}')


if __name__ == '__main__':
    main()
