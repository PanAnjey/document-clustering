# show_themes.py
# Визуальный просмотр ТЕМАТИЧЕСКИХ кластеров (Раунд 2, temp_theme2_*).
#
# Для выбранной темы создаётся папка с жёсткими ссылками на PDF-артефакты
# Phase 1 (Extracted\Excel_Xlsx\{stem}.pdf) + _top_words.txt с top-словами
# темы, и открывается pdf_viewer.py.
#
# Запуск:
#   python show_themes.py                # список тем с размерами и top-словами
#   python show_themes.py 8              # тема 8 + viewer
#   python show_themes.py -1             # шумовой кластер (не вошло в темы)
#   python show_themes.py 9 --sample 50  # случайные 50 файлов темы
#   python show_themes.py --all          # плоские папки для всех тем (+ шум)
#   python show_themes.py --layout       # ДЕРЕВО: тема → типы документов
#   python show_themes.py 8 --no-launch  # только папка, без viewer

import argparse
import re
import shutil
import sys

import psycopg2

from experiment_config import DB_URL  # noqa: sys.path+utf8
from show_cluster import build_view_folder, launch_viewer, VIEW_ROOT, _report

THEMES_ROOT = VIEW_ROOT / 'themes'
MAX_TIP_FOLDERS = 12  # топ типов на тему отдельными папками, хвост → _прочее


def fetch_theme_files(conn, label, sample=0):
    """file_path'ы train-набора для темы (label=-1 → шум)."""
    order = "ORDER BY random() LIMIT %s" if sample else "ORDER BY t.file_path"
    params = (label, sample) if sample else (label,)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT t.file_path
            FROM temp_excel_train t
            JOIN temp_theme2_clusters c ON c.train_id = t.id
            WHERE c.theme_label = %s
            {order}
            """,
            params,
        )
        return [r[0] for r in cur.fetchall()]


def fetch_theme_files_by_tip(conn, label):
    """{канонический_тип: [file_path, ...]} для темы (второй уровень).
    Канонические имена — из temp_tip_groups (HDBSCAN-группы, phase_4c);
    при отсутствии таблицы — сырая нормализация."""
    tip_group_map = load_tip_group_map(conn)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT s.tip, t.file_path
            FROM temp_theme2_clusters c
            JOIN temp_excel_train t ON t.id = c.train_id
            LEFT JOIN temp_llm_subject s ON s.train_id = c.train_id
            WHERE c.theme_label = %s
        """, (label,))
        rows = cur.fetchall()
    by_tip = {}
    for tip, fp in rows:
        norm = normalize_tip(tip)
        canonical = tip_group_map.get(norm, norm)
        by_tip.setdefault(canonical, []).append(fp)
    return by_tip


def load_tip_group_map(conn):
    """{нормализованный tip: каноническое group_name} из temp_tip_groups."""
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.temp_tip_groups')")
        if cur.fetchone()[0] is None:
            return {}
        cur.execute("SELECT tip, group_name FROM temp_tip_groups")
        return {t: g for t, g in cur.fetchall()}


def normalize_tip(tip):
    """Нормализация строки ТИП от LLM: ё→е, схлопывание пробелов, капитализация."""
    t = (tip or '').strip().lower().replace('ё', 'е')
    t = re.sub(r'\s+', ' ', t)
    return (t[:1].upper() + t[1:]) if t else 'Без типа'


def sanitize_dirname(name, maxlen=48):
    n = re.sub(r'[\\/:*?"<>|]', ' ', name)
    n = re.sub(r'\s+', ' ', n).strip()
    return n[:maxlen] or 'x'


def fetch_theme_files(conn, label, sample=0):
    """file_path'ы train-набора для темы (label=-1 → шум)."""
    order = "ORDER BY random() LIMIT %s" if sample else "ORDER BY t.file_path"
    params = (label, sample) if sample else (label,)
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT t.file_path
            FROM temp_excel_train t
            JOIN temp_theme2_clusters c ON c.train_id = t.id
            WHERE c.theme_label = %s
            {order}
            """,
            params,
        )
        return [r[0] for r in cur.fetchall()]


def theme_list(conn):
    """(label, count, top_words) — темы по убыванию размера + шум в конце."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT c.theme_label, COUNT(*) c, MAX(w.top_words)
            FROM temp_theme2_clusters c
            LEFT JOIN temp_theme2_topwords w ON w.theme_label = c.theme_label
            GROUP BY c.theme_label
            ORDER BY (c.theme_label = -1), c DESC
        """)
        return cur.fetchall()


def write_top_words(out_dir, label, words):
    """Самодокументирование папки темы (viewer читает только *.pdf)."""
    try:
        (out_dir / '_top_words.txt').write_text(
            f'Тема {label}\nTop-слова: {words or ""}\n', encoding='utf-8')
    except Exception:
        pass


def build_theme_layout(conn, label, words):
    """Дерево theme_NN/<тип>/*.pdf: второй уровень по ТИПу документа от LLM."""
    by_tip = fetch_theme_files_by_tip(conn, label)
    if not by_tip:
        return None
    theme_dir = THEMES_ROOT / ('theme_noise' if label == -1 else f'theme_{label:02d}')
    # Полная перестройка папки темы (старые плоские pdf и подпапки удаляются;
    # hardlink-удаление не трогает оригиналы в Extracted)
    if theme_dir.exists():
        shutil.rmtree(theme_dir, ignore_errors=True)
    theme_dir.mkdir(parents=True, exist_ok=True)
    write_top_words(theme_dir, label, words)

    # Топ-N типов отдельными папками, хвост → _прочее
    tips_sorted = sorted(by_tip.items(), key=lambda kv: -len(kv[1]))
    top_tips = tips_sorted[:MAX_TIP_FOLDERS]
    rest = [fp for _tip, fps in tips_sorted[MAX_TIP_FOLDERS:] for fp in fps]
    if rest:
        top_tips.append(('_прочее', rest))

    total = 0
    for tip, fps in top_tips:
        sub_dir = theme_dir / sanitize_dirname(tip)
        out_dir, linked, copied, missing = build_view_folder(
            f'themes/{theme_dir.name}/{sub_dir.name}', fps)
        total += linked + copied
        print(f'    {tip}: {len(fps)} → {sub_dir.name}/')
    print(f'✅ theme {label}: {total} ссылок → {theme_dir} '
          f'({len(top_tips)} папок типов)')
    return theme_dir


def main():
    ap = argparse.ArgumentParser(description='Визуальный просмотр тематических кластеров (Раунд 2)')
    ap.add_argument('label', type=int, nargs='?', default=None,
                    help='номер темы (-1 = шум); без аргумента — список тем')
    ap.add_argument('--all', action='store_true',
                    help='плоские папки всех тем (+ шум), viewer не открывать')
    ap.add_argument('--layout', action='store_true',
                    help='дерево тема → папки типов документов (второй уровень)')
    ap.add_argument('--sample', type=int, default=0,
                    help='случайные N файлов вместо всей темы')
    ap.add_argument('--no-launch', action='store_true',
                    help='только создать папку, не открывать pdf_viewer')
    args = ap.parse_args()

    conn = psycopg2.connect(DB_URL)
    rows = theme_list(conn)
    if not rows:
        conn.close()
        sys.exit('❌ temp_theme2_clusters пуст — сначала phase_2d')

    if args.label is None and not args.all and not args.layout:
        print('Темы Раунда 2 (номер: документов | top-слова):')
        for lab, cnt, words in rows:
            tag = ' (шум)' if lab == -1 else ''
            print(f'  {lab:>2}: {cnt:>5}{tag} | {(words or "")[:75]}')
        print('\nПримеры: python show_themes.py 8        — тема 8')
        print('         python show_themes.py --all   — все папки тем (плоско)')
        print('         python show_themes.py --layout — дерево тема → типы документов')
        conn.close()
        return

    words_map = {lab: words for lab, _c, words in rows}

    # ── Дерево тема → типы документов ────────────────────────────
    if args.layout:
        for lab, _cnt, words in rows:
            build_theme_layout(conn, lab, words)
        conn.close()
        print(f'\nДерево готово: {THEMES_ROOT}')
        return

    labels = [lab for lab, _c, _w in rows] if args.all else [args.label]
    launch_folder = None
    suffix = f"_sample{args.sample}" if args.sample else ""

    for lab in labels:
        files = fetch_theme_files(conn, lab, args.sample)
        if not files:
            print(f'⚠️ Тема {lab}: нет файлов')
            continue
        name = f'theme_noise{suffix}' if lab == -1 else f'theme_{lab:02d}{suffix}'
        out_dir, linked, copied, missing = build_view_folder(
            f'themes/{name}', files)
        write_top_words(out_dir, lab, words_map.get(lab))
        _report(f'theme {lab}', files, out_dir, linked, copied, missing)
        if launch_folder is None:
            launch_folder = out_dir

    conn.close()

    if not args.all and not args.no_launch and launch_folder:
        launch_viewer(launch_folder)
        print(f'🖥️  pdf_viewer открыт: {launch_folder}')


if __name__ == '__main__':
    main()
