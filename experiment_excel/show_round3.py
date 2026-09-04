# show_round3.py
# Визуальный просмотр результатов Раунда 3 (PDF_Text + PDF_Tables).
#
# Источники — сами PDF (Sorted\PDF_Text, Sorted\PDF_Tables), hardlink'и
# напрямую на исходные файлы (без PDF-артефактов Extracted).
#
# Структура:
#   ClusterViews\round3\pdf_text\theme_NN\       — назначение по темам Р2
#   ClusterViews\round3\pdf_text\unknown\        — дрейф-буфер (234)
#   ClusterViews\round3\pdf_tables\theme_NN\     — назначение по темам Р2
#   ClusterViews\round3\pdf_tables\unknown\      — дрейф-буфер (2467)
#   ClusterViews\round3\pdf_tables\new_theme_NN\ — переоткрытые темы
#
# Запуск:
#   python show_round3.py                          # сводка по потокам
#   python show_round3.py --stream pdf_text --theme 9      # + viewer
#   python show_round3.py --stream pdf_tables --new-theme 0  # SIM-карты
#   python show_round3.py --all [--sample N]       # все папки

import argparse
import os
import re
import sys
from pathlib import Path

import psycopg2

from experiment_config import DB_URL  # noqa: sys.path+utf8
from show_cluster import launch_viewer, VIEW_ROOT, _report
from show_themes import normalize_tip, sanitize_dirname, load_tip_group_map

R3_ROOT = VIEW_ROOT / 'round3'
MAX_TYPE_FOLDERS = 15  # топ типов на поток, хвост → _прочее

STREAMS = {
    'pdf_text': {
        'files': 'temp_pdftext_files',
        'assign': 'temp_pdftext_assign',
        'newthemes': None,
        'newthemes_top': None,
    },
    'pdf_tables': {
        'files': 'temp_pdftables_files',
        'assign': 'temp_pdftables_assign',
        'newthemes': 'temp_pdftables_newthemes',
        'newthemes_top': 'temp_pdftables_newthemes_top',
    },
}


def build_links_direct(out_dir: Path, file_paths):
    """Hardlink'и напрямую на исходные PDF; fallback — копия."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob('*.pdf'):
        try:
            old.unlink()
        except Exception:
            pass
    linked, copied, missing = 0, 0, 0
    seen = set()
    for fp in file_paths:
        src = Path(fp)
        if src.name in seen:
            continue
        seen.add(src.name)
        if not src.exists():
            missing += 1
            continue
        dst = out_dir / src.name
        try:
            os.link(src, dst)
            linked += 1
        except OSError:
            try:
                import shutil
                shutil.copy2(src, dst)
                copied += 1
            except Exception:
                missing += 1
    return out_dir, linked, copied, missing


def fetch_theme_files(conn, stream_cfg, theme, sample=0, unknown=False):
    files_t, assign_t = stream_cfg['files'], stream_cfg['assign']
    cond = "a.is_unknown" if unknown else "a.assigned_theme = %s AND NOT a.is_unknown"
    order = "ORDER BY random() LIMIT %s" if sample else "ORDER BY f.file_path"
    params = (sample,) if (sample and unknown) else \
        ((theme, sample) if sample else (() if unknown else (theme,)))
    q = f"""
        SELECT f.file_path
        FROM {files_t} f
        JOIN {assign_t} a ON a.pdftext_id = f.id
        WHERE {cond} {order}
    """
    with conn.cursor() as cur:
        cur.execute(q, params)
        return [r[0] for r in cur.fetchall()]


def fetch_newtheme_files(conn, stream_cfg, new_theme, sample=0):
    files_t, nt_t = stream_cfg['files'], stream_cfg['newthemes']
    order = "ORDER BY random() LIMIT %s" if sample else "ORDER BY f.file_path"
    params = (new_theme, sample) if sample else (new_theme,)
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT f.file_path
            FROM {files_t} f
            JOIN {nt_t} n ON n.pdftext_id = f.id
            WHERE n.new_theme = %s {order}
        """, params)
        return [r[0] for r in cur.fetchall()]


def summary(conn):
    for name, cfg in STREAMS.items():
        with conn.cursor() as cur:
            cur.execute(f"""
                SELECT a.assigned_theme, COUNT(*) c, AVG(a.similarity) s,
                       COUNT(*) FILTER (WHERE a.is_unknown)
                FROM {cfg['assign']} a GROUP BY a.assigned_theme
                ORDER BY c DESC
            """)
            rows = cur.fetchall()
        total = sum(r[1] for r in rows)
        unk = max((r[3] for r in rows), default=0)
        print(f'--- {name}: {total} док., дрейф {unk} ({unk / total:.1%})')
        for lab, c, s, _u in rows:
            print(f'    тема {lab:>2}: {c:>6} avg={s:.3f}')
        if cfg['newthemes']:
            cur.execute(f"""
                SELECT n.new_theme, COUNT(*) c, MAX(w.top_words)
                FROM {cfg['newthemes']} n
                LEFT JOIN {cfg['newthemes_top']} w ON w.new_theme = n.new_theme
                GROUP BY n.new_theme ORDER BY c DESC
            """)
            print(f'    новые темы:')
            for lab, c, w in cur.fetchall():
                print(f'    new_theme {lab:>2}: {c:>5} | {(w or "")[:60]}')


def write_words(out_dir, text):
    try:
        (out_dir / '_top_words.txt').write_text(text, encoding='utf-8')
    except Exception:
        pass


def theme_words(conn, label):
    with conn.cursor() as cur:
        cur.execute("SELECT top_words FROM temp_theme2_topwords WHERE theme_label = %s", (label,))
        r = cur.fetchone()
        return f'Тема {label}\nTop-слова: {r[0] if r else ""}\n'


def load_tip_group_map_v2(conn):
    """{нормализованный tip: каноническое group_name} из temp_tip_groups_v2."""
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.temp_tip_groups_v2')")
        if cur.fetchone()[0] is None:
            return {}
        cur.execute("SELECT tip, group_name FROM temp_tip_groups_v2")
        return {t: g for t, g in cur.fetchall()}


def build_type_layout(conn, stream_name, cfg, sample=0):
    """by_type/<поток>/<канонический тип>/*.pdf — визуальный анализ по типам."""
    subjects_t = {'pdf_text': 'temp_llm_subject_pdftext',
                  'pdf_tables': 'temp_llm_subject_pdftables'}[stream_name]
    gmap = load_tip_group_map_v2(conn)
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT f.file_path, s.tip
            FROM {cfg['files']} f
            JOIN {subjects_t} s ON s.train_id = f.id
        """)
        rows = cur.fetchall()
    by_type = {}
    for fp, tip in rows:
        norm = normalize_tip(tip)
        by_type.setdefault(gmap.get(norm, norm), []).append(fp)

    stream_dir = R3_ROOT / 'by_type' / stream_name
    if stream_dir.exists():
        import shutil
        shutil.rmtree(stream_dir, ignore_errors=True)
    tips_sorted = sorted(by_type.items(), key=lambda kv: -len(kv[1]))
    top = tips_sorted[:MAX_TYPE_FOLDERS]
    rest = [fp for _t, fps in tips_sorted[MAX_TYPE_FOLDERS:] for fp in fps]
    if rest:
        top.append(('_прочее', rest))
    print(f'=== by_type/{stream_name}: {len(by_type)} канонических типов, '
          f'топ-{MAX_TYPE_FOLDERS} папками')
    for tip, fps in top:
        if sample and len(fps) > sample:
            import random
            fps = random.sample(fps, sample)
        out_dir, linked, copied, missing = build_links_direct(
            stream_dir / sanitize_dirname(tip), fps)
        _report(f'  {tip}', fps, out_dir, linked, copied, missing)


def main():
    ap = argparse.ArgumentParser(description='Визуальный просмотр Раунда 3')
    ap.add_argument('--stream', choices=['pdf_text', 'pdf_tables'])
    ap.add_argument('--theme', type=int, default=None)
    ap.add_argument('--new-theme', type=int, default=None)
    ap.add_argument('--unknown', action='store_true')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--layout-types', action='store_true',
                    help='дерево by_type\\поток\\канонический тип (temp_tip_groups_v2)')
    ap.add_argument('--sample', type=int, default=0)
    ap.add_argument('--no-launch', action='store_true')
    args = ap.parse_args()

    conn = psycopg2.connect(DB_URL)

    if args.layout_types:
        for name, cfg in STREAMS.items():
            build_type_layout(conn, name, cfg, sample=args.sample)
        conn.close()
        print(f'\nДерево по типам готово: {R3_ROOT / "by_type"}')
        return

    if args.all:
        for name, cfg in STREAMS.items():
            print(f'=== {name}')
            with conn.cursor() as cur:
                cur.execute(f"""
                    SELECT assigned_theme, COUNT(*) c FROM {cfg['assign']}
                    WHERE NOT is_unknown GROUP BY assigned_theme ORDER BY c DESC
                """)
                themes = [r[0] for r in cur.fetchall()]
            for lab in themes:
                files = fetch_theme_files(conn, cfg, lab, sample=args.sample)
                out_dir, linked, copied, missing = build_links_direct(
                    R3_ROOT / name / f'theme_{lab:02d}', files)
                write_words(out_dir, theme_words(conn, lab))
                _report(f'{name} theme {lab}', files, out_dir, linked, copied, missing)
            # unknown
            files = fetch_theme_files(conn, cfg, None, sample=args.sample, unknown=True)
            out_dir, linked, copied, missing = build_links_direct(
                R3_ROOT / name / 'unknown', files)
            write_words(out_dir, f'{name}: дрейф-буфер (sim < 0.8)\n')
            _report(f'{name} unknown', files, out_dir, linked, copied, missing)
            # новые темы
            if cfg['newthemes']:
                with conn.cursor() as cur:
                    cur.execute(f"""
                        SELECT n.new_theme, COUNT(*) c, MAX(w.top_words)
                        FROM {cfg['newthemes']} n
                        LEFT JOIN {cfg['newthemes_top']} w ON w.new_theme = n.new_theme
                        GROUP BY n.new_theme ORDER BY c DESC
                    """)
                    nts = cur.fetchall()
                for lab, c, w in nts:
                    files = fetch_newtheme_files(conn, cfg, lab, sample=args.sample)
                    out_dir, linked, copied, missing = build_links_direct(
                        R3_ROOT / name / f'new_theme_{lab:02d}', files)
                    write_words(out_dir, f'НОВАЯ тема {lab} (дрейф PDF_Tables)\nTop-слова: {w or ""}\n')
                    _report(f'{name} NEW {lab}', files, out_dir, linked, copied, missing)
        conn.close()
        print(f'\nДерево готово: {R3_ROOT}')
        return

    if not args.stream:
        summary(conn)
        conn.close()
        print('\nПримеры: python show_round3.py --stream pdf_tables --new-theme 0')
        print('         python show_round3.py --stream pdf_text --theme 9')
        print('         python show_round3.py --all')
        return

    cfg = STREAMS[args.stream]
    launch_folder = None
    if args.new_theme is not None and cfg['newthemes']:
        files = fetch_newtheme_files(conn, cfg, args.new_theme, sample=args.sample)
        out_dir, linked, copied, missing = build_links_direct(
            R3_ROOT / args.stream / f'new_theme_{args.new_theme:02d}', files)
        _report(f'NEW {args.new_theme}', files, out_dir, linked, copied, missing)
        launch_folder = out_dir
    elif args.unknown:
        files = fetch_theme_files(conn, cfg, None, sample=args.sample, unknown=True)
        out_dir, linked, copied, missing = build_links_direct(
            R3_ROOT / args.stream / 'unknown', files)
        _report('unknown', files, out_dir, linked, copied, missing)
        launch_folder = out_dir
    elif args.theme is not None:
        files = fetch_theme_files(conn, cfg, args.theme, sample=args.sample)
        out_dir, linked, copied, missing = build_links_direct(
            R3_ROOT / args.stream / f'theme_{args.theme:02d}', files)
        write_words(out_dir, theme_words(conn, args.theme))
        _report(f'theme {args.theme}', files, out_dir, linked, copied, missing)
        launch_folder = out_dir
    else:
        summary(conn)
        conn.close()
        return

    conn.close()
    if launch_folder and not args.no_launch:
        launch_viewer(launch_folder)
        print(f'🖥️  pdf_viewer открыт: {launch_folder}')


if __name__ == '__main__':
    main()
