# show_cluster.py
# Визуальный просмотр файлов кластера (Phase 2) и под-кластеров (Phase 4)
# через pdf_viewer.py.
#
# Источник визуализации: PDF-артефакты Phase 1 в cfg.EXTRA['excel_xlsx_pages']
# ({stem}.pdf, первые 1-2 страницы). Для выбранного кластера создаётся папка
# с ЖЁСТКИМИ ССЫЛКАМИ (hardlink, тот же том D: — мгновенно, без копирования)
# на эти PDF, и в ней открывается pdf_viewer.py.
#
# Запуск:
#   python show_cluster.py 1              # тип 1 (первый уровень) + viewer
#   python show_cluster.py -1             # шумовой кластер
#   python show_cluster.py 11.3           # ПОД-кластер 3 типа 11 (Phase 4)
#   python show_cluster.py 7 --sample 50  # случайные 50 файлов кластера
#   python show_cluster.py --all          # плоские папки для всех типов (+ шум)
#   python show_cluster.py --layout       # дерево cluster_NN\sub_M (+ шум)
#   python show_cluster.py 1 --no-launch  # только создать папку

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import psycopg2

from experiment_config import DB_URL  # также чинит sys.path и UTF-8 stdout
from config import cfg

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_VIEWER = PROJECT_ROOT / 'pdf_viewer.py'
VIEW_ROOT = Path(r'D:\FileOrganizer\Reports\ClusterViews')


def fetch_cluster_files(conn, label, sample=0):
    """file_path'ы train-набора для указанного label (label=-1 → шум)."""
    with conn.cursor() as cur:
        if sample and sample > 0:
            cur.execute(
                """
                SELECT t.file_path
                FROM temp_excel_train t
                JOIN temp_hdbscan h ON h.train_id = t.id
                WHERE h.label = %s
                ORDER BY random()
                LIMIT %s
                """,
                (label, sample),
            )
        else:
            cur.execute(
                """
                SELECT t.file_path
                FROM temp_excel_train t
                JOIN temp_hdbscan h ON h.train_id = t.id
                WHERE h.label = %s
                ORDER BY t.file_path
                """,
                (label,),
            )
        return [r[0] for r in cur.fetchall()]


def fetch_subcluster_files(conn, type_label, sub_label, sample=0):
    """file_path'ы под-кластера (type_label, sub_label) из temp_second_level."""
    with conn.cursor() as cur:
        if sample and sample > 0:
            cur.execute(
                """
                SELECT t.file_path
                FROM temp_excel_train t
                JOIN temp_second_level s ON s.train_id = t.id
                WHERE s.type_label = %s AND s.sub_label = %s
                ORDER BY random()
                LIMIT %s
                """,
                (type_label, sub_label, sample),
            )
        else:
            cur.execute(
                """
                SELECT t.file_path
                FROM temp_excel_train t
                JOIN temp_second_level s ON s.train_id = t.id
                WHERE s.type_label = %s AND s.sub_label = %s
                ORDER BY t.file_path
                """,
                (type_label, sub_label),
            )
        return [r[0] for r in cur.fetchall()]


def available_labels(conn):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT label, COUNT(*) FROM temp_hdbscan "
            "GROUP BY label ORDER BY COUNT(*) DESC"
        )
        return cur.fetchall()


def available_subs(conn):
    """(type_label, sub_label, count) из temp_second_level; [] если фаза 4 не запускалась."""
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('public.temp_second_level')")
        if cur.fetchone()[0] is None:
            return []
        cur.execute(
            "SELECT type_label, sub_label, COUNT(*) FROM temp_second_level "
            "GROUP BY type_label, sub_label ORDER BY type_label, COUNT(*) DESC"
        )
        return cur.fetchall()


def build_view_folder(name, file_paths):
    """Папка с hardlink'ами на PDF-артефакты файлов кластера."""
    pdf_dir = cfg.EXTRA['excel_xlsx_pages']
    out_dir = VIEW_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Идемпотентность: чистим старые ссылки (hardlink-удаление не трогает оригинал)
    for old in out_dir.glob('*.pdf'):
        try:
            old.unlink()
        except Exception:
            pass

    linked, missing, copied = 0, 0, 0
    seen = set()
    for fp in file_paths:
        stem = Path(fp).stem
        if stem in seen:          # коллизия stem → один PDF на двоих (редко)
            continue
        seen.add(stem)
        src = pdf_dir / f"{stem}.pdf"
        if not src.exists():      # файл извлечён COM-fallback'ом → PDF нет
            missing += 1
            continue
        dst = out_dir / src.name
        try:
            os.link(src, dst)     # жёсткая ссылка: мгновенно, место не занимает
            linked += 1
        except OSError:
            try:
                shutil.copy2(src, dst)  # fallback (другой том и т.п.)
                copied += 1
            except Exception:
                missing += 1
    return out_dir, linked, copied, missing


def launch_viewer(folder):
    """Открыть pdf_viewer.py на папке (отсоединённый GUI-процесс)."""
    creationflags = 0
    if os.name == 'nt':
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    subprocess.Popen(
        [sys.executable, str(PDF_VIEWER), str(folder)],
        cwd=str(PROJECT_ROOT),
        creationflags=creationflags,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _report(tag, files, out_dir, linked, copied, missing):
    note = f', {copied} скопировано' if copied else ''
    miss = f', ⚠️ {missing} без PDF (COM-fallback)' if missing else ''
    print(f'✅ {tag}: {len(files)} файлов → {out_dir} ({linked} ссылок{note}{miss})')


def main():
    ap = argparse.ArgumentParser(description='Визуальный просмотр кластеров Excel')
    ap.add_argument('label', nargs='?', default=None,
                    help='тип (1), шум (-1) или тип.подтип (11.3); без аргумента — список')
    ap.add_argument('--all', action='store_true',
                    help='плоские папки всех типов (+ шум), viewer не открывать')
    ap.add_argument('--layout', action='store_true',
                    help='полное дерево ClusterViews\\cluster_NN\\sub_M (+ cluster_noise)')
    ap.add_argument('--sample', type=int, default=0,
                    help='случайные N файлов вместо всего кластера')
    ap.add_argument('--no-launch', action='store_true',
                    help='только создать папку, не открывать pdf_viewer')
    args = ap.parse_args()

    conn = psycopg2.connect(DB_URL)

    # ── Список кластеров и под-кластеров ─────────────────────────
    if args.label is None and not args.all and not args.layout:
        print('Типы (label: docs):')
        for lab, cnt in available_labels(conn):
            tag = ' (шум)' if lab == -1 else ''
            print(f'  {lab}: {cnt}{tag}')
        subs = available_subs(conn)
        if subs:
            print('\nПод-кластеры (тип.подтип: docs):')
            cur_t = None
            for t, s, c in subs:
                if t != cur_t:
                    cur_t = t
                print(f'  {t}.{s}: {c}')
        print('\nПримеры: python show_cluster.py 1      — тип 1')
        print('         python show_cluster.py 11.3   — под-кластер 3 типа 11')
        print('         python show_cluster.py --layout — всё дерево директорий')
        conn.close()
        return

    launch_folder = None
    suffix = f"_sample{args.sample}" if args.sample else ""

    # ── Полное дерево cluster_NN\sub_M + cluster_noise ───────────
    if args.layout:
        subs = available_subs(conn)
        if not subs:
            conn.close()
            sys.exit('❌ temp_second_level нет — сначала запустите Phase 4.')
        by_type = {}
        for t, s, _c in subs:
            by_type.setdefault(t, []).append(s)
        for t in sorted(by_type):
            for s in sorted(by_type[t]):
                files = fetch_subcluster_files(conn, t, s)
                out_dir, linked, copied, missing = build_view_folder(
                    f'cluster_{t:02d}/sub_{s}', files)
                _report(f'{t}.{s}', files, out_dir, linked, copied, missing)
        # Шум плоско (второго уровня для него нет)
        files = fetch_cluster_files(conn, -1)
        out_dir, linked, copied, missing = build_view_folder('cluster_noise', files)
        _report('noise', files, out_dir, linked, copied, missing)
        conn.close()
        print(f'\nДерево готово: {VIEW_ROOT}')
        return

    # ── Плоские папки всех типов (+ шум) ─────────────────────────
    if args.all:
        for lab, _cnt in available_labels(conn):
            files = fetch_cluster_files(conn, lab, args.sample)
            if not files:
                continue
            name = f'cluster_noise{suffix}' if lab == -1 else f'cluster_{lab}{suffix}'
            out_dir, linked, copied, missing = build_view_folder(name, files)
            _report(f'cluster {lab}', files, out_dir, linked, copied, missing)
        conn.close()
        return

    # ── Одиночный label: "11.3" (под-кластер) или "1"/"-1" (тип/шум) ──
    if '.' in args.label:
        t_part, s_part = args.label.split('.', 1)
        t, s = int(t_part), int(s_part)
        files = fetch_subcluster_files(conn, t, s, args.sample)
        if not files:
            conn.close()
            sys.exit(f'⚠️ Под-кластер {t}.{s}: нет файлов (проверьте label)')
        out_dir, linked, copied, missing = build_view_folder(
            f'cluster_{t:02d}/sub_{s}{suffix}', files)
        _report(f'{t}.{s}', files, out_dir, linked, copied, missing)
        launch_folder = out_dir
    else:
        lab = int(args.label)
        files = fetch_cluster_files(conn, lab, args.sample)
        if not files:
            conn.close()
            sys.exit(f'⚠️ Кластер {lab}: нет файлов (проверьте label)')
        name = f'cluster_noise{suffix}' if lab == -1 else f'cluster_{lab}{suffix}'
        out_dir, linked, copied, missing = build_view_folder(name, files)
        _report(f'cluster {lab}', files, out_dir, linked, copied, missing)
        launch_folder = out_dir

    conn.close()

    if not args.no_launch and launch_folder:
        launch_viewer(launch_folder)
        print(f'🖥️  pdf_viewer открыт: {launch_folder}')


if __name__ == '__main__':
    main()
