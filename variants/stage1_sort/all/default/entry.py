"""Golden Stage 1 — вариант `default`: запускает существующий
stages.stage1_sorting.run_stage_1_sort() как «золотой»_stage 1 и
снимает snapshot в gold_stage1.documents + D:\\FileOrganizer\\Sorted_golden\\.

Контракт см. `variants/README.md`. Никаких правок stages/ — только обёртка.

run_id — служебный: попадает в `experiment_registry` для аудита,
для golden-прогонов можно передавать что-то вроде `golden_stage1_20260712`.
subformat — должен быть 'all' (подэтапов у Stage 1 нет).
params:
    source_dir (опц.): путь к источнику; по умолчанию cfg.SOURCE_DIR.
                     Настройка cfg.SOURCE_DIR не переопределяется — путь
                     читается напрямую, но НЕ муттируется global Config.
"""
import asyncio
import shutil
import sys
from pathlib import Path

from logger_utils import logger

# imports из пакета experiments — платеж заisers
from airflow.experiments import db, paths, registry


def _truncate_legacy_pipeline_state():
    """Чистит БД + директории состояния прошлых прогонов.

    TRUNCATE public.documents CASCADE → автоматически фильтруется на
    documents_<fmt>, text_embeddings_<fmt>, image_embeddings_<fmt>,
    *_backup (FK ON DELETE CASCADE).
    """
    conn = db.get_conn()
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE public.documents RESTART IDENTITY CASCADE")
        cur.execute("TRUNCATE TABLE gold_stage1.documents RESTART IDENTITY CASCADE")
    conn.commit()
    logger.info("[golden_stage1] truncated public.documents + gold_stage1.documents")


def _clean_dirs():
    """Чистит выходные директории прошлых прогонов Stage 1.

    Source (SourceFiles/) НЕ трогается — immutable.
    """
    from pipeline_state import _force_remove_dir
    from config import cfg

    targets = [
        cfg.ROOT / "Sorted",
        paths.gold_sorted_dir(),
        cfg.ERRORS_DIR,
        cfg.FAILED_EXTRACTION_DIR,
        cfg.EXTRACT_ROOT,
        cfg.EMBEDDINGS_DIR,
        cfg.ROOT / "Clusters",
        cfg.ROOT / "CentralDocuments",
        cfg.ROOT / "PipelineData",
    ]
    for d in targets:
        if d is None or not d.exists():
            continue
        if _force_remove_dir(d):
            logger.info(f"[golden_stage1] cleaned {d}")
        else:
            logger.warning(f"[golden_stage1] could not clean {d}")


def _snapshot_public_into_gold() -> int:
    """INSERT INTO gold_stage1.documents SELECT FROM public.documents."""
    conn = db.get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO gold_stage1.documents (file_path, format_type)
            SELECT file_path, format_type
            FROM public.documents
            WHERE file_path IS NOT NULL AND format_type IS NOT NULL
            ON CONFLICT (file_path) DO NOTHING
        """)
        cur.execute("SELECT count(*) FROM gold_stage1.documents")
        n = cur.fetchone()[0]
    conn.commit()
    return n


def _copy_sorted_to_golden() -> dict:
    """Копирует D:\\FileOrganizer\\Sorted\\ → D:\\FileOrganizer\\Sorted_golden\\.

    Использует dirs_exist_ok=True (Python 3.8+).
    """
    from config import cfg

    src = cfg.ROOT / "Sorted"
    dst = paths.gold_sorted_dir()
    if not src.exists():
        logger.warning(f"[golden_stage1] no Sorted/ dir to copy from: {src}")
        return {"copied": False, "format_counts": {}}
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    logger.info(f"[golden_stage1] copytree {src} → {dst}")

    # Подсчёт по формат-папкам
    format_counts = {}
    for child in sorted(dst.iterdir()):
        if child.is_dir():
            cnt = sum(1 for _ in child.rglob("*") if _.is_file())
            if cnt > 0:
                format_counts[child.name] = cnt
    return {"copied": True, "format_counts": format_counts}


def run(run_id: str, subformat: str, params: dict) -> dict:
    """Запуск золотой Stage 1.

    См. модуль docstring для контракта.

    Параметры (params dict):
        source_dir (опц.):           путь к источнику; по умолчанию cfg.SOURCE_DIR.
        skip_clean (опц., bool):      Не чистить выходные директории (Sorted/,
                                     Sorted_golden/, ErrorFiles/, etc).
                                     По умолчанию False (полный пересчёт).
        skip_truncate (опц., bool):  Не TRUNCATE'ить public.documents и
                                     gold_stage1.documents. По умолчанию False.
        skip_copy_to_golden (опц., bool):
                                     Не делать copytree Sorted/→Sorted_golden/.
                                     Полезно для smoke-test, чтобы не трогать
                                     существующее дерево. По умолчанию False.
    """
    from config import cfg

    if subformat != "all":
        logger.warning(
            f"[golden_stage1] subformat={subformat!r} ignored (expected 'all')"
        )

    skip_clean = bool(params.get("skip_clean", False))
    skip_truncate = bool(params.get("skip_truncate", False))
    skip_copy_to_golden = bool(params.get("skip_copy_to_golden", False))

    # Источник
    source_dir_str = params.get("source_dir") or str(cfg.SOURCE_DIR)
    source_dir = Path(source_dir_str)
    if not source_dir.exists():
        msg = f"Source dir not found: {source_dir}"
        logger.critical(f"[golden_stage1] {msg}")
        return {"status": "error", "error": msg, "metrics": {}}

    # Регистрируем прогон
    registry.insert_registry_row(
        run_id=run_id,
        dag_name="golden_stage1_run",
        note=f"Golden Stage 1: full re-sort of {source_dir} "
             f"(skip_clean={skip_clean}, skip_truncate={skip_truncate}, "
             f"skip_copy_to_golden={skip_copy_to_golden})",
        params={"source_dir": str(source_dir)},
    )

    try:
        # ===== 1. Чистим прошлые выходы =====
        if skip_clean or skip_truncate:
            logger.info(
                f"[golden_stage1] Step 1/4: SKIPPED cleaning "
                f"(skip_clean={skip_clean}, skip_truncate={skip_truncate})"
            )
        else:
            logger.info("[golden_stage1] Step 1/4: cleaning previous outputs...")
            _clean_dirs()
            _truncate_legacy_pipeline_state()

        # ===== 2. Запускает existing stages.stage1_sorting.run_stage_1_sort() =====
        # NOTE: stages.stage1_sorting.run_stage_1_sort() читает cfg.SOURCE_DIR
        # directly. Если params['source_dir'] != cfg.SOURCE_DIR, нам нужно
        # временно переопределить cfg.SOURCE_DIR. Поскольку golden Stage 1
        # запускается строго один за раз (schedule=None, manual trigger),
        # мутация singleton cfg.SOURCE_DIR безопасна в пределах процесса.
        original_source = cfg.SOURCE_DIR
        try:
            cfg.SOURCE_DIR = source_dir
            logger.info(
                f"[golden_stage1] Step 2/4: running stages.stage1_sorting.run_stage_1_sort()..."
            )
            from stages.stage1_sorting import run_stage_1_sort
            sorted_files = asyncio.run(run_stage_1_sort())
            n_sorted = len(sorted_files)
        finally:
            cfg.SOURCE_DIR = original_source
        logger.info(f"[golden_stage1] sorted {n_sorted} files")

        # ===== 3. Snapshot public.documents → gold_stage1.documents =====
        logger.info("[golden_stage1] Step 3/4: snapshotting into gold_stage1...")
        gold_count = _snapshot_public_into_gold() if n_sorted > 0 else 0

        # ===== 4. Copy Sorted/ → Sorted_golden/ =====
        if skip_copy_to_golden:
            logger.info("[golden_stage1] Step 4/4: SKIPPED copytree to Sorted_golden")
            copy_info = {"copied": False, "format_counts": {}, "skipped": True}
        else:
            logger.info("[golden_stage1] Step 4/4: copytree Sorted → Sorted_golden...")
            copy_info = _copy_sorted_to_golden()

        # ===== Метрики =====
        # Количество файлов в источнике (для аудита)
        try:
            source_file_count = sum(1 for _ in source_dir.rglob("*") if _.is_file())
        except Exception:
            source_file_count = -1

        # Количество ошибок
        error_count = 0
        if cfg.ERRORS_DIR and cfg.ERRORS_DIR.exists():
            error_count = sum(1 for _ in cfg.ERRORS_DIR.rglob("*") if _.is_file())

        # БД-раскладка по форматам
        conn = db.get_conn()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT format_type, count(*)
                FROM gold_stage1.documents
                WHERE format_type IS NOT NULL
                GROUP BY format_type
                ORDER BY 2 DESC
            """)
            db_format_counts = {row[0]: row[1] for row in cur.fetchall()}
        conn.close()

        metrics = {
            "source_files": source_file_count,
            "sorted_files": n_sorted,
            "gold_stage1_rows": gold_count,
            "files_in_errors_dir": error_count,
            "db_format_counts": db_format_counts,         # из gold_stage1.documents
            "disk_format_counts": copy_info.get("format_counts", {}),  # из Sorted_golden/
        }
        registry.update_registry_metrics(run_id, metrics)
        registry.update_registry_status(run_id, "success")
        logger.info(f"[golden_stage1] SUCCESS: metrics={metrics}")
        return {"status": "ok", "metrics": metrics}

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.exception(f"[golden_stage1] FAILED: {e}\n{tb}")
        registry.update_registry_status(run_id, "failed")
        return {"status": "error", "error": str(e), "metrics": {}, "traceback": tb}