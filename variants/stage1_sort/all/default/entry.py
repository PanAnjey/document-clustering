"""Golden Stage 1 — вариант `default`: полный пересчёт Stage 1 с нуля.

Запускает stages.stage1_sorting.run_stage_1_sort() после безусловной полной
очистки всех рабочих директорий (Sorted/, ErrorFiles/, FailedExtraction/,
Extracted/, Embeddings/, Clusters/, CentralDocuments/, PipelineData/,
Logs/, unpacked_zip/, Sorted_golden/) + TRUNCATE public.documents +
gold_stage1.documents. Затем восстанавливает SourceFiles/ из эталона
SourceFiles_golden/ (или создаёт эталон при первом прогоне), запускает
сортировку, снимает snapshot в gold_stage1.documents и копирует
Sorted/ → Sorted_golden/.

Контракт см. `variants/README.md`. Никаких правок stages/ — только обёртка.

run_id — служебный: попадает в `experiment_registry` для аудита,
для golden-прогонов можно передавать что-то вроде `golden_stage1_20260712`.
subformat — должен быть 'all' (подэтапов у Stage 1 нет).
params:
    source_dir (опц.): путь к источнику для ПЕРВОГО прогона (когда
                      SourceFiles_golden/ ещё пуст). При последующих прогонах
                      всегда берётся из SourceFiles_golden/.
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
    """Полная очистка всех рабочих директорий и логов прошлых прогонов.

    Source (SourceFiles/) НЕ трогается — immutable. Все остальные рабочие
    директории удаляются целиком и пересоздаются пустыми: Sorted/, ErrorFiles/,
    FailedExtraction/, Extracted/, Embeddings/, Clusters/, CentralDocuments/,
    PipelineData/, Sorted_golden/, unpacked_zip/, Logs/ (с реоткрытием логгера).
    Logs/ чистится последней — чтобы логи очистки остальных директорий
    попадали в старый лог-файл; затем handlers логгера закрываются,
    Logs/ удаляется, пересоздаётся пустой и логгер реинициализируется.
    """
    from pipeline_state import _force_remove_dir
    from config import cfg

    # Рабочие директории (без Logs/)
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
        cfg.ROOT / "unpacked_zip",
    ]
    for d in targets:
        if d is None or not d.exists():
            continue
        if _force_remove_dir(d):
            logger.info(f"[golden_stage1] cleaned {d}")
        else:
            logger.warning(f"[golden_stage1] could not clean {d}")

    # Logs/ — последней: закрываем handlers, удаляем, пересоздаём пустой,
    # реинициализируем логгер поверх чистого файла.
    if cfg.LOG_DIR and cfg.LOG_DIR.exists():
        logger.info(f"[golden_stage1] cleaning Logs/ (last)...")
        _reset_logger_before_dir_wipe()
        if _force_remove_dir(cfg.LOG_DIR):
            pass  # логирование возобновится после реинита
        else:
            # Не удалось удалить — всё равно реиниц логгер
            pass

    # Пересоздание обязательных пустых директорий
    for d in (cfg.ROOT / "Sorted", cfg.ERRORS_DIR, cfg.FAILED_EXTRACTION_DIR,
              cfg.EXTRACT_ROOT, cfg.EMBEDDINGS_DIR, cfg.LOG_DIR):
        if d is not None:
            d.mkdir(parents=True, exist_ok=True)

    # Реинициализация логгера поверх чистой Logs/
    _reset_logger_after_dir_wipe()
    logger.info(f"[golden_stage1] full cleanup done — fresh logs")


def _reset_logger_before_dir_wipe():
    """Закрыть только FileHandler'ы FileOrganizer-логгера, чтобы освободить
    файловый handle на Logs/file_organizer.log. StreamHandler-ы (обёрнутые
    поверх sys.stdout) НЕ закрываем и не снимаем — закрытие TextIOWrapper
    потянет за собой sys.stdout.buffer, а GC StreamHandler-а закроет его
    stream (TextIOWrapper.close() → buffer.close()). StreamHandler остаётся
    висеть на время пересоздания Logs/ и дальше — он будет переиспользован."""
    import logging
    lg = logging.getLogger("FileOrganizer")
    for h in list(lg.handlers):
        if isinstance(h, logging.FileHandler):
            try:
                h.close()
            except Exception:
                pass
            try:
                lg.removeHandler(h)
            except Exception:
                pass


def _reset_logger_after_dir_wipe():
    """Пересоздать ТОЛЬКО FileHandler-ы. StreamHandler оставляем как есть —
    он держит sys.stdout-обёртку от исходного setup_logger и GC StreamHandler-а
    в момент добавления нового привёл бы к закрытию sys.stdout.buffer. Так что
    консольный handler просто переиспользуется.
    """
    import logging
    from config import cfg

    lg = logging.getLogger("FileOrganizer")
    formatter = logging.Formatter(cfg.LOG_FORMAT, datefmt=cfg.LOG_DATE_FORMAT)

    file_handler = logging.FileHandler(cfg.LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    lg.addHandler(file_handler)


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


def _count_files(d: Path) -> int:
    """Подсчёт файлов в директории рекурсивно (0 при отсутствии)."""
    try:
        return sum(1 for _ in d.rglob("*") if _.is_file())
    except Exception:
        return 0


def _ensure_source_from_golden(source_dir: Path) -> dict:
    """Восстанавливает SourceFiles/ из эталона или создаёт эталон.

    Логика:
      (A) Эталон SourceFiles_golden/ уже заполнен (есть файлы):
          → SourceFiles/ WIP'ается и копируется из SourceFiles_golden/.
          → SourceFiles/ становится точной копией эталона.
      (B) Эталон пуст, но source_dir (SourceFiles/) не пуст:
          → ПЕРВЫЙ прогон. Копируем source_dir → SourceFiles_golden/.
          → SourceFiles/ не трогаем.
      (C) И эталон пуст, и source пуст:
          → ошибка: нельзя сортировать ≤0 файлов.

    Возвращает dict с ключами:
        mode: 'restored' | 'snapshot_created' | 'error'
        source_files: int — сколько файлов оказалось в SourceFiles/ после операции
        golden_files: int — сколько файлов в SourceFiles_golden/
    """
    from pipeline_state import _force_remove_dir

    golden_dir = paths.gold_source_dir()
    golden_dir.mkdir(parents=True, exist_ok=True)

    golden_files = _count_files(golden_dir)
    source_files = _count_files(source_dir)

    if golden_files > 0:
        # (A) Восстановление источника из эталона
        logger.info(
            f"[golden_stage1] restoring SourceFiles/ from gold snapshot "
            f"({golden_files} files in {golden_dir})"
        )
        # Wipe SourceFiles/ целиком
        if source_dir.exists():
            if not _force_remove_dir(source_dir):
                logger.warning(
                    f"[golden_stage1] could not wipe {source_dir} before restore"
                )
        source_dir.mkdir(parents=True, exist_ok=True)
        shutil.copytree(golden_dir, source_dir, dirs_exist_ok=True)
        restored = _count_files(source_dir)
        logger.info(f"[golden_stage1] restored {restored} files into {source_dir}")
        if restored != golden_files:
            logger.warning(
                f"[golden_stage1] restored {restored} != golden {golden_files}"
            )
        return {
            "mode": "restored",
            "source_files": restored,
            "golden_files": golden_files,
        }

    if source_files > 0:
        # (B) Первый прогон — создаём эталон из source_dir
        logger.info(
            f"[golden_stage1] FIRST RUN: snapshotting {source_dir} "
            f"({source_files} files) → {golden_dir}"
        )
        shutil.copytree(source_dir, golden_dir, dirs_exist_ok=True)
        golden_after = _count_files(golden_dir)
        logger.info(
            f"[golden_stage1] gold snapshot created: {golden_after} files in {golden_dir}"
        )
        return {
            "mode": "snapshot_created",
            "source_files": source_files,
            "golden_files": golden_after,
        }

    # (C) Оба пусты
    msg = (f"Both source ({source_dir}) and gold ({golden_dir}) are empty — "
           f"nothing to sort. Place files into {source_dir} and re-run.")
    logger.critical(f"[golden_stage1] {msg}")
    return {
        "mode": "error",
        "source_files": 0,
        "golden_files": 0,
        "error": msg,
    }


def run(run_id: str, subformat: str, params: dict) -> dict:
    """Запуск золотой Stage 1 — полный пересчёт с нуля.

    Всегда:
      1. Безусловно очищает все рабочие директории (Sorted/, ErrorFiles/,
         FailedExtraction/, Extracted/, Embeddings/, Clusters/,
         CentralDocuments/, PipelineData/, Logs/, unpacked_zip/,
         Sorted_golden/) + TRUNCATE public.documents + gold_stage1.documents.
      2. Восстанавливает SourceFiles/ из эталона SourceFiles_golden/.
         Если эталона ещё нет (первый прогон) — создаёт его из текущего
         SourceFiles/ (snapshot).
      3. Запускает stages.stage1_sorting.run_stage_1_sort().
      4. Снимает snapshot public.documents → gold_stage1.documents.
      5. Копирует Sorted/ → Sorted_golden/.

    Параметры (params dict):
        source_dir (опц.): путь к источнику; по умолчанию cfg.SOURCE_DIR.
                          Используется только при создании эталона (первый
                          прогон). При последующих прогонах игнорируется —
                          источник всегда берётся из SourceFiles_golden/.
    """
    from config import cfg

    if subformat != "all":
        logger.warning(
            f"[golden_stage1] subformat={subformat!r} ignored (expected 'all')"
        )

    # Источник
    source_dir_str = params.get("source_dir") or str(cfg.SOURCE_DIR)
    source_dir = Path(source_dir_str)
    if not source_dir.exists():
        # SourceFiles может быть пустым/удалённым — это ОК, если эталон есть.
        # Но.Future-_ensure_source_from_golden сам создаст source_dir при
        # восстановлении.
        source_dir.mkdir(parents=True, exist_ok=True)

    # Регистрируем прогон
    registry.insert_registry_row(
        run_id=run_id,
        dag_name="golden_stage1_run",
        note=f"Golden Stage 1: full re-sort of {source_dir} (clean from scratch + restore source)",
        params={"source_dir": str(source_dir)},
    )

    try:
        # ===== 1. Чистим прошлые выходы (безусловно) =====
        logger.info("[golden_stage1] Step 1/5: cleaning previous outputs...")
        _clean_dirs()
        _truncate_legacy_pipeline_state()

        # ===== 2. Восстанавливаем источник из эталона =====
        logger.info("[golden_stage1] Step 2/5: restore SourceFiles from gold snapshot...")
        restore_info = _ensure_source_from_golden(source_dir)
        if restore_info.get("mode") == "error":
            registry.update_registry_status(run_id, "failed")
            return {"status": "error", "error": restore_info.get("error"),
                    "metrics": {"restore": restore_info}}
        source_file_count = restore_info["source_files"]

        # ===== 3. Запускает existing stages.stage1_sorting.run_stage_1_sort() =====
        # NOTE: stages.stage1_sorting.run_stage_1_sort() читает cfg.SOURCE_DIR
        # directly. Если source_dir != cfg.SOURCE_DIR (пользователь передал
        # кастомный путь), временно переопределяем. Так как golden Stage 1
        # запускается строго один за раз (schedule=None, manual trigger),
        # мутация singleton cfg.SOURCE_DIR безопасна в пределах процесса.
        original_source = cfg.SOURCE_DIR
        try:
            cfg.SOURCE_DIR = source_dir
            logger.info(
                f"[golden_stage1] Step 3/5: running stages.stage1_sorting.run_stage_1_sort()..."
            )
            from stages.stage1_sorting import run_stage_1_sort
            sorted_files = asyncio.run(run_stage_1_sort())
            n_sorted = len(sorted_files)
        finally:
            cfg.SOURCE_DIR = original_source
        logger.info(f"[golden_stage1] sorted {n_sorted} files")

        # ===== 4. Snapshot public.documents → gold_stage1.documents =====
        logger.info("[golden_stage1] Step 4/5: snapshotting into gold_stage1...")
        gold_count = _snapshot_public_into_gold() if n_sorted > 0 else 0

        # ===== 5. Copy Sorted/ → Sorted_golden/ =====
        logger.info("[golden_stage1] Step 5/5: copytree Sorted → Sorted_golden...")
        copy_info = _copy_sorted_to_golden()

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
            "source_restore_mode": restore_info.get("mode"),
            "golden_source_files": restore_info.get("golden_files"),
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