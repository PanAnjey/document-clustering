# stages/stage1_sorting.py
# Версия: 5.0
# Дата: 2026-06-07
# Описание: Этап 1 - Сортировка файлов по форматам.

from __future__ import annotations

import asyncio
import shutil
import sys
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from config import cfg
from logger_utils import logger


def _sort_worker(f: Path) -> Optional[Dict]:
    """Worker для multiprocessing: сортирует один файл, возвращает dict с source/type или None."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from sorting import pdf_processor, text_processor, excel_processor, image_processor, xml_processor
    from sorting.verify_signature import rename_by_signature

    try:
        # Проверка сигнатуры — переименование если расширение не соответствует содержимому
        f, sig_category = rename_by_signature(f)

        # Ошибочные файлы (PKCS#7, неизвестная сигнатура) — сразу в ErrorFiles
        if sig_category == 'error':
            from file_processor import move_file_to_error
            move_file_to_error(f)
            return None

        # Диспетчер по расширению (теперь проверенному)
        ext = f.suffix.lower()

        if ext == '.pdf':
            result = pdf_processor.process_pdf(f)
        elif ext in ('.docx', '.doc', '.rtf', '.txt', '.odt', '.odp'):
            result = text_processor.process_text(f)
        elif ext in ('.xlsx', '.xls', '.xlsm', '.xlsb', '.xltx', '.xlt', '.xltm', '.csv', '.ods'):
            result = excel_processor.process_excel(f)
        elif ext in ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.svg', '.jfif'):
            result = image_processor.process_image(f)
        elif ext in ('.xml', '.xsd', '.xsl', '.xslt', '.wsdl'):
            result = xml_processor.process_xml(f)
        else:
            # Неизвестное расширение после проверки сигнатуры — в ErrorFiles
            from file_processor import move_file_to_error
            move_file_to_error(f)
            return None

        if result:
            new_path, cat = result
            # PDF classification for dispatched processors
            if cat == "pdf" and isinstance(new_path, Path):
                subcat = file_processor.classify_pdf(new_path)
                sorted_dir = cfg.TARGETS.get(subcat)
                if sorted_dir:
                    sorted_dir.mkdir(parents=True, exist_ok=True)
                    dst = sorted_dir / new_path.name
                    cnt = 1
                    while dst.exists():
                        dst = sorted_dir / f"{new_path.stem}_{cnt}{new_path.suffix}"
                        cnt += 1
                    shutil.move(str(new_path), str(dst))
                    new_path = dst
                    cat = subcat
            return {"source": str(new_path), "type": cat}
        return None

    except Exception as e:
        logger.debug(f"Sort worker error for {f.name}: {e}")
        return None


def _run_sort_pool(files: List[Path], num_workers: int) -> List[Dict]:
    """Запускает pool процессов для сортировки файлов."""
    actual_workers = min(num_workers, len(files))
    total = len(files)
    result_errors = 0
    results = []
    done = 0
    log_step = max(1, total // 20)

    with ProcessPoolExecutor(max_workers=actual_workers) as executor:
        fut_map = {executor.submit(_sort_worker, f): f for f in files}
        for fut in as_completed(fut_map):
            r = fut.result()
            if r:
                results.append(r)
            else:
                result_errors += 1
            done += 1
            if done % log_step == 0 or done == total:
                logger.info(f"Сортировка: {done}/{total}")

    logger.info(f"Сортировка завершена: {len(results)} OK, {result_errors} ошибок")
    return results


async def run_stage_1_sort() -> List[Dict]:
    """Выполняет этап 1: сортировку файлов по форматам."""
    from database import DatabaseManager

    source_path = cfg.SOURCE_DIR
    if not source_path.exists():
        logger.critical(f"Исходная директория не найдена: {source_path}")
        sys.exit(1)

    all_items = list(source_path.rglob("*"))
    logger.info(f"Найдено элементов: {len(all_items)}")

    files = [f for f in all_items if f.is_file()]
    logger.info(f"Файлов: {len(files)}")

    if not files:
        logger.error("Нет файлов для обработки.")
        return []

    # ШАГ 1: Архивы — обрабатываются ДО пула (последовательно, с общим отчётом)
    from sorting.archive_processor import process_archives, ARCHIVE_EXTS
    archive_files = [f for f in files if f.suffix.lower() in ARCHIVE_EXTS or str(f).endswith('.tar.gz')]
    non_archive_files = [f for f in files if f not in archive_files]

    loop = asyncio.get_running_loop()
    archive_results = await loop.run_in_executor(None, process_archives, source_path)
    sorted_files = list(archive_results)

    # ШАГ 2: Остальные файлы — параллельная сортировка в пуле
    if non_archive_files:
        num_workers = min(cfg.MAX_CONCURRENT_FILES, len(non_archive_files))
        logger.info(f"Запуск сортировки {len(non_archive_files)} файлов в {num_workers} процессов...")

        pool_results = await loop.run_in_executor(None, _run_sort_pool, non_archive_files, num_workers)
        sorted_files.extend(pool_results)

    if not sorted_files:
        logger.warning("Ни один файл не был отсортирован.")
        return []

    db = None
    try:
        db = DatabaseManager()
        records = [(d["source"], d["type"]) for d in sorted_files]
        db.insert_documents_batch(records)
        logger.info(f"Сохранено {len(sorted_files)} документов в БД.")
    except Exception as e:
        logger.error(f"БД недоступна, данные сохранены только в памяти: {e}")
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass

    logger.info(f"Этап 1 завершён. Отсортировано: {len(sorted_files)} файлов.")

    # Очистка пустых поддиректорий в Sorted/
    _cleanup_empty_dirs()

    return sorted_files


def _cleanup_empty_dirs():
    """Удаляет пустые поддиректории в Sorted/ — остаются только те, где есть файлы."""
    sorted_root = cfg.ROOT / "Sorted"
    if not sorted_root.exists():
        return

    removed = 0
    # Проходим снизу вверх (post-order) — сначала внутренние, потом внешние
    for d in sorted(sorted_root.rglob("*"), key=lambda x: len(x.parts), reverse=True):
        if d.is_dir():
            try:
                next(d.iterdir())
            except StopIteration:
                d.rmdir()
                removed += 1
                logger.debug(f"Удалена пустая директория: {d}")

    if removed:
        logger.info(f"Удалено пустых директорий: {removed}")
