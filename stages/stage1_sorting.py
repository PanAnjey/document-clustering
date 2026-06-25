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
    import sys
    sys.path.insert(0, r"D:\Yandex.Disk\PYTHON\NLTK\Кластеризация")
    from sorting import pdf_processor, text_processor, excel_processor, image_processor, xml_processor

    ext = f.suffix.lower()

    try:
        # Dispatch to format-specific processors
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
            # Unknown format — try signature detection
            import file_processor
            result = file_processor.detect_file_type_by_signature(f)
            if not result:
                file_processor.move_file_to_error(f)
                return None
            detected_ext, cat = result
            new_path = f.with_suffix(detected_ext)
            try:
                f.rename(new_path)
            except Exception:
                new_path = f
            moved = file_processor.move_file_to_target(new_path, cat)
            if not moved:
                return None
            # PDF classification
            if cat == "pdf":
                subcat = file_processor.classify_pdf(moved)
                sorted_dir = cfg.TARGETS.get(subcat)
                if sorted_dir:
                    sorted_dir.mkdir(parents=True, exist_ok=True)
                    dst = sorted_dir / moved.name
                    cnt = 1
                    while dst.exists():
                        dst = sorted_dir / f"{moved.stem}_{cnt}{moved.suffix}"
                        cnt += 1
                    shutil.move(str(moved), str(dst))
                    moved = dst
                    cat = subcat
            return {"source": str(moved), "type": cat}

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

    except Exception:
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

    num_workers = min(cfg.MAX_CONCURRENT_FILES, len(files))
    logger.info(f"Запуск сортировки в {num_workers} процессов...")

    loop = asyncio.get_running_loop()
    sorted_files = await loop.run_in_executor(None, _run_sort_pool, files, num_workers)

    if not sorted_files:
        logger.warning("Ни один файл не был отсорирован.")
        return []

    try:
        db = DatabaseManager()
        records = [(d["source"], d["type"]) for d in sorted_files]
        db.insert_documents_batch(records)
        db.close()
        logger.info(f"Сохранено {len(sorted_files)} документов в БД.")
    except Exception as e:
        logger.error(f"БД недоступна, данные сохранены только в памяти: {e}")

    logger.info(f"Этап 1 завершён. Отсортировано: {len(sorted_files)} файлов.")
    return sorted_files
