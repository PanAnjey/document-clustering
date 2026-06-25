# stages/stage2_processing.py
# Версия: 5.0
# Дата: 2026-06-07
# Описание: Этап 2 - Обработка форматов (извлечение + эмбеддинги).

from __future__ import annotations

import asyncio
import multiprocessing
import time
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
import numpy as np

from config import cfg
from logger_utils import logger


# ===== Lazy import stop_event из main (circular-safe) =====
def _is_stopped() -> bool:
    from main import stop_event
    return stop_event.is_set()


def _get_pipeline():
    from pipeline_state import PipelineState
    return PipelineState()


def _get_progress():
    from state.progress_tracker import get_progress_tracker
    return get_progress_tracker()


# ===== Глобальное состояние этапа 2 =====
_extraction_groups: List[Dict] = []
_current_group_index: int = -1
_stage2_results: Dict[str, Dict] = {}
_substage_to_start: Optional[tuple] = None
_substage_select_event = asyncio.Event()
_awaiting_substage_select: bool = False
_stage2_active: Optional[tuple] = None
_stage2_busy: bool = False
_pending_substage: Optional[tuple] = None


def _format_has_extraction(format_type: str) -> bool:
    """Есть ли у формата подэтап извлечения текста/изображений."""
    EMBEDDINGS_ONLY_FORMATS = set()
    return format_type not in EMBEDDINGS_ONLY_FORMATS


def _find_group(format_type: str) -> Optional[Dict]:
    for g in _extraction_groups:
        if g["type"] == format_type:
            return g
    return None


FORMAT_FAMILY_ORDER = [
    "pdf", "word", "excel", "image", "xml",
]

def _family_priority(format_type: str) -> int:
    for i, family in enumerate(FORMAT_FAMILY_ORDER):
        if format_type.startswith(family + "_") or format_type == family:
            return i
    return len(FORMAT_FAMILY_ORDER)


def _prepare_extraction_groups(sorted_data: List[Dict]) -> None:
    """Подготавливает/восстанавливает группы форматов, сортируя по семейству."""
    global _extraction_groups, _current_group_index
    
    _extraction_groups = []
    _current_group_index = -1
    
    # Группируем файлы по типу формата
    format_files = defaultdict(list)
    for d in sorted_data:
        format_files[d["type"]].append(d)
    
    # Создаем группы для каждого формата
    groups = []
    for format_type, files in format_files.items():
        has_extraction = _format_has_extraction(format_type)
        
        if format_type.startswith("word_") or format_type == "rtf":
            workers = min(cfg.ASPOSE_WORKERS, cfg.MAX_CONCURRENT_FILES)
        elif format_type.startswith("excel_"):
            workers = min(cfg.COM_EXCEL_WORKERS, cfg.MAX_CONCURRENT_FILES)
        else:
            workers = cfg.MAX_CONCURRENT_FILES
        
        groups.append({
            "type": format_type,
            "total": len(files),
            "workers": workers,
            "has_extraction": has_extraction,
            "extract_status": "pending",
            "embed_status": "locked" if has_extraction else "pending",
            "extract_ok": 0,
            "extract_errors": 0,
            "extract_elapsed": 0.0,
            "embed_ok": 0,
            "embed_errors": 0,
            "embed_elapsed": 0.0,
        })
    
    groups.sort(key=lambda g: (_family_priority(g["type"]), g["type"]))
    _extraction_groups.extend(groups)


def _persist_stage2_state() -> None:
    """Сохраняет под-статусы всех групп этапа 2 на диск."""
    data = {}
    for g in _extraction_groups:
        data[g["type"]] = {
            "extract": g["extract_status"],
            "embed": g["embed_status"],
            "extract_stats": {"ok": g["extract_ok"], "errors": g["extract_errors"], "elapsed": g["extract_elapsed"]},
            "embed_stats": {"ok": g["embed_ok"], "errors": g["embed_errors"], "elapsed": g["embed_elapsed"]},
        }
    _get_pipeline().save_stage2_groups(data)


def _check_gpu_for_embeddings() -> bool:
    """Проверка доступности GPU и моделей эмбеддингов."""
    global _gpu_checked
    
    if hasattr(_check_gpu_for_embeddings, "_checked"):
        return True
    
    from dependency_checker import validate_gpu
    validate_gpu()
    
    try:
        import torch
        if not torch.cuda.is_available():
            logger.warning("CUDA недоступна — эмбеддинги будут вычислены на CPU (медленно) или пропущены")
        
        dev = cfg.EMB_GPU_DEVICE
        if dev.startswith('cuda:'):
            idx = int(dev.split(':')[1])
            if torch.cuda.is_available() and idx >= torch.cuda.device_count():
                logger.error(f"GPU {dev} недоступна (всего {torch.cuda.device_count()}). Эмбеддинги могут не сработать.")
                return False
        
        # Проверка наличия моделей эмбеддингов на диске
        if not cfg.TEXT_EMB_MODEL_PATH.exists():
            logger.error(f"Модель эмбеддингов текста не найдена: {cfg.TEXT_EMB_MODEL_PATH}")
            return False
        
        _check_gpu_for_embeddings._checked = True
        logger.info(f"GPU/модели эмбеддингов проверены: {dev}")
        return True
    except ImportError:
        logger.error("torch не установлен — эмбеддинги недоступны")
        return False


def _embed_group_results(group_results: List[Dict]) -> List[Dict]:
    """Генерация эмбеддингов для результатов одной группы формата."""
    if not group_results:
        return group_results

    if not _check_gpu_for_embeddings():
        logger.warning("Эмбеддинги пропущены (GPU/модели недоступны).")
        return group_results

    from embeddings_engine import get_engine
    from database import DatabaseManager

    fmt = group_results[0].get("type", "pdf_text")

    try:
        db = DatabaseManager()
    except Exception:
        db = None

    path_to_id = {}
    already_text = set()
    already_image = set()
    
    if db:
        try:
            already_text = db.get_embedded_text_paths()
            already_image = db.get_embedded_image_paths()
            with db.conn.cursor() as cur:
                cur.execute("SELECT id, file_path FROM documents;")
                path_to_id = {row[1]: row[0] for row in cur.fetchall()}
        except Exception as e:
            logger.warning(f"Не удалось загрузить статус эмбеддингов из БД: {e}")

    engine = get_engine()
    emb_dir = cfg.EMBEDDINGS_DIR
    emb_dir.mkdir(parents=True, exist_ok=True)

    text_inputs, text_indices, text_names, text_doc_ids = [], [], [], []
    image_inputs, image_indices, image_names, image_doc_ids = [], [], [], []

    for i, d in enumerate(group_results):
        file_name = Path(d["source"]).name
        src_path = d["source"]
        doc_id = d.get("id") or path_to_id.get(src_path)
        tq = d.get("text_quality", "good")

        # Текстовый эмбеддинг (если есть текст)
        if tq in ("good", "poor") and d.get("text") and src_path not in already_text:
            text_inputs.append(d["text"])
            text_indices.append(i)
            text_names.append(file_name)
            if doc_id:
                text_doc_ids.append(doc_id)

        # Графический эмбеддинг (если есть изображение, независимо от текста)
        if d.get("image") and src_path not in already_image:
            image_inputs.append(d["image"])
            image_indices.append(i)
            image_names.append(file_name)
            if doc_id:
                image_doc_ids.append(doc_id)

    if not text_inputs and not image_inputs:
        logger.info("Эмбеддинги: нет новых данных для группы. Пропуск.")
        if db:
            try:
                db.close()
            except Exception:
                pass
        return group_results

    total_emb = len(text_inputs) + len(image_inputs)
    logger.info(f"Подэтап: эмбеддинги ({len(text_inputs)} текст, {len(image_inputs)} изображений)...")
    batch_size = cfg.EMB_BATCH_SIZE
    emb_done = 0

    if text_inputs:
        saved = 0
        for start in range(0, len(text_inputs), batch_size):
            if _is_stopped():
                break
            end = min(start + batch_size, len(text_inputs))
            batch_texts = text_inputs[start:end]
            batch_names = text_names[start:end]
            batch_indices = text_indices[start:end]
            text_emb = engine.get_text_embeddings(batch_texts, emb_dir, batch_names)
            
            if text_emb.size > 0:
                valid_doc_ids_t, valid_emb_t = [], []
                for local_i, global_i in enumerate(batch_indices):
                    emb = text_emb[local_i]
                    group_results[global_i]["text_embedding"] = emb
                    if np.any(emb):
                        if text_doc_ids:
                            valid_doc_ids_t.append(text_doc_ids[start + local_i])
                            valid_emb_t.append(emb)
                
                if db and valid_doc_ids_t:
                    try:
                        db.save_text_embeddings_batch(list(zip(valid_doc_ids_t, valid_emb_t)), fmt)
                        saved += len(valid_doc_ids_t)
                    except Exception as e:
                        logger.warning(f"Сохранение текстовых эмбеддингов: {e}")
            
            emb_done += len(batch_texts)
            logger.info(f"  Текстовые эмбеддинги: {end}/{len(text_inputs)}")

    if image_inputs:
        saved_img = 0
        for start in range(0, len(image_inputs), batch_size):
            if _is_stopped():
                break
            end = min(start + batch_size, len(image_inputs))
            batch_images = image_inputs[start:end]
            batch_names = image_names[start:end]
            batch_indices = image_indices[start:end]
            img_emb = engine.get_image_embeddings(batch_images, emb_dir, batch_names)
            
            if img_emb.size > 0:
                valid_doc_ids, valid_emb = [], []
                for local_i, global_i in enumerate(batch_indices):
                    emb = img_emb[local_i]
                    group_results[global_i]["image_embedding"] = emb
                    if np.any(emb):
                        if image_doc_ids:
                            valid_doc_ids.append(image_doc_ids[start + local_i])
                            valid_emb.append(emb)
                
                if db and valid_doc_ids:
                    try:
                        db.save_image_embeddings_batch(list(zip(valid_doc_ids, valid_emb)), fmt)
                        saved_img += len(valid_doc_ids)
                    except Exception as e:
                        logger.warning(f"Сохранение графических эмбеддингов: {e}")
                zero_count = len(batch_indices) - len(valid_doc_ids)
                if zero_count:
                    logger.warning(f"Пропущено {zero_count} нулевых image-эмбеддингов (файлы не найдены)")
            
            emb_done += len(batch_images)
            logger.info(f"  Графические эмбеддинги: {end}/{len(image_inputs)}")

    if db:
        try:
            db.close()
        except Exception:
            pass

    logger.info(f"Подэтап: эмбеддинги завершены ({emb_done}/{total_emb}).")
    return group_results


def _init_worker():
    """Инициализация воркера."""
    try:
        from extractors.office_com_extractor import _kill_zombie_office
        _kill_zombie_office('word')
        _kill_zombie_office('excel')
    except Exception as e:
        logger.exception(f"Failed to kill zombie Office processes in worker: {e}")


def _extract_worker(task):
    from extractors import process_file
    fp, cat = task
    return process_file(fp, cat)


def _run_extraction_pool(tasks, max_workers, total, group_name: str = "", global_offset: int = 0):
    """Запуск пула обработки для одной группы файлов."""
    results = []
    completed = 0
    moved_count = 0
    path_updates = []

    log_step = max(1, total // 10)

    try:
        with multiprocessing.Pool(processes=max_workers, initializer=_init_worker) as pool:
            for res in pool.imap_unordered(_extract_worker, tasks):
                if _is_stopped():
                    logger.warning("Остановка extraction pool по запросу.")
                    pool.terminate()
                    break
                
                if res is None:
                    continue
                
                completed += 1
                global_completed = global_offset + completed
                _get_progress().update(global_completed)
                
                try:
                    if res.get("error"):
                        src = Path(res["source"])
                        logger.warning(
                            f"[{global_completed}/{total}] {res['error']} — {src.name}"
                        )
                        
                        if src.exists():
                            if res["error"] == "Empty or extraction failed":
                                new_loc = file_processor.move_file_to_target(src, "failed_extraction")
                            else:
                                new_loc = file_processor.move_file_to_error(src)
                            
                            if new_loc:
                                path_updates.append((res["source"], str(new_loc)))
                                res["source"] = str(new_loc)
                            moved_count += 1
                    
                    else:
                        if completed % log_step == 0 or completed == total:
                            logger.info(f"[{global_completed}/{total}] OK")
                    
                    results.append(res)

                except Exception as e:
                    logger.error(f"[{global_completed}/{total}] Ошибка обработки результата: {e}")
    except Exception as e:
        logger.error(f"Критическая ошибка пула процессов: {e}")

    if moved_count:
        logger.info(f"  [{group_name}] Перемещено в ошибки: {moved_count} файлов")
    
    return results, path_updates


def _write_extraction_to_db(results: List[Dict], path_updates: List[tuple]) -> None:
    """Запись результатов извлечения в БД."""
    if not results:
        return
    
    try:
        from database import DatabaseManager
        db = DatabaseManager()
        
        for old_path, new_path in path_updates:
            try:
                db.update_file_path(old_path, new_path)
            except Exception as e:
                logger.warning(f"DB path update failed: {e}")
        
        extract_batch = [
            (r["source"],
             r.get("text", "")[:5000] if r.get("text") else None,
             r.get("image", ""),
             r.get("error", ""))
            for r in results
        ]
        db.update_extraction_batch(extract_batch)
        db.close()
    except Exception as e:
        logger.error(f"Запись извлечения в БД: {e}")


def _load_format_docs_for_embedding(file_type: str, sorted_data: List[Dict], has_extraction: bool) -> List[Dict]:
    """Готовит документы формата для построения эмбеддингов."""
    docs: List[Dict] = []

    if not has_extraction:
        sdir = _format_sorted_dir(file_type)
        if sdir and sdir.exists():
            for f in sdir.rglob("*"):
                if f.is_file():
                    docs.append({
                        "source": str(f), "type": file_type,
                        "text": None, "image": str(f), "text_quality": "none",
                    })
        return docs

    try:
        from database import DatabaseManager
        db = DatabaseManager()
        
        with db.conn.cursor() as cur:
            cur.execute("""
                SELECT id, file_path, text, image_path
                FROM documents
                WHERE format_type = %s AND (
                    (text IS NOT NULL AND text != '') OR image_path IS NOT NULL
                )
            """, (file_type,))
            
            for doc_id, fp, text, image in cur.fetchall():
                tq = "good" if (text and text.strip()) else "none"
                docs.append({
                    "id": doc_id, "source": fp, "type": file_type,
                    "text": text, "image": image or "", "text_quality": tq,
                })
        db.close()
    except Exception as e:
        logger.warning(f"Не удалось загрузить документы формата {file_type} для эмбеддингов: {e}")
    
    return docs


def _format_sorted_dir(file_type: str) -> Optional[Path]:
    """Директория Sorted/{format} для формата."""
    rel = cfg.FORMAT_TARGETS.get(file_type)
    if rel:
        return cfg.ROOT / rel
    return cfg.TARGETS.get(file_type)


def _delete_npy_for_files(file_paths: List[str]) -> None:
    """Удаляет .npy эмбеддинги для указанных файлов."""
    emb_dir = cfg.EMBEDDINGS_DIR
    if not emb_dir.exists():
        return
    
    removed = 0
    for fp in file_paths:
        stem = Path(fp).stem
        for suffix in (f"{stem}_text.npy", f"{stem}_image.npy"):
            p = emb_dir / suffix
            if p.exists():
                try:
                    p.unlink()
                    removed += 1
                except Exception:
                    pass
    
    if removed:
        logger.debug(f"Удалено .npy эмбеддингов: {removed}")


def _delete_embeddings_db(db, file_paths: List[str], file_type: str = None) -> None:
    """Удаляет строки эмбеддингов в формат-специфичных таблицах и сбрасывает флаги."""
    with db.conn.cursor() as cur:
        from database import DatabaseManager as _db
        for emb_type in ("text", "image"):
            table = _db.get_embedding_table_name(file_type or "pdf_text", emb_type)
            if table:
                cur.execute(f"""
                    DELETE FROM {table} WHERE doc_id IN (
                        SELECT id FROM documents WHERE file_path = ANY(%s)
                    );
                """, (file_paths,))
        cur.execute("""
            UPDATE documents
            SET text_embedded = FALSE, image_embedded = FALSE, updated_at = NOW()
            WHERE file_path = ANY(%s)
        """, (file_paths,))
        db.conn.commit()


async def _run_extraction_substage(file_type: str, sorted_data: List[Dict]) -> None:
    """Подэтап A: извлечение текста/изображений для одного формата."""
    global _stage2_active, _stage2_busy
    
    grp = _find_group(file_type)
    if grp is None or not grp.get("has_extraction", True):
        return

    loop = asyncio.get_running_loop()
    _stage2_busy = True
    _stage2_active = (file_type, "extract")
    grp["extract_status"] = "running"
    _persist_stage2_state()

    try:
        # Чистый старт: откатываем любое частичное извлечение
        await loop.run_in_executor(None, _rollback_extraction, file_type, sorted_data)

        # Задачи берём из актуальной директории Sorted/{format}
        sdir = _format_sorted_dir(file_type)
        group_tasks = [(f, file_type) for f in sdir.rglob("*") if f.is_file()] if sdir and sdir.exists() else []
        group_size = len(group_tasks)
        grp["total"] = group_size

        logger.info(f"\n{'─' * 50}")
        logger.info(f"Извлечение: {file_type.upper()} ({group_size} файлов, {grp['workers']} воркеров)")
        logger.info(f"{'─' * 50}")

        _get_progress().begin_stage(2, f"Извлечение: {file_type}", group_size)
        start = time.time()

        results, path_updates = await loop.run_in_executor(
            None, _run_extraction_pool, group_tasks, grp["workers"], group_size, file_type.upper(), 0
        )

        await loop.run_in_executor(None, _write_extraction_to_db, results, path_updates)

        for r in results:
            r["type"] = file_type
            _stage2_results[r["source"]] = r

        elapsed = time.time() - start

        if _is_stopped():
            grp["extract_status"] = "stopped"
            grp["embed_status"] = "locked"
            logger.warning(f"Извлечение {file_type.upper()} остановлено.")
        else:
            ok = sum(1 for r in results if not r.get("error"))
            err = sum(1 for r in results if r.get("error"))
            grp["extract_status"] = "completed"
            grp["extract_ok"] = ok
            grp["extract_errors"] = err
            grp["extract_elapsed"] = round(elapsed, 1)
            
            # Разблокируем подэтап эмбеддингов
            if grp["embed_status"] in ("locked", "rolled_back", "stopped"):
                grp["embed_status"] = "pending"
            
            logger.info(f"  Извлечение {file_type.upper()} завершено за {elapsed:.1f}с (OK: {ok}, ошибок: {err})")
    finally:
        _persist_stage2_state()
        _stage2_active = None
        _stage2_busy = False


async def _run_embeddings_substage(file_type: str, sorted_data: List[Dict]) -> None:
    """Подэтап B: построение эмбеддингов для одного формата."""
    global _stage2_active, _stage2_busy
    
    grp = _find_group(file_type)
    if grp is None:
        return

    loop = asyncio.get_running_loop()
    _stage2_busy = True
    _stage2_active = (file_type, "embed")
    grp["embed_status"] = "running"
    _persist_stage2_state()

    try:
        # Чистый старт: откатываем любые частичные эмбеддинги
        await loop.run_in_executor(None, _rollback_embeddings, file_type, sorted_data)

        docs = await loop.run_in_executor(
            None, _load_format_docs_for_embedding, file_type, sorted_data, grp.get("has_extraction", True)
        )

        logger.info(f"\n{'─' * 50}")
        logger.info(f"Эмбеддинги: {file_type.upper()} ({len(docs)} документов)")
        logger.info(f"{'─' * 50}")

        _get_progress().begin_stage(2, f"Эмбеддинги: {file_type}", len(docs))
        start = time.time()

        docs = _embed_group_results(docs)

        for d in docs:
            d.setdefault("type", file_type)
            existing = _stage2_results.get(d["source"], {})
            existing.update(d)
            _stage2_results[d["source"]] = existing

        elapsed = time.time() - start

        if _is_stopped():
            grp["embed_status"] = "stopped"
            logger.warning(f"Эмбеддинги {file_type.upper()} остановлены.")
        else:
            text_emb = sum(1 for d in docs if d.get("text_embedding") is not None)
            img_emb = sum(1 for d in docs if d.get("image_embedding") is not None)
            grp["embed_status"] = "completed"
            grp["embed_ok"] = text_emb + img_emb
            grp["embed_errors"] = max(0, len(docs) - (text_emb + img_emb))
            grp["embed_elapsed"] = round(elapsed, 1)
            
            logger.info(f"  Эмбеддинги {file_type.upper()} завершены за {elapsed:.1f}с (текст: {text_emb}, vision: {img_emb})")
    finally:
        _persist_stage2_state()
        _stage2_active = None
        _stage2_busy = False


def _rollback_extraction(file_type: str, sorted_data: List[Dict]) -> int:
    """Откат подэтапа извлечения для формата."""
    from database import DatabaseManager
    
    # Всегда очищаем временные директории, независимо от наличия sorted_data
    import shutil
    for tmp_dir in cfg.EXTRA.values():
        if tmp_dir.exists():
            try:
                fcount = sum(1 for _ in tmp_dir.rglob("*") if _.is_file())
                shutil.rmtree(tmp_dir)
                tmp_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Очищена временная директория {tmp_dir.name} ({fcount} файлов)")
            except Exception as e:
                logger.warning(f"Не удалось очистить {tmp_dir}: {e}")

    rolled_back = 0
    
    format_files = [d for d in sorted_data if d.get("type") == file_type]
    if not format_files:
        logger.info(f"Нет файлов формата {file_type} для отката извлечения")
        for src in [k for k, d in _stage2_results.items() if d.get("type") == file_type]:
            _stage2_results.pop(src, None)
        return 0
    
    try:
        db = DatabaseManager()
    except Exception as e:
        logger.error(f"Не удалось подключиться к БД для отката: {e}")
        return 0
    
    try:
        file_paths = [d["source"] for d in format_files]
        
        with db.conn.cursor() as cur:
            # Удаляем эмбеддинги из формат-специфичных таблиц
            from database import DatabaseManager as _db
            for emb_type in ("text", "image"):
                table = _db.get_embedding_table_name(file_type, emb_type)
                if table:
                    cur.execute(f"""
                        DELETE FROM {table} WHERE doc_id IN (
                            SELECT id FROM documents WHERE file_path = ANY(%s)
                        );
                    """, (file_paths,))
            
            # Очищаем извлечение + флаги
            cur.execute("""
                UPDATE documents 
                SET text = NULL, image_path = NULL, error = NULL,
                    enriched_text = NULL, topic = NULL, doc_type = NULL, purpose = NULL,
                    stage_2_done = FALSE, stage_3_done = FALSE,
                    text_embedded = FALSE, image_embedded = FALSE,
                    updated_at = NOW()
                WHERE file_path = ANY(%s)
            """, (file_paths,))
            
            db.conn.commit()
            logger.info(f"Очищены данные извлечения и эмбеддингов для формата {file_type}")

        _delete_npy_for_files(file_paths)

        # Возвращаем файлы из FailedExtraction и ErrorFiles обратно в Sorted/{format}
        target_dir = _format_sorted_dir(file_type)
        if not target_dir:
            logger.error(f"Не найдена директория для формата {file_type}")
            return 0
        
        target_dir.mkdir(parents=True, exist_ok=True)

        with db.conn.cursor() as cur:
            cur.execute("""
                SELECT file_path FROM documents
                WHERE file_path = ANY(%s)
            """, (file_paths,))
            db_paths = [row[0] for row in cur.fetchall()]

        for db_path_str in db_paths:
            current_path = Path(db_path_str)
            current_parent = str(current_path.parent).lower().rstrip('\\')
            failed_dir = str(cfg.FAILED_EXTRACTION_DIR).lower().rstrip('\\')
            errors_dir = str(cfg.ERRORS_DIR).lower().rstrip('\\')

            if current_parent in (failed_dir, errors_dir):
                new_path = target_dir / current_path.name
                
                counter = 1
                original_new_path = new_path
                while new_path.exists():
                    stem = original_new_path.stem
                    ext = original_new_path.suffix
                    new_path = target_dir / f"{stem}_{counter}{ext}"
                    counter += 1

                try:
                    shutil.move(str(current_path), str(new_path))
                    
                    with db.conn.cursor() as cur:
                        cur.execute("DELETE FROM documents WHERE file_path = %s;", (str(current_path),))
                    
                    with db.conn.cursor() as cur:
                        cur.execute("""
                            INSERT INTO documents (file_path, format_type, created_at)
                            VALUES (%s, %s, NOW())
                            ON CONFLICT (file_path) DO UPDATE SET format_type = EXCLUDED.format_type;
                        """, (str(new_path), file_type))
                    
                    db.conn.commit()
                    rolled_back += 1
                    logger.debug(f"Возвращён файл: {current_path.name} -> {new_path}")
                except Exception as e:
                    logger.error(f"Не удалось вернуть файл {current_path.name}: {e}")

        logger.info(f"Откат извлечения формата {file_type}: возвращено {rolled_back} файлов")

    except Exception as e:
        logger.error(f"Ошибка при откате формата {file_type}: {e}")
    finally:
        try:
            db.close()
        except Exception:
            pass
    
    # Очищаем накопитель результатов для формата
    for src in [k for k, d in _stage2_results.items() if d.get("type") == file_type]:
        _stage2_results.pop(src, None)
    
    return rolled_back


def _rollback_embeddings(file_type: str, sorted_data: List[Dict]) -> int:
    """Откат ТОЛЬКО подэтапа эмбеддингов для формата."""
    from database import DatabaseManager

    # При откате эмбеддингов тоже чистим временные директории
    import shutil
    for tmp_dir in cfg.EXTRA.values():
        if tmp_dir.exists():
            try:
                fcount = sum(1 for _ in tmp_dir.rglob("*") if _.is_file())
                shutil.rmtree(tmp_dir)
                tmp_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Очищена временная директория {tmp_dir.name} ({fcount} файлов)")
            except Exception as e:
                logger.warning(f"Не удалось очистить {tmp_dir}: {e}")
    
    format_files = [d for d in sorted_data if d.get("type") == file_type]
    if not format_files:
        logger.info(f"Нет файлов формата {file_type} для отката эмбеддингов")
    
    file_paths = [d["source"] for d in format_files]

    try:
        db = DatabaseManager()
    except Exception as e:
        logger.error(f"Не удалось подключиться к БД для отката эмбеддингов: {e}")
        db = None
    
    if db:
        try:
            # Берём актуальные пути из БД (файлы могли переместиться)
            with db.conn.cursor() as cur:
                cur.execute("SELECT file_path FROM documents WHERE file_path = ANY(%s)", (file_paths,))
                db_paths = [row[0] for row in cur.fetchall()] or file_paths
            
            _delete_embeddings_db(db, db_paths, file_type)
            _delete_npy_for_files(db_paths)
            logger.info(f"Откат эмбеддингов формата {file_type}: очищены векторы и флаги")
        except Exception as e:
            logger.error(f"Ошибка при откате эмбеддингов формата {file_type}: {e}")
        finally:
            try:
                db.close()
            except Exception:
                pass
    else:
        _delete_npy_for_files(file_paths)

    # Чистим эмбеддинги в накопителе результатов, сохраняя извлечённый текст
    for d in _stage2_results.values():
        if d.get("type") == file_type:
            d.pop("text_embedding", None)
            d.pop("image_embedding", None)

    return len(file_paths)


async def _await_substage_selection() -> Optional[tuple]:
    """Ожидание выбора подэтапа (format_type, substage) от пользователя."""
    global _awaiting_substage_select, _substage_to_start, _substage_select_event, _pending_substage
    
    _awaiting_substage_select = True
    _substage_to_start = None
    _substage_select_event.clear()

    # Если есть отложенный запрос — используем его сразу
    if _pending_substage is not None:
        _substage_to_start = _pending_substage
        _pending_substage = None
        _awaiting_substage_select = False
        logger.info(f"Подэтап из отложенного запроса: {_substage_to_start}")
        return _substage_to_start

    logger.info("Ожидание выбора подэтапа этапа 2 через web-интерфейс...")

    while _substage_to_start is None and not _is_stopped():
        await asyncio.sleep(1.0)

    _awaiting_substage_select = False
    selected = _substage_to_start

    if _is_stopped():
        return None

    logger.info(f"Выбран подэтап: {selected}")
    return selected


def select_substage_to_start(format_type: str, substage: str) -> bool:
    """Запрос на запуск подэтапа из web-интерфейса."""
    global _substage_to_start, _substage_select_event, _pending_substage

    if substage not in ("extract", "embed"):
        return False

    grp = _find_group(format_type)
    if grp is None:
        return False

    # Валидация: эмбеддинги доступны только после завершения извлечения
    if substage == "embed":
        if grp.get("has_extraction", True) and grp.get("extract_status") != "completed":
            return False
    
    if substage == "extract" and not grp.get("has_extraction", True):
        return False

    if _awaiting_substage_select:
        # Пайплайн уже ждёт выбора — отправляем немедленно
        _substage_to_start = (format_type, substage)
        _substage_select_event.set()
    else:
        # Пайплайн ещё не дошёл до выбора — откладываем запрос
        _pending_substage = (format_type, substage)
        logger.info(f"Подэтап отложен (пайплайн ещё на этапе 1): {format_type}/{substage}")
    return True


async def run_stage_2_process_formats(sorted_data: List[Dict]) -> List[Dict]:
    """Этап 2: обработка форматов раздельными подэтапами."""
    global _extraction_groups, _current_group_index, _stage2_results
    
    import time
    
    _stage2_results = {}

    logger.info(f"Файлов для обработки (из sorted_data): {len(sorted_data)}")

    # Если sorted_data пустой — загружаем файлы из директории Sorted напрямую
    if not sorted_data:
        logger.info("sorted_data пустой, загружаю файлы из директории Sorted...")
        sorted_files = []
        
        for format_type, target_rel in cfg.FORMAT_TARGETS.items():
            target_dir = cfg.ROOT / target_rel
            if target_dir.exists():
                for f in target_dir.rglob("*"):
                    if f.is_file():
                        sorted_files.append({"source": str(f), "type": format_type})
        
        logger.info(f"Загружено {len(sorted_files)} файлов из директории Sorted")
        sorted_data = sorted_files

    # Строим/восстанавливаем группы с под-статусами
    _prepare_extraction_groups(sorted_data)

    if not _extraction_groups:
        logger.warning("Нет групп форматов для обработки на этапе 2.")
        return list(_stage2_results.values())

    logger.info(f"Групп форматов: {len(_extraction_groups)}")
    for g in _extraction_groups:
        logger.info(
            f"  {g['type']}: {g['total']} файлов | извлечение={g['extract_status']} | эмбеддинги={g['embed_status']}"
        )

    stage_start_time = time.time()

    # Основной цикл: ждём выбор подэтапа от пользователя
    while True:
        if _is_stopped():
            logger.warning("Остановка пайплайна по запросу пользователя.")
            break

        # Завершение: у всех групп эмбеддинги в completed/skipped
        all_done = all(g["embed_status"] in ("completed", "skipped") for g in _extraction_groups)
        if all_done:
            logger.info("Все группы этапа 2 обработаны (эмбеддинги построены).")
            break

        selection = await _await_substage_selection()
        
        if selection is None or _is_stopped():
            logger.warning("Остановка пайплайна по запросу пользователя.")
            break

        file_type, substage = selection
        grp = _find_group(file_type)
        
        if grp is None:
            logger.warning(f"Неизвестный формат: {file_type}")
            continue

        if substage == "extract":
            if not grp.get("has_extraction", True):
                logger.warning(f"Формат {file_type} не имеет подэтапа извлечения.")
                continue
            
            _current_group_index = _extraction_groups.index(grp)
            await _run_extraction_substage(file_type, sorted_data)
        
        elif substage == "embed":
            if grp.get("has_extraction", True) and grp["extract_status"] != "completed":
                logger.warning(f"Эмбеддинги {file_type} недоступны: извлечение не завершено.")
                continue
            
            _current_group_index = _extraction_groups.index(grp)
            await _run_embeddings_substage(file_type, sorted_data)

    stage_elapsed = time.time() - stage_start_time
    logger.info(f"\nЭтап 2: цикл подэтапов завершён за {stage_elapsed:.1f} сек")

    # Итоговые данные для этапа 3 — из БД (с эмбеддингами), фоллбэк на накопитель
    final_results = []
    
    try:
        from database import DatabaseManager
        db = DatabaseManager()
        final_results = db.get_all_data_with_embeddings()
        db.close()
    except Exception as e:
        logger.warning(f"Не удалось загрузить данные из БД для возврата: {e}")
        final_results = list(_stage2_results.values())

    logger.info(f"Итого результатов этапа 2: {len(final_results)}")
    return final_results
