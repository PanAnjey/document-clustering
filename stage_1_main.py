# stage_1_main.py
# Версия: 2.1
# Дата: 2026-06-06
# Описание: Этап 1. Сортировка файлов, извлечение данных, LLM саммаризация,
#            генерация эмбеддингов и сохранение в БД.

import asyncio
import time
import concurrent.futures
from pathlib import Path
from typing import List, Dict

from config import cfg
from logger_utils import logger
from extractors import process_file
from embeddings_engine import get_engine
from database import DatabaseManager
from llm_summarizer import summarize_batch, check_llm_available

total_files = 0

def get_category(ext: str) -> str:
    ext = ext.lower()
    if ext in cfg.PDF_FORMATS: return "pdf"
    if ext in cfg.EXCEL_FORMATS: return "excel"
    if ext in cfg.WORD_FORMATS: return "word"
    if ext in cfg.IMAGE_FORMATS: return "image"
    if ext in cfg.XML_FORMATS: return "xml"
    return "unknown"

async def sort_files():
    global total_files
    logger.info("Stage 1.1: Sorting files...")

    if not cfg.SOURCE_DIR.exists():
        logger.critical(f"Source dir not found: {cfg.SOURCE_DIR}")
        return []

    files = list(cfg.SOURCE_DIR.rglob("*.*"))
    total_files = len(files)
    logger.info(f"Found {total_files} files.")

    sorted_list = []
    for f in files:
        if f.is_file():
            cat = get_category(f.suffix)
            if cat != "unknown":
                from extractors import move_file_to_target as _move_file_to_target
                moved = _move_file_to_target(f, cat)
                if moved: sorted_list.append((moved, cat))
            else:
                logger.warning(f"Unknown format skipped: {f.name}")
    return sorted_list

async def process_files(file_list: List[tuple]):
    logger.info("Stage 1.2: Extracting content...")

    loop = asyncio.get_running_loop()
    results = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.MAX_CONCURRENT_FILES) as executor:
        futures = [loop.run_in_executor(executor, process_file, path, cat) for path, cat in file_list]

        for future in asyncio.as_completed(futures):
            try:
                res = await future
                results.append(res)

                if res.get('error'):
                    src = Path(res['source'])
                    if src.exists():
                        logger.warning(f"Processing error: {res['error']} -> {src.name}")
                        from extractors import move_file_to_error as _move_file_to_error
                        _move_file_to_error(src)
            except Exception as e:
                logger.error(f"Worker failed: {e}")
    return results

async def summarize_data(processed_data: List[Dict]):
    if not cfg.LLM_ENABLED:
        logger.info("LLM summarization disabled. Skipping.")
        return processed_data
    if not check_llm_available():
        logger.warning("LM Studio not available. Skipping summarization.")
        return processed_data
    return await summarize_batch(processed_data)

async def save_to_database(processed_data: List[Dict]):
    logger.info("Stage 1.3: Generating embeddings and saving to DB...")

    engine = get_engine()
    db = DatabaseManager()

    valid_data = [d for d in processed_data if Path(d['source']).exists()]

    text_inputs = []
    text_names = []
    text_indices = []
    image_inputs = []
    image_names = []
    image_indices = []

    for i, d in enumerate(valid_data):
        if d.get('text'):
            text_inputs.append(d['text'])
            text_names.append(Path(d['source']).name)
            text_indices.append(i)
        if d.get('image'):
            image_inputs.append(d['image'])
            image_names.append(Path(d['source']).name)
            image_indices.append(i)

    emb_dir = cfg.EMBEDDINGS_DIR
    emb_dir.mkdir(parents=True, exist_ok=True)

    if text_inputs:
        logger.info(f"Generating TEXT embeddings for {len(text_inputs)} items...")
        text_embeddings = engine.get_text_embeddings(text_inputs, emb_dir, text_names)
        if text_embeddings.size > 0:
            for local_i, global_i in enumerate(text_indices):
                valid_data[global_i]['text_embedding'] = text_embeddings[local_i]

    if image_inputs:
        logger.info(f"Generating IMAGE embeddings for {len(image_inputs)} items...")
        image_embeddings = engine.get_image_embeddings(image_inputs, emb_dir, image_names)
        if image_embeddings.size > 0:
            for local_i, global_i in enumerate(image_indices):
                valid_data[global_i]['image_embedding'] = image_embeddings[local_i]

    db.save_all_embeddings(valid_data)
    db.close()
    logger.info("Stage 1 Complete. Embeddings saved to Database.")

async def main():
    from logger_utils import suppress_proactor_connection_errors
    suppress_proactor_connection_errors()

    start_time = time.time()
    try:
        sorted_files = await sort_files()
        if not sorted_files: return

        processed_data = await process_files(sorted_files)
        summarized_data = await summarize_data(processed_data)
        await save_to_database(summarized_data)

    except Exception as e:
        logger.exception("Stage 1 Critical Error")
    finally:
        logger.info(f"Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    asyncio.run(main())