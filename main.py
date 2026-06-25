# main.py
# Версия: 5.0
# Дата: 2026-06-07
# Описание: Оркестратор пайплайна кластеризации (рефакторинг).
#           Разбит на функциональные модули для лучшей поддерживаемости.

from __future__ import annotations

import argparse
import asyncio
import time
import os
import sys
import threading
from pathlib import Path
from typing import List, Dict, Optional

from config import cfg
from logger_utils import logger
from pipeline_state import PipelineState
from state.progress_tracker import get_progress_tracker
from stages.stage1_sorting import run_stage_1_sort
from stages.stage2_processing import run_stage_2_process_formats
from stages.stage3_clustering import run_stage_3_cluster_refine

# ===== Глобальное состояние пайплайна =====
pipeline = PipelineState()
start_time: int = 0
stop_event = threading.Event()
progress = get_progress_tracker()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Парсинг командных аргументов."""
    parser = argparse.ArgumentParser(description="Pipeline for document clustering")
    parser.add_argument("--resume", action="store_true", help="Resume from last stage")
    parser.add_argument("--clear-all", action="store_true", help="Clear all data before run")
    parser.add_argument("--stage", type=int, choices=[1, 2, 3], help="Run specific stage only")
    return parser.parse_args(argv)


def wait_for_confirmation(stage_label: str) -> bool:
    """Ожидание подтверждения через web-интерфейс."""
    global _awaiting_confirmation, _current_stage_completed, _stage_confirmation_event
    
    _awaiting_confirmation = True
    _current_stage_completed = stage_label
    _stage_confirmation_event.clear()
    
    logger.info(f"Ожидание подтверждения для продолжения после этапа: {stage_label}")
    print(f"\n  >>> Ожидание подтверждения через web-интерфейс...")
    
    while not _stage_confirmation_event.is_set():
        if stop_event.is_set():
            _awaiting_confirmation = False
            _current_stage_completed = None
            return False
        _stage_confirmation_event.wait(timeout=1.0)
    
    _awaiting_confirmation = False
    _current_stage_completed = None
    logger.info(f"Подтверждение получено, продолжаем пайплайн.")
    return True


def confirm_next_stage():
    """Установка флага подтверждения (вызывается из web-интерфейса)."""
    global _stage_confirmation_event
    _stage_confirmation_event.set()


# ===== Глобальные события для web-интерфейса =====
_awaiting_confirmation = False
_current_stage_completed: Optional[str] = None
_stage_confirmation_event = threading.Event()


async def run_pipeline() -> int:
    """Entry point for web-server: runs full pipeline."""
    return await main(parse_args([]))


async def main(args: argparse.Namespace) -> int:
    """Основная функция запуска пайплайна."""
    global start_time, stop_event
    
    start_time = time.time()
    
    logger.info("=" * 60)
    logger.info("FILE ORGANIZER PIPELINE v5.0")
    logger.info("=" * 60)
    
    try:
        # Этап 1: Сортировка файлов
        if args.stage is None or args.stage == 1:
            logger.info("\n" + "=" * 60)
            logger.info("STAGE 1: SORTING FILES")
            logger.info("=" * 60)
            
            sorted_data = await run_stage_1_sort()
            
            if not sorted_data:
                logger.warning("Нет файлов для обработки на этапе 1.")
                # Проверяем, есть ли уже отсортированные файлы (предыдущий запуск)
                sorted_dirs = [cfg.ROOT / v for v in cfg.FORMAT_TARGETS.values()]
                has_sorted = any(d.exists() and any(d.iterdir()) for d in sorted_dirs)
                if not has_sorted:
                    return 1
                logger.info("Найдены отсортированные файлы от предыдущего запуска — пропускаю этап 1.")
            
            pipeline.mark_completed("stage_1_sort")
            pipeline.save_data("stage_1_sort", sorted_data or [])
            
            if args.stage == 1:
                logger.info(f"Этап 1 завершен. Файлов отсортировано: {len(sorted_data)}")
                return 0
        
        # Этап 2: Обработка форматов (извлечение + эмбеддинги)
        if args.stage is None or args.stage == 2:
            logger.info("\n" + "=" * 60)
            logger.info("STAGE 2: PROCESSING FORMATS")
            logger.info("=" * 60)
            
            processed_data = await run_stage_2_process_formats(sorted_data)
            
            pipeline.mark_completed("stage_2_process_formats")
            
            if args.stage == 2:
                logger.info(f"Этап 2 завершен. Обработано документов: {len(processed_data)}")
                return 0
        
        # Этап 3: Кластеризация и уточнение (LLM)
        if args.stage is None or args.stage == 3:
            logger.info("\n" + "=" * 60)
            logger.info("STAGE 3: CLUSTERING & REFINEMENT")
            logger.info("=" * 60)
            
            clustered_data = await run_stage_3_cluster_refine(processed_data)
            
            pipeline.mark_completed("stage_3_cluster_refine")
            
            if args.stage == 3:
                logger.info(f"Этап 3 завершен. Кластеров создано: {len(clustered_data)}")
                return 0
        
        # Общий тайминг
        total_elapsed = time.time() - start_time
        logger.info("\n" + "=" * 60)
        logger.info(f"PIPELINE COMPLETED IN {total_elapsed:.1f} seconds")
        logger.info("=" * 60)
        
        return 0
        
    except KeyboardInterrupt:
        stop_event.set()
        logger.warning("Пайплайн остановлен пользователем.")
        return 130
    except Exception as e:
        logger.critical(f"Критическая ошибка пайплайна: {e}")
        import traceback
        logger.exception(traceback.format_exc())
        return 1


if __name__ == "__main__":
    args = parse_args()
    exit_code = asyncio.run(main(args))
    sys.exit(exit_code)
