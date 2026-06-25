# state/progress_tracker.py
# Версия: 5.0
# Дата: 2026-06-07
# Описание: Отслеживание прогресса пайплайна.

from __future__ import annotations

import time
from typing import Dict, Optional
from logger_utils import logger


class ProgressTracker:
    """Отслеживание прогресса выполнения пайплайна."""
    
    def __init__(self):
        self._current_stage: int = 0
        self._stage_label: str = ""
        self._total_items: int = 0
        self._completed_items: int = 0
        self._start_time: float = 0.0
        self._stage_start_time: float = 0.0
    
    def begin_stage(self, stage_num: int, label: str, total_items: int) -> None:
        """Начало этапа."""
        self._current_stage = stage_num
        self._stage_label = label
        self._total_items = total_items
        self._completed_items = 0
        self._start_time = time.time()
        self._stage_start_time = time.time()
        
        logger.info(f"Начало этапа {stage_num}: {label} ({total_items} элементов)")
    
    def update(self, completed: int) -> None:
        """Обновление прогресса."""
        self._completed_items = completed
        
        if self._total_items > 0:
            percentage = (self._completed_items / self._total_items) * 100
            
            # Логирование каждые 10%
            if int(percentage) % 10 == 0 and int(percentage) != int((percentage - 5)) % 10:
                elapsed = time.time() - self._stage_start_time
                
                logger.info(
                    f"Прогресс {self._stage_label}: "
                    f"{self._completed_items}/{self._total_items} ({percentage:.1f}%), "
                    f"время: {elapsed:.1f}s"
                )
    
    def complete_stage(self) -> Dict:
        """Завершение этапа."""
        elapsed = time.time() - self._stage_start_time
        
        logger.info(
            f"Этап {self._current_stage} завершен: "
            f"{self._completed_items}/{self._total_items}, "
            f"время: {elapsed:.1f}s"
        )
        
        result = {
            "stage": self._current_stage,
            "label": self._stage_label,
            "total": self._total_items,
            "completed": self._completed_items,
            "duration": elapsed,
            "success": self._completed_items == self._total_items,
        }
        
        return result
    
    def get_current_progress(self) -> Dict:
        """Получение текущего прогресса."""
        if self._total_items > 0:
            percentage = (self._completed_items / self._total_items) * 100
        else:
            percentage = 0
        
        elapsed = time.time() - self._start_time
        
        return {
            "stage": self._current_stage,
            "label": self._stage_label,
            "total": self._total_items,
            "completed": self._completed_items,
            "percentage": percentage,
            "elapsed": elapsed,
        }


# Глобальный экземпляр (для совместимости с old code)
_progress_instance: Optional[ProgressTracker] = None


def get_progress_tracker() -> ProgressTracker:
    """Получение глобального экземпляра ProgressTracker."""
    global _progress_instance
    
    if _progress_instance is None:
        _progress_instance = ProgressTracker()
    
    return _progress_instance
