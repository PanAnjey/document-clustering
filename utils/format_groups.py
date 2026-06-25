# utils/format_groups.py
# Версия: 5.0
# Дата: 2026-06-07
# Описание: Управление группами форматов для этапа 2.

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

from config import cfg
from logger_utils import logger


class FormatGroupsManager:
    """Управление группами форматов для этапа 2."""
    
    def __init__(self):
        self._groups: List[Dict] = []
        self._current_group_index: int = -1
        self._substage_to_start: Optional[tuple] = None
        self._substage_select_event = asyncio.Event()
        self._awaiting_substage_select: bool = False
    
    def prepare_groups(self, sorted_data: List[Dict]) -> None:
        """Подготавливает группы форматов из данных."""
        self._groups = []
        self._current_group_index = -1
        
        # Группируем файлы по типу формата
        format_files = defaultdict(list)
        for d in sorted_data:
            format_files[d["type"]].append(d)
        
        # Создаем группы для каждого формата
        for format_type, files in format_files.items():
            has_extraction = self._format_has_extraction(format_type)
            
            # Определение количества воркеров
            if format_type.startswith("word_") or format_type == "rtf":
                workers = min(cfg.ASPOSE_WORKERS, cfg.MAX_CONCURRENT_FILES)
            elif format_type.startswith("excel_"):
                workers = min(cfg.COM_EXCEL_WORKERS, cfg.MAX_CONCURRENT_FILES)
            else:
                workers = cfg.MAX_CONCURRENT_FILES
            
            group = {
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
            }
            
            self._groups.append(group)
    
    def _format_has_extraction(self, format_type: str) -> bool:
        """Есть ли у формата подэтап извлечения текста/изображений."""
        EMBEDDINGS_ONLY_FORMATS = set()
        return format_type not in EMBEDDINGS_ONLY_FORMATS
    
    def find_group(self, format_type: str) -> Optional[Dict]:
        """Поиск группы по типу формата."""
        for g in self._groups:
            if g["type"] == format_type:
                return g
        return None
    
    @property
    def groups(self) -> List[Dict]:
        """Получение всех групп."""
        return self._groups
    
    @property
    def current_group_index(self) -> int:
        """Текущий индекс группы."""
        return self._current_group_index
    
    @current_group_index.setter
    def current_group_index(self, value: int) -> None:
        self._current_group_index = value
    
    async def await_substage_selection(self) -> Optional[tuple]:
        """Ожидание выбора подэтапа от пользователя."""
        self._awaiting_substage_select = True
        self._substage_to_start = None
        self._substage_select_event.clear()

        logger.info("Ожидание выбора подэтапа этапа 2 через web-интерфейс...")

        while self._substage_to_start is None:
            await asyncio.sleep(1.0)

        self._awaiting_substage_select = False
        selected = self._substage_to_start
        
        logger.info(f"Выбран подэтап: {selected}")
        return selected
    
    def select_substage(self, format_type: str, substage: str) -> bool:
        """Запрос на запуск подэтапа из web-интерфейса."""
        if substage not in ("extract", "embed"):
            return False
        
        if not self._awaiting_substage_select:
            return False

        grp = self.find_group(format_type)
        
        if grp is None:
            return False

        # Валидация: эмбеддинги доступны только после завершения извлечения
        if substage == "embed":
            if grp.get("has_extraction", True) and grp.get("extract_status") != "completed":
                return False
        
        if substage == "extract" and not grp.get("has_extraction", True):
            return False

        self._substage_to_start = (format_type, substage)
        self._substage_select_event.set()
        return True
    
    def get_summary(self) -> List[Dict]:
        """Получение сводки по всем группам."""
        summary = []
        
        for g in self._groups:
            summary.append({
                "type": g["type"],
                "total": g["total"],
                "extract_status": g["extract_status"],
                "embed_status": g["embed_status"],
                "extract_ok": g["extract_ok"],
                "extract_errors": g["extract_errors"],
                "embed_ok": g["embed_ok"],
                "embed_errors": g["embed_errors"],
            })
        
        return summary


# Глобальный экземпляр (для совместимости с old code)
_format_groups_manager: Optional[FormatGroupsManager] = None


def get_format_groups_manager() -> FormatGroupsManager:
    """Получение глобального экземпляра FormatGroupsManager."""
    global _format_groups_manager
    
    if _format_groups_manager is None:
        _format_groups_manager = FormatGroupsManager()
    
    return _format_groups_manager


# Функции для совместимости с old code
def select_substage_to_start(format_type: str, substage: str) -> bool:
    """Запрос на запуск подэтапа из web-интерфейса."""
    manager = get_format_groups_manager()
    return manager.select_substage(format_type, substage)
