# state/pipeline_state.py
# Версия: 5.0
# Дата: 2026-06-07
# Описание: Управление состоянием пайплайна (сохранение на диск).

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from config import cfg
from logger_utils import logger


class PipelineState:
    """Управление состоянием пайплайна с сохранением на диск."""
    
    def __init__(self):
        self.state_file = cfg.ROOT / "pipeline_state.json"
        self._state: Dict = {}
        self._load()
    
    def _load(self) -> None:
        """Загрузка состояния из файла."""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    self._state = json.load(f)
                logger.info(f"Состояние пайплайна загружено из {self.state_file}")
            except Exception as e:
                logger.warning(f"Ошибка загрузки состояния: {e}")
                self._state = {}
        else:
            self._state = {
                "created_at": time.time(),
                "last_updated": time.time(),
                "stages": {},
                "groups": {},
            }
    
    def _save(self) -> None:
        """Сохранение состояния в файл."""
        try:
            self._state["last_updated"] = time.time()
            
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self._state, f, indent=2, ensure_ascii=False)
            
            logger.debug(f"Состояние пайплайна сохранено в {self.state_file}")
        except Exception as e:
            logger.error(f"Ошибка сохранения состояния: {e}")
    
    def set_stage_status(self, stage_num: int, status: str, details: Optional[Dict] = None) -> None:
        """Установка статуса этапа."""
        self._state["stages"][f"stage_{stage_num}"] = {
            "status": status,
            "started_at": time.time(),
            "completed_at": time.time() if status == "completed" else None,
            "details": details or {},
        }
        self._save()
    
    def get_stage_status(self, stage_num: int) -> Optional[Dict]:
        """Получение статуса этапа."""
        return self._state["stages"].get(f"stage_{stage_num}")
    
    def is_stage_completed(self, stage_num: int) -> bool:
        """Проверка завершения этапа."""
        status = self.get_stage_status(stage_num)
        return status and status.get("status") == "completed"
    
    def save_stage2_groups(self, groups_data: Dict[str, Dict]) -> None:
        """Сохранение данных групп этапа 2."""
        self._state["groups"] = groups_data
        self._save()
    
    def get_stage2_groups(self) -> Dict[str, Dict]:
        """Получение данных групп этапа 2."""
        return self._state.get("groups", {})
    
    def clear_all(self) -> None:
        """Очистка всего состояния."""
        self._state = {
            "created_at": time.time(),
            "last_updated": time.time(),
            "stages": {},
            "groups": {},
        }
        self._save()
    
    def get_summary(self) -> Dict:
        """Получение сводки состояния пайплайна."""
        return {
            "created_at": self._state.get("created_at"),
            "last_updated": self._state.get("last_updated"),
            "stages": self._state.get("stages", {}),
            "groups_count": len(self._state.get("groups", {})),
        }


# Глобальный экземпляр (для совместимости с old code)
_pipeline_instance: Optional[PipelineState] = None


def get_pipeline_state() -> PipelineState:
    """Получение глобального экземпляра PipelineState."""
    global _pipeline_instance
    
    if _pipeline_instance is None:
        _pipeline_instance = PipelineState()
    
    return _pipeline_instance
