# pipeline_progress.py
# Global progress state for pipeline stages 2-4
# Thread-safe for concurrent access from web server + pipeline threads

import time
import threading


class ProgressState:
    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        with self._lock:
            self.stage = 0
            self.stage_label = ""
            self.completed = 0
            self.total = 0
            self.status = ""
            self._start_time = 0.0

    def begin_stage(self, stage: int, label: str, total: int):
        with self._lock:
            self.stage = stage
            self.stage_label = label
            self.total = total
            self.completed = 0
            self._start_time = time.time()
            self.status = ""

    def update(self, completed: int, status: str = ""):
        with self._lock:
            self.completed = completed
            if status:
                self.status = status

    def increment(self, status: str = ""):
        with self._lock:
            self.completed += 1
            if status:
                self.status = status

    def get_snapshot(self) -> dict:
        with self._lock:
            elapsed = time.time() - self._start_time if self._start_time > 0 else 0
            eta = None
            percentage = 0.0
            if self.total > 0:
                percentage = round(self.completed / self.total * 100, 1)
                if self.completed > 0 and self.completed < self.total:
                    speed = self.completed / elapsed if elapsed > 0 else 0
                    if speed > 0:
                        remaining = self.total - self.completed
                        eta = round(remaining / speed, 1)
            return {
                "stage": self.stage,
                "stage_label": self.stage_label,
                "completed": self.completed,
                "total": self.total,
                "percentage": percentage,
                "elapsed": round(elapsed, 1),
                "eta": eta,
                "status": self.status,
            }


progress = ProgressState()
