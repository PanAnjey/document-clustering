# logger_utils.py
# Версия: 1.0
# Дата: 16.04.2026

import logging
import os
import sys
import io
import tempfile
from contextlib import contextmanager
from pathlib import Path
from config import cfg

# Сообщения MuPDF, которые нужно подавлять (полностью убирать из stderr).
# Прочие сообщения MuPDF (и любой другой вывод в stderr) сохраняются.
SUPPRESSED_STDERR_SUBSTRINGS = (
    "No common ancestor in structure tree",
)


class _FilteredStderr:
    """Обёртка над stderr: отбрасывает строки с подавляемыми подстроками,
    остальной вывод (включая прочие ошибки MuPDF) пропускает без изменений."""

    def __init__(self, target, substrings):
        self._target = target
        self._substrings = substrings

    def write(self, text):
        s = str(text)
        for sub in self._substrings:
            if sub in s:
                return
        self._target.write(text)

    def flush(self):
        try:
            self._target.flush()
        except Exception:
            pass

    def __getattr__(self, name):
        # Делегируем остальные атрибуты (isatty, encoding, fileno и т.п.)
        return getattr(self._target, name)


@contextmanager
def filter_mupdf_stderr(substrings=SUPPRESSED_STDERR_SUBSTRINGS):
    """Контекст-менеджер: на время блока фильтрует stderr, убирая только
    строки с указанными подстроками. Все прочие сообщения (включая другие
    ошибки MuPDF) выводятся в stderr без изменений.

    Перенаправляет C-level stderr (fd 2) во временный файл. После выхода
    из блока файл читается построчно, строки с подавляемыми подстроками
    отбрасываются, остальные выводятся в оригинальный stderr.

    Не использует threads/pipe — thread-safe и быстро, никакого оверхеда
    на создание потока для каждого вызова (было узким местом при 24K PDF).
    """
    old_fd = os.dup(2)
    with tempfile.TemporaryFile(mode="w+b", suffix=".stderr") as tmp:
        try:
            os.dup2(tmp.fileno(), 2)

            old_stderr = sys.stderr
            sys.stderr = _FilteredStderr(old_stderr, substrings)

            try:
                yield
            finally:
                sys.stderr = old_stderr

        finally:
            os.dup2(old_fd, 2)
            os.close(old_fd)

        # Воспроизводим отфильтрованный stderr
        tmp.seek(0)
        buf = b""
        for chunk_bytes in iter(lambda: tmp.read(4096), b""):
            buf += chunk_bytes
            while b"\n" in buf:
                line_bytes, buf = buf.split(b"\n", 1)
                line = line_bytes.decode("utf-8", errors="replace")
                if not any(sub in line for sub in substrings):
                    os.write(2, line_bytes + b"\n")
        if buf:
            line = buf.decode("utf-8", errors="replace")
            if not any(sub in line for sub in substrings):
                os.write(2, buf)

def setup_logger():
    log_dir = cfg.LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    
    formatter = logging.Formatter(cfg.LOG_FORMAT, datefmt=cfg.LOG_DATE_FORMAT)
    
    logger = logging.getLogger("FileOrganizer")
    logger.setLevel(logging.DEBUG)
    
    file_handler = logging.FileHandler(cfg.LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler(
        io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    )
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
    return logger

logger = setup_logger()


def suppress_proactor_connection_errors(loop=None):
    """Подавляет benign Windows socket errors в ProactorEventLoop.

    ConnectionResetError [10054] — удалённый хост закрыл соединение.
    WinError 64 — сброс сетевого стека (смена WiFi/VPN).
    Эти ошибки возникают в IocpProactor.accept / _call_connection_lost
    при штатной работе асинхронных сокетов на Windows.
    """
    import asyncio
    import sys

    if sys.platform != 'win32':
        return

    if loop is None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return

    original_handler = loop.get_exception_handler()

    _BENIGN_WINERRORS = {64, 10053, 10054}

    def _is_benign(exc, msg) -> bool:
        if isinstance(exc, (ConnectionResetError, ConnectionAbortedError)):
            return True
        if isinstance(exc, OSError):
            if getattr(exc, 'winerror', None) in _BENIGN_WINERRORS:
                return True
        return False

    def _proactor_exception_handler(loop, context):
        exc = context.get('exception')
        msg = context.get('message', '')
        if _is_benign(exc, msg):
            return
        if original_handler:
            original_handler(loop, context)
        else:
            loop.default_exception_handler(context)

    loop.set_exception_handler(_proactor_exception_handler)


def assess_text_quality(text) -> str:
    if not text or not text.strip():
        return "none"
    cleaned = text.strip()
    if len(cleaned) < 100:
        return "poor"
    alpha = sum(1 for c in cleaned if c.isalpha())
    total = len(cleaned)
    if total > 0 and (alpha / total) < 0.3:
        return "poor"
    return "good"

