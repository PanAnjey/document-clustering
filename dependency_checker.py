# dependency_checker.py
# Pre-flight validation всех зависимостей пайплайна

import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

from config import cfg
from logger_utils import logger


def _check_com_app(prog_id: str, display_name: str) -> Tuple[bool, str]:
    try:
        import win32com.client
        import pythoncom
        pythoncom.CoInitialize()
        try:
            app = win32com.client.DispatchEx(prog_id)
            app.Quit()
            return True, f"{display_name} COM — доступен"
        except Exception as e:
            return False, f"{display_name} COM — НЕ ДОСТУПЕН ({e})"
        finally:
            pythoncom.CoUninitialize()
    except ImportError:
        return False, f"{display_name} COM — pywin32 не установлен"


def _check_import(module_name: str, pip_name: str = None) -> Tuple[bool, str]:
    try:
        importlib.import_module(module_name)
        return True, f"{module_name} — OK"
    except ImportError:
        pkg = pip_name or module_name.split('.')[0]
        return False, f"{module_name} — НЕ УСТАНОВЛЕН (pip install {pkg})"


def _check_cli(path: str, name: str) -> Tuple[bool, str]:
    exe = Path(path)
    if exe.exists():
        return True, f"{name} — {exe}"
    try:
        subprocess.run([name, "--version"], capture_output=True, timeout=5)
        return True, f"{name} — найден в PATH"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, f"{name} — НЕ НАЙДЕН (ожидается: {path})"


def _check_java() -> Tuple[bool, str]:
    java_home = os.environ.get('JAVA_HOME', r"C:\Java\jdk-17")
    java_bin = Path(java_home) / "bin" / "java.exe"
    if not java_bin.exists():
        return False, f"Java — НЕ НАЙДЕН (JAVA_HOME={java_home})"
    try:
        result = subprocess.run([str(java_bin), "-version"], capture_output=True, text=True, timeout=5)
        ver = result.stderr.strip().split('\n')[0] if result.stderr else "unknown"
        return True, f"Java — {ver}"
    except Exception as e:
        return False, f"Java — ОШИБКА ({e})"


def _check_aspose_java() -> Tuple[bool, str]:
    jar = Path(r"D:\Yandex.Disk\Aspose\Aspose.Words for Java\aspose-words-20.3-jdk16.jar")
    lic = Path(r"D:\Yandex.Disk\Aspose\Aspose.Words for Java\Aspose.Total.lic")
    cls = Path(r"D:\Yandex.Disk\Aspose\Aspose.Words for Java\helpers\ExtractHelper.class")
    if not jar.exists():
        return False, f"Aspose.Words for Java — JAR не найден ({jar})"
    if not lic.exists():
        return False, "Aspose.Words for Java — лицензия не найдена"
    if not cls.exists():
        return False, "Aspose.Words for Java — ExtractHelper.class не найден"
    return True, "Aspose.Words for Java — OK (JAR + лицензия + ExtractHelper)"


def _check_gpu(device: str, label: str) -> Tuple[bool, str]:
    try:
        import torch
        if not torch.cuda.is_available():
            return False, f"{label} ({device}) — CUDA недоступна"
        count = torch.cuda.device_count()
        idx = int(device.split(':')[1])
        if idx >= count:
            return False, f"{label} ({device}) — устройство {idx} недоступно (всего {count})"
        name = torch.cuda.get_device_name(idx)
        return True, f"{label} ({device}) — {name}"
    except ImportError:
        return False, f"{label} ({device}) — torch не установлен"


DEPENDENCIES: List[dict] = [
    {"type": "import", "name": "olefile", "pip": "olefile"},
    {"type": "import", "name": "striprtf", "pip": "striprtf"},
    {"type": "import", "name": "docx", "pip": "python-docx"},
    {"type": "import", "name": "pytesseract", "pip": "pytesseract"},
    {"type": "import", "name": "fitz", "pip": "PyMuPDF"},
    {"type": "import", "name": "pymupdf4llm", "pip": "pymupdf4llm", "optional": True},
    {"type": "import", "name": "win32com.client", "pip": "pywin32"},
    {"type": "com", "prog_id": "Word.Application", "display": "Microsoft Word", "enabled_attr": "COM_ENABLED", "optional": True},
    {"type": "com", "prog_id": "Excel.Application", "display": "Microsoft Excel", "enabled_attr": "COM_ENABLED", "optional": True},
    {"type": "cli", "path_attr": "PANDOC_PATH", "name": "Pandoc"},
    {"type": "cli", "path_attr": "TESSERACT_PATH", "name": "Tesseract"},
    {"type": "java", "name": "Java"},
    {"type": "aspose_java", "name": "Aspose.Words for Java"},
    # GPU/модели проверяются отдельно (validate_gpu) перед этапом эмбеддингов,
    # а не при старте системы.
    {"type": "import", "name": "torch", "pip": "torch", "gpu_check": True},
    {"type": "import", "name": "transformers", "pip": "transformers", "gpu_check": True},
    {"type": "import", "name": "pandas", "pip": "pandas", "gpu_check": True},
    {"type": "import", "name": "psycopg2", "pip": "psycopg2-binary", "gpu_check": True},
    {"type": "import", "name": "numpy", "pip": "numpy", "gpu_check": True},
    {"type": "gpu", "device_attr": "EMB_GPU_DEVICE", "label": "GPU эмбеддингов", "gpu_check": True},
    {"type": "gpu", "device_attr": "LLM_GPU_DEVICE", "label": "GPU LLM", "gpu_check": True},
]


def _run_checks(deps: List[dict]) -> Tuple[bool, list]:
    all_ok = True
    results = []
    for dep in deps:
        enabled_attr = dep.get("enabled_attr")
        if enabled_attr and not getattr(cfg, enabled_attr, True):
            continue
        if dep["type"] == "import":
            ok, msg = _check_import(dep["name"], dep.get("pip"))
        elif dep["type"] == "cli":
            path = getattr(cfg, dep["path_attr"], "")
            ok, msg = _check_cli(path, dep["name"])
        elif dep["type"] == "gpu":
            device = getattr(cfg, dep["device_attr"], "")
            ok, msg = _check_gpu(device, dep["label"])
        elif dep["type"] == "com":
            ok, msg = _check_com_app(dep["prog_id"], dep["display"])
        elif dep["type"] == "java":
            ok, msg = _check_java()
        elif dep["type"] == "aspose_java":
            ok, msg = _check_aspose_java()
        else:
            ok, msg = False, f"Неизвестный тип проверки: {dep['type']}"

        optional = dep.get("optional", False)
        if ok:
            logger.info(f"  ✓ {msg}")
        elif optional:
            logger.warning(f"  ? {msg} (необязательно, есть fallback)")
        else:
            logger.error(f"  ✗ {msg}")
            all_ok = False
        results.append((ok, msg))
    return all_ok, results


def validate_base() -> bool:
    """Базовая pre-flight валидация (без GPU/моделей). Вызывается при старте.
    GPU и модели проверяются отдельно через validate_gpu() перед этапом
    эмбеддингов и кластеризации.
    """
    logger.info("=" * 50)
    logger.info("ПРОВЕРКА БАЗОВЫХ ЗАВИСИМОСТЕЙ")
    logger.info("=" * 50)

    base_deps = [d for d in DEPENDENCIES if not d.get("gpu_check")]
    all_ok, results = _run_checks(base_deps)

    logger.info("=" * 50)
    if all_ok:
        logger.info("Базовые зависимости в порядке. Запуск пайплайна.")
    else:
        logger.critical("Обнаружены отсутствующие критические зависимости. Исправьте и запустите снова.")
        for ok, msg in results:
            if not ok:
                print(f"  [FAIL] {msg}")
    logger.info("=" * 50)
    return all_ok


def validate_gpu() -> bool:
    """Валидация GPU и моделей (torch, transformers, GPU устройства).
    Вызывается перед началом этапа эмбеддингов/кластеризации, а не при старте.
    """
    logger.info("=" * 50)
    logger.info("ПРОВЕРКА GPU И МОДЕЛЕЙ (эмбеддинги/LLM)")
    logger.info("=" * 50)

    gpu_deps = [d for d in DEPENDENCIES if d.get("gpu_check")]
    all_ok, results = _run_checks(gpu_deps)

    logger.info("=" * 50)
    if all_ok:
        logger.info("GPU и модели доступны.")
    else:
        logger.warning("Не все GPU/модели доступны — эмбеддинги/LLM могут быть пропущены.")
    logger.info("=" * 50)
    return all_ok


def validate_all() -> bool:
    """Полная валидация (базовые + GPU). Для обратной совместимости."""
    base_ok = validate_base()
    gpu_ok = validate_gpu()
    return base_ok


def validate_or_exit():
    if not validate_all():
        sys.exit(1)
