"""paths.py — пути файловой системы для исследовательских прогонов.

Структура:
    D:\\FileOrganizer\\SourceFiles\\                # источник, восстанавливается из эталона
    D:\\FileOrganizer\\SourceFiles_golden\\         # ЭТАЛОН источника — immutable snapshot
        *.xlsx, *.pdf, ...
    D:\\FileOrganizer\\Sorted_golden\\              # золотой выход Stage 1, общий на все прогоны
        PDF_Text\\
        Word_Docx\\
        ...
    D:\\FileOrganizer\\Experiments\\                  # все артефакты прогонов
        <run_id>\\                                   # один прогон
            Sorted\\                                  # ленивая копия подформата перед extract
                <format_target_name>\\                # напр. PDF_Tables_fin
            Clusters\\
                Cluster_1\\
                ...
            CentralDocuments\\
            report.txt
            pipeline_log.txt
"""
from pathlib import Path
from typing import Optional

from config import cfg

EXPERIMENTS_ROOT = cfg.ROOT / "Experiments"
SORTED_GOLDEN_ROOT = cfg.ROOT / "Sorted_golden"
SOURCE_GOLDEN_ROOT = cfg.ROOT / "SourceFiles_golden"


def experiments_root() -> Path:
    return EXPERIMENTS_ROOT


def exp_dir(run_id: str) -> Path:
    return EXPERIMENTS_ROOT / run_id


def exp_sorted_root(run_id: str) -> Path:
    return exp_dir(run_id) / "Sorted"


def exp_sorted_subformat_dir(run_id: str, subformat: str) -> Path:
    """Разрешает путь ленивой копии подформата внутри прогона.

    Имя поддиректории берётся из `cfg.FORMAT_TARGETS[subformat]`
    (только последний компонент, напр. `PDF_Tables_fin>/<...>`),
    чтобы не дублировать дерево `Sorted/...`.
    """
    rel = cfg.FORMAT_TARGETS.get(subformat)
    if rel is None:
        # fallback: использовать subformat как имя директории напрямую
        rel = subformat
    return exp_sorted_root(run_id) / rel


def exp_clusters_dir(run_id: str) -> Path:
    return exp_dir(run_id) / "Clusters"


def exp_central_dir(run_id: str) -> Path:
    return exp_dir(run_id) / "CentralDocuments"


def exp_report_path(run_id: str) -> Path:
    return exp_dir(run_id) / "report.txt"


def exp_log_path(run_id: str) -> Path:
    return exp_dir(run_id) / "pipeline_log.txt"


def gold_sorted_dir() -> Path:
    return SORTED_GOLDEN_ROOT


def gold_sorted_subformat_dir(subformat: str) -> Path:
    """Путь поддиректории в Sorted_golden относительно подформата."""
    rel = cfg.FORMAT_TARGETS.get(subformat)
    if rel is None:
        rel = subformat
    return SORTED_GOLDEN_ROOT / rel


def gold_source_dir() -> Path:
    """Путь к эталонному источнику (SourceFiles_golden/).

    Эталон создаётся при первом прогоне golden_stage1_run из D:\\FileOrganizer\\SourceFiles\\
    (если он был непуст), и затем на каждом следующем прогоне SourceFiles/ восстанавливается
    из SourceFiles_golden/ — это гарантирует детерминированный вход dag.
    """
    return SOURCE_GOLDEN_ROOT


def ensure_gold_source_root() -> Path:
    SOURCE_GOLDEN_ROOT.mkdir(parents=True, exist_ok=True)
    return SOURCE_GOLDEN_ROOT


def ensure_exp_dirs(run_id: str) -> Path:
    """Создаёт базовое дерево директорий прогона. Возвращает корень."""
    root = exp_dir(run_id)
    root.mkdir(parents=True, exist_ok=True)
    exp_sorted_root(run_id).mkdir(parents=True, exist_ok=True)
    exp_clusters_dir(run_id).mkdir(parents=True, exist_ok=True)
    exp_central_dir(run_id).mkdir(parents=True, exist_ok=True)
    return root


def ensure_gold_sorted_root() -> Path:
    SORTED_GOLDEN_ROOT.mkdir(parents=True, exist_ok=True)
    return SORTED_GOLDEN_ROOT


def remove_exp_dir(run_id: str) -> int:
    """Удаляет директорию прогона рекурсивно. Возвращает кол-во файлов."""
    import shutil
    root = exp_dir(run_id)
    if not root.exists():
        return 0
    fcount = sum(1 for _ in root.rglob("*") if _.is_file())
    shutil.rmtree(root, ignore_errors=False)
    return fcount