"""variant_loader.py — выбор и загрузка варианта обработки.

Контракт варианта (см. variants/README.md):
    - Путь: `variants/<stage>/<subformat>/<variant_name>/entry.py`
    - Сигнатура: def run(run_id: str, subformat: str, params: dict) -> dict
    - Возвращает: {
          "status": "ok" | "error",
          "metrics": {...},
          "error":   "..." (только при status="error"),
      }

Выбор варианта:
    1. Если `overrides` dict содержит ключ `<stage>.<subformat>`,
       используется это имя варианта.
    2. Иначе — Airflow Variable `variant.<stage>.<subformat>`
       (через airflow.models.Variable, импортируется лениво).
    3. Иначе — 'default'.

Валидация: каталог варианта должен существовать и entry.py должен
иметь callable `run`. Иначе — VariantNotFound.
"""
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Optional, Callable

from logger_utils import logger

VARIANT_NOT_FOUND_DEFAULT_MSG = (
    "Нет обработчика для нового подформата — добавьте вариант в каталог"
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VARIANTS_ROOT = _PROJECT_ROOT / "variants"


class VariantNotFound(Exception):
    """Подформату нет варианта, или вариант недоступен/битый."""


def _variants_path_for(stage: str, subformat: str, variant_name: str) -> Path:
    return VARIANTS_ROOT / stage / subformat / variant_name / "entry.py"


def _airflow_variable(name: str, default: Optional[str] = None) -> Optional[str]:
    """Ленивое чтение Variable из Airflow.

    Если Airflow не установлен / нет метаданных (например, вызов вне DAG),
    возвращает default.
    """
    try:
        from airflow.models import Variable  # type: ignore
        return Variable.get(name, default_var=default)
    except Exception:
        return default


def resolve_variant(
    stage: str,
    subformat: str,
    overrides: Optional[dict] = None,
) -> str:
    """Возвращает имя варианта (без загрузки модуля).

    Приоритет: overrides["<stage>.<subformat>"] > Airflow Variable >
    'default'.

    Валидация существования НЕ выполняется — для этого вызывайте
    load_variant_module().
    """
    key = f"{stage}.{subformat}"
    if overrides and key in overrides:
        return overrides[key]
    var_name = f"variant.{key}"
    val = _airflow_variable(var_name, default=None)
    if val:
        return val
    return "default"


def load_variant_module(
    stage: str,
    subformat: str,
    variant_name: str,
) -> Callable:
    """Импортирует entry.py варианта и возвращает callable `run`.

    Импорт выполняется по файлу (importlib.util.spec_from_file_location)
    с уникальным module-name, чтобы варианты с одинаковым именем
    (например, 'default' в разных подформатах) не конфликтовали в
    sys.modules.
    """
    entry_path = _variants_path_for(stage, subformat, variant_name)
    if not entry_path.exists():
        raise VariantNotFound(
            f"Variant entry not found: {entry_path}\n"
            f"  stage={stage} subformat={subformat} variant={variant_name}\n"
            f"  {VARIANT_NOT_FOUND_DEFAULT_MSG}"
        )

    # Валидация: subformat и variant_name — простые идентификаторы
    # (защита от path traversal).
    import re
    if not re.match(r"^[a-z0-9_]+$", subformat) or not re.match(r"^[a-z0-9_]+$", variant_name):
        raise VariantNotFound(f"Invalid subformat/variant name: {subformat!r}/{variant_name!r}")
    # stage может быть 'stage1_sort', 'stage2_extract', 'stage2_summarize',
    # 'stage2_embed', 'stage3_cluster' — тоже допускаем только идентификаторы.
    if not re.match(r"^[a-z0-9_]+$", stage):
        raise VariantNotFound(f"Invalid stage name: {stage!r}")

    mod_name = f"_variant_{stage}_{subformat}_{variant_name}"
    if mod_name in sys.modules:
        return sys.modules[mod_name].run

    spec = importlib.util.spec_from_file_location(mod_name, entry_path)
    if spec is None or spec.loader is None:
        raise VariantNotFound(f"Cannot load spec: {entry_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "run") or not callable(module.run):
        raise VariantNotFound(f"Variant has no callable `run(run_id, subformat, params)`: {entry_path}")

    return module.run


def list_variants(stage: str, subformat: str) -> list:
    """Возвращает список доступных вариантов для пары (stage, subformat)."""
    base = VARIANTS_ROOT / stage / subformat
    if not base.exists():
        return []
    variants = []
    for d in base.iterdir():
        if d.is_dir() and (d / "entry.py").exists():
            variants.append(d.name)
    return sorted(variants)


def list_subformats(stage: str) -> list:
    """Возвращает все подформаты, для которых есть хотя бы один вариант."""
    base = VARIANTS_ROOT / stage
    if not base.exists():
        return []
    sfs = []
    for d in base.iterdir():
        if d.is_dir() and any(c.is_dir() and (c / "entry.py").exists() for c in d.iterdir()):
            sfs.append(d.name)
    return sorted(sfs)