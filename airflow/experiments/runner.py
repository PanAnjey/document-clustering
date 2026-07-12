"""runner.py — CLI диспетчер вызова варианта из Airflow.

Airflow SSHOperator на connection `windows_ssh` вызывает этот модуль
на Windows-хосте проекта:

    cd /d "D:\\Yandex.Disk\\PYTHON\\NLTK\\Кластеризация" && ^
        "D:\\VENV\\LLM\\Scripts\\python.exe" -m airflow.experiments.runner ^
        --stage stage2_extract --subformat pdf_text ^
        --variant default --run-id exp_20260712_xxxx

Варианты лежат в `variants/<stage>/<subformat>/<variant>/entry.py`
и импортирует общие утилиты (db, paths, extractors/...) из проекта.

Действия runner'а:
    1. Импортирует вариант (variant_loader.load_variant_module).
    2. Загружает Airflow Variables overrides (если не заданы через --override).
    3. Вызывает variant.run(run_id, subformat, params).
    4. Замеряет elapsed, печатает JSON-сводку в stdout для Airflow XCom.
    5. В случае падения варианта печатает JSON со status=error в stdout и
       выходит с кодом 1 (Airflow пометит таску как failed).
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from logger_utils import logger  # noqa: E402

from . import variant_loader  # noqa: E402


def _parse_overrides(items):
    """Парсит --override key=value в dict. key — dotted (stage.subformat)."""
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--override must be key=value, got: {item!r}")
        k, v = item.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def _parse_params(items):
    """Парсит --param key=value в dict. Значение пытается JSON-декодироваться,
    иначе остаётся строкой."""
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--param must be key=value, got: {item!r}")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Диспетчер вызова варианта обработки из Airflow",
    )
    ap.add_argument("--stage", required=True,
                    help="stage1_sort | stage2_extract | stage2_summarize | stage2_embed | stage3_cluster")
    ap.add_argument("--subformat", required=True,
                    help="pdf_text | word_docx | pdf_tables_fin | ...")
    ap.add_argument("--variant", default=None,
                    help="имя варианта (default, если не задано)")
    ap.add_argument("--run-id", default=None,
                    help="идентификатор прогона, напр. exp_20260712_143022_abcd. "
                         "Не обязателен для --list-variants.")
    ap.add_argument("--override", action="append", default=[],
                    help="override варианта: stage.subformat=variant_name")
    ap.add_argument("--param", action="append", default=[],
                    help="параметр варианта: key=value (значение пытается JSON-парсится)")
    ap.add_argument("--list-variants", action="store_true",
                    help="листинг доступных вариантов для stage+subformat и выход")
    args = ap.parse_args(argv)

    if args.list_variants:
        variants = variant_loader.list_variants(args.stage, args.subformat)
        print(json.dumps({
            "stage": args.stage,
            "subformat": args.subformat,
            "variants": variants,
        }, ensure_ascii=False, indent=2))
        return 0

    if not args.run_id:
        ap.error("--run-id is required unless --list-variants is set")

    overrides = _parse_overrides(args.override)
    params = _parse_params(args.param)

    # Выбор варианта
    if args.variant:
        variant_name = args.variant
    else:
        variant_name = variant_loader.resolve_variant(
            args.stage, args.subformat, overrides=overrides
        )

    logger.info(
        f"[runner] stage={args.stage} subformat={args.subformat} "
        f"variant={variant_name} run_id={args.run_id} params={params!r}"
    )

    t0 = time.time()
    try:
        run_fn = variant_loader.load_variant_module(
            args.stage, args.subformat, variant_name
        )
    except variant_loader.VariantNotFound as e:
        # Осмысленная ошибка — для Airflow JSON в stdout
        print(json.dumps({
            "status": "error",
            "stage": args.stage,
            "subformat": args.subformat,
            "variant": variant_name,
            "run_id": args.run_id,
            "error_kind": "variant_not_found",
            "error": str(e),
        }, ensure_ascii=False))
        return 1

    try:
        result = run_fn(
            run_id=args.run_id,
            subformat=args.subformat,
            params=params,
        )
    except Exception as e:
        # Вариант упал — пишем JSON со status=error и traceback в stderr
        tb = traceback.format_exc()
        logger.error(f"[runner] variant crashed: {e}\n{tb}")
        print(json.dumps({
            "status": "error",
            "stage": args.stage,
            "subformat": args.subformat,
            "variant": variant_name,
            "run_id": args.run_id,
            "error": str(e),
            "elapsed": round(time.time() - t0, 2),
        }, ensure_ascii=False))
        print(tb, file=sys.stderr)
        return 1

    elapsed = round(time.time() - t0, 2)
    if not isinstance(result, dict):
        logger.warning(f"[runner] variant result is not dict: {type(result)!r}")
        result = {"status": "ok", "raw_result": str(result)}

    # Дополняем сводку
    out = {
        "status": result.get("status", "ok"),
        "stage": args.stage,
        "subformat": args.subformat,
        "variant": variant_name,
        "run_id": args.run_id,
        "elapsed": result.get("elapsed", elapsed),
        "metrics": result.get("metrics", {}),
        "error": result.get("error"),
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0 if out["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())