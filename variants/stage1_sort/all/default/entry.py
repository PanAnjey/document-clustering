"""Stub варианта `default` для stage=stage1_sort subformat=all.

Эта заглушка ничего не делает. Реальная реализация появится в Блоках D-G.
Контракт см. `variants/README.md`.
"""
from logger_utils import logger


def run(run_id: str, subformat: str, params: dict) -> dict:
    logger.info(
        f"[stub] {__name__}: stage=stage1_sort subformat=all "
        f"run_id={run_id} params={params!r}"
    )
    return {
        "status": "ok",
        "metrics": {
            "stub": True,
            "stage": "stage1_sort",
            "subformat": "all",
            "variant": "default",
        },
    }
