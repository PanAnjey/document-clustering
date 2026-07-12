"""variants/_bootstrap.py — генератор stub-вариантов `default/entry.py`.

Запуск:
    python variants/_bootstrap.py            # создать недостающие stub'ы
    python variants/_bootstrap.py --force    # перезаписать все stub'ы
    python variants/_bootstrap.py --list     # показать что есть

Назначение: создавать `default/entry.py` для всех подформатов,
перечисленных в `cfg.FORMAT_TARGETS` (etапы stage2_extract, stage2_summarize
по тем, что summary_strategy != off) и для stage1_sort / stage3_cluster.
Не трогает уже существующие варианты с реальной реализацией (если файл
entry.py присутствует и --force не задан).
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import cfg  # noqa: E402

VARIANTS_ROOT = Path(__file__).resolve().parent

# Подформаты, исключаемые из Stage 2 (вне обработки экстракторов)
OUT_OF_STAGE2 = {"zip", "multimedia"}


STUB_TEMPLATE = '''"""Stub варианта `{variant}` для stage={stage} subformat={subformat}.

Эта заглушка ничего не делает. Реальная реализация появится в Блоках D-G.
Контракт см. `variants/README.md`.
"""
from logger_utils import logger


def run(run_id: str, subformat: str, params: dict) -> dict:
    logger.info(
        f"[stub] {{__name__}}: stage={stage} subformat={subformat} "
        f"run_id={{run_id}} params={{params!r}}"
    )
    return {{
        "status": "ok",
        "metrics": {{
            "stub": True,
            "stage": "{stage}",
            "subformat": "{subformat}",
            "variant": "{variant}",
        }},
    }}
'''


def _subformats_for_stage2_extract() -> list:
    """Все подформаты из FORMAT_TARGETS, кроме OUT_OF_STAGE2."""
    out = []
    for fmt in cfg.FORMAT_TARGETS.keys():
        if fmt in OUT_OF_STAGE2:
            continue
        out.append(fmt)
    return sorted(out)


def _subformats_for_stage2_summarize() -> list:
    """Подформаты с summary_strategy != off из public.format_taxonomy.

    Если таблица пуста/недоступна — fallback к финансовым подформатам.
    """
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=cfg.PG_HOST, port=cfg.PG_PORT,
            dbname=cfg.PG_DB, user=cfg.PG_USER, password=cfg.PG_PASSWORD,
        )
        with conn.cursor() as cur:
            cur.execute("""
                SELECT format_type FROM public.format_taxonomy
                WHERE summary_strategy != 'off'
                ORDER BY 1
            """)
            rows = cur.fetchall()
        conn.close()
        if rows:
            return [r[0] for r in rows]
    except Exception as e:
        print(f"[bootstrap] format_taxonomy unavailable: {e}")
    # Fallback
    return ["pdf_tables", "pdf_tables_fin", "pdf_tables_tech",
            "pdf_tables_contr", "pdf_tables_reports", "pdf_tables_other"]


def _subformats_for_stage2_embed() -> list:
    """Все подформаты, по которым строятся эмбеддинги (для нас — каждый
    подформат Stage 2). Совпадает с extract."""
    return _subformats_for_stage2_extract()


def _write_stub(path: Path, *, stage: str, subformat: str, variant: str, force: bool) -> str:
    if path.exists() and not force:
        return "skip"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = STUB_TEMPLATE.format(stage=stage, subformat=subformat, variant=variant)
    path.write_text(content, encoding="utf-8")
    return "wrote"


def main():
    ap = argparse.ArgumentParser(description="Bootstrap stub variants")
    ap.add_argument("--force", action="store_true",
                    help="перезаписать существующие stub'ы (по умолчанию skip)")
    ap.add_argument("--list", action="store_true",
                    help="показать все варианты и выйти")
    args = ap.parse_args()

    if args.list:
        for stage_dir in sorted(VARIANTS_ROOT.iterdir()):
            if not stage_dir.is_dir() or stage_dir.name.startswith("_") or stage_dir.name.startswith("."):
                continue
            for sf_dir in sorted(stage_dir.iterdir()):
                if not sf_dir.is_dir():
                    continue
                for v_dir in sorted(sf_dir.iterdir()):
                    if v_dir.is_dir() and (v_dir / "entry.py").exists():
                        print(f"  {stage_dir.name}/{sf_dir.name}/{v_dir.name}")
        return 0

    plan = []  # (stage, subformat, variant)
    plan.append(("stage1_sort", "all", "default"))
    for sf in _subformats_for_stage2_extract():
        plan.append(("stage2_extract", sf, "default"))
    for sf in _subformats_for_stage2_summarize():
        plan.append(("stage2_summarize", sf, "default"))
    for sf in _subformats_for_stage2_embed():
        plan.append(("stage2_embed", sf, "default"))
    plan.append(("stage3_cluster", "all", "default"))

    wrote = 0
    skipped = 0
    for stage, subformat, variant in plan:
        entry = VARIANTS_ROOT / stage / subformat / variant / "entry.py"
        action = _write_stub(entry, stage=stage, subformat=subformat,
                             variant=variant, force=args.force)
        if action == "wrote":
            print(f"  [wrote] {entry.relative_to(VARIANTS_ROOT)}")
            wrote += 1
        else:
            print(f"  [skip]  {entry.relative_to(VARIANTS_ROOT)}")
            skipped += 1

    print(f"\nDone: wrote={wrote} skipped={skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())