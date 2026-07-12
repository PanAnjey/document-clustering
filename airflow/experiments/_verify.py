"""Проверка: что Block A создал в БД и на диске.

Запуск:
    python -m airflow.experiments._verify
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from airflow.experiments import (
    db, paths, registry, variant_loader,
)


def main():
    print("=== Block A verification ===\n")

    print("[1] DB objects:")
    conn = db.get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT schema_name FROM information_schema.schemata
            WHERE schema_name IN ('public','gold_stage1')
            ORDER BY 1
        """)
        print("  schemas:", [r[0] for r in cur.fetchall()])

        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema='public'
              AND table_name IN ('experiment_registry','golden_state',
                                 'format_taxonomy','_experiments_migrations')
            ORDER BY 1
        """)
        print("  public tables created:", [r[0] for r in cur.fetchall()])

        cur.execute("SELECT count(*) FROM public.experiment_registry")
        print(f"  experiment_registry rows: {cur.fetchone()[0]}")
        cur.execute("SELECT id, git_commit, variants_jsonb FROM public.golden_state")
        print(f"  golden_state: {cur.fetchall()}")
        cur.execute("SELECT count(*) FROM public.format_taxonomy")
        print(f"  format_taxonomy rows: {cur.fetchone()[0]}")
        cur.execute("""
            SELECT format_type, family, summary_strategy, array_length(finance_topics, 1)
            FROM public.format_taxonomy
            ORDER BY array_length(finance_topics, 1) DESC NULLS LAST
            LIMIT 5
        """)
        print(f"  top 5 format_taxonomy (by finance_topics count):")
        for row in cur.fetchall():
            print(f"    {row}")
        cur.execute("SELECT count(*) FROM gold_stage1.documents")
        print(f"  gold_stage1.documents rows: {cur.fetchone()[0]}")
    conn.close()

    print("\n[2] Filesystem paths:")
    print(f"  experiments_root: {paths.experiments_root()}")
    print(f"  gold_sorted_dir: {paths.gold_sorted_dir()}")
    print(f"  exp_dir('exp_test'): {paths.exp_dir('exp_test')}")

    print("\n[3] Variant catalog:")
    print(f"  variants_root: {variant_loader.VARIANTS_ROOT}")
    print(f"  known subformats in stage2_extract:")
    for sf in variant_loader.list_subformats('stage2_extract'):
        vs = variant_loader.list_variants('stage2_extract', sf)
        print(f"    {sf}: variants={vs}")
    print(f"  stage1_sort subformats: {variant_loader.list_subformats('stage1_sort')}")
    print(f"  stage3_cluster subformats: {variant_loader.list_subformats('stage3_cluster')}")
    print(f"  stage2_summarize subformats: {variant_loader.list_subformats('stage2_summarize')}")

    print("\n[4] Resolve variant (defaults to 'default'):")
    print(f"  resolve('stage2_extract','pdf_tables_fin'): "
          f"{variant_loader.resolve_variant('stage2_extract', 'pdf_tables_fin')}")
    print(f"  resolve with override: "
          f"{variant_loader.resolve_variant('stage2_extract', 'pdf_tables_fin', overrides={'stage2_extract.pdf_tables_fin': 'aspose_16w'})}")

    print("\n[5] Load stub variant and run it:")
    run_fn = variant_loader.load_variant_module('stage2_extract', 'pdf_tables_fin', 'default')
    result = run_fn(run_id='exp_test_unused', subformat='pdf_tables_fin', params={'foo': 'bar'})
    print(f"  result: {result}")

    print("\n[6] Sandbox schema lifecycle test (create, clone, drop):")
    test_run_id = 'exp_test_lifecycle'
    if db.schema_exists(test_run_id):
        db.drop_exp_schema(test_run_id)
        print(f"  dropped stale schema: {test_run_id}")
    db.create_exp_schema(test_run_id, with_format_tables=True)
    print(f"  created schema: {test_run_id}")
    n = db.clone_gold_to_exp(test_run_id)
    print(f"  cloned {n} rows into {test_run_id}.documents")
    # sample format_type counts in cloned schema
    conn = db.get_conn()
    db.set_search_path(conn, test_run_id)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT format_type, count(*) FROM documents
            GROUP BY format_type ORDER BY 2 DESC LIMIT 5
        """)
        print(f"  top 5 format_type in {test_run_id}.documents:")
        for row in cur.fetchall():
            print(f"    {row}")
    # cleanup
    db.drop_exp_schema(test_run_id)
    print(f"  dropped schema: {test_run_id}")
    print(f"  exists now: {db.schema_exists(test_run_id)}")

    print("\n[OK] Block A — infrastructure ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())