"""Применяет миграции airflow.experiments и снимает snapshot gold_stage1.

Использование:
    python -m airflow.experiments.migrations.apply              # применить все .sql
    python -m airflow.experiments.migrations.apply --snapshot  # дополнительно snapshot public.documents → gold_stage1
    python -m airflow.experiments.migrations.apply --list      # список применённых миграций
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from config import cfg  # noqa: E402
from .. import db  # noqa: E402

MIGRATIONS_DIR = Path(__file__).resolve().parent
APPLIED_LOG_TABLE = "public._experiments_migrations"


def _ensure_log_table(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {APPLIED_LOG_TABLE} (
                filename TEXT PRIMARY KEY,
                applied_at TIMESTAMP NOT NULL DEFAULT NOW(),
                sql_hash TEXT
            );
        """)
        conn.commit()


def _applied_set(conn) -> set:
    with conn.cursor() as cur:
        try:
            cur.execute(f"SELECT filename FROM {APPLIED_LOG_TABLE}")
            return {row[0] for row in cur.fetchall()}
        except Exception:
            return set()


def _hash_sql(sql: str) -> str:
    import hashlib
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()[:16]


def apply_migrations(conn, *, verbose: bool = True):
    _ensure_log_table(conn)
    applied = _applied_set(conn)
    sql_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not sql_files:
        print("[apply] .sql миграции не найдены.")
        return 0

    new_count = 0
    for fp in sql_files:
        if fp.name in applied:
            if verbose:
                print(f"[skip] {fp.name} (уже применена)")
            continue
        sql = fp.read_text(encoding="utf-8")
        with conn.cursor() as cur:
            try:
                cur.execute(sql)
                cur.execute(
                    f"INSERT INTO {APPLIED_LOG_TABLE} (filename, sql_hash) VALUES (%s, %s)",
                    (fp.name, _hash_sql(sql)),
                )
                conn.commit()
                new_count += 1
                if verbose:
                    print(f"[ok]   {fp.name}")
            except Exception as e:
                conn.rollback()
                print(f"[FAIL] {fp.name}: {e}", file=sys.stderr)
                raise
    return new_count


def snapshot_gold_stage1(conn, *, verbose: bool = True) -> int:
    """Снимает snapshot public.documents → gold_stage1.documents.

    Полностью заменяет содержимое gold_stage1.documents текущим снимком
    public.documents (поле file_path + format_type). Источник
    считается immutable (фиксированный набор 182k файлов), поэтому
    повторный вызов просто перезаписывает.
    """
    with conn.cursor() as cur:
        cur.execute("TRUNCATE TABLE gold_stage1.documents RESTART IDENTITY")
        cur.execute("""
            INSERT INTO gold_stage1.documents (file_path, format_type)
            SELECT file_path, format_type
            FROM public.documents
            WHERE format_type IS NOT NULL AND file_path IS NOT NULL
            ON CONFLICT (file_path) DO NOTHING
        """)
        cur.execute("SELECT count(*) FROM gold_stage1.documents")
        total = cur.fetchone()[0]
        conn.commit()
    if verbose:
        print(f"[snapshot] gold_stage1.documents: {total} rows")
    return total


def main():
    ap = argparse.ArgumentParser(description="Применить миграции airflow.experiments")
    ap.add_argument("--snapshot", action="store_true",
                    help="Снять snapshot public.documents → gold_stage1.documents")
    ap.add_argument("--list", action="store_true",
                    help="Показать применённые миграции и выйти")
    args = ap.parse_args()

    if args.list:
        conn = db.get_conn()
        applied = _applied_set(conn)
        conn.close()
        if not applied:
            print("Применённых миграций нет.")
        else:
            for name in sorted(applied):
                print(f"  {name}")
        return 0

    conn = db.get_conn()
    try:
        new_count = apply_migrations(conn)
        print(f"[apply] применено новых миграций: {new_count}")
        if args.snapshot:
            snapshot_gold_stage1(conn)
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())