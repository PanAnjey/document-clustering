"""registry.py — CRUD для public.experiment_registry.

Схема см. migrations/001_init.sql. Каждая строка — один прогон
эксперимента (DAG run). Airflow-задачи пишут сюда через SSH-вызовы
хелперов ниже.
"""
import json
from datetime import datetime
from typing import Any, Optional

from .db import get_conn


def insert_registry_row(
    *,
    run_id: str,
    dag_name: str,
    dag_run_id: Optional[str] = None,
    parent_run_id: Optional[str] = None,
    note: Optional[str] = None,
    params: Optional[dict] = None,
    git_commit: Optional[str] = None,
) -> None:
    """Создаёт строку прогона со статусом 'running'."""
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO public.experiment_registry
                (run_id, dag_run_id, parent_run_id, status, note,
                 params_jsonb, git_commit, dag_name, started_at)
            VALUES (%s, %s, %s, 'running', %s, %s, %s, %s, NOW())
            ON CONFLICT (run_id) DO UPDATE SET
                dag_name      = EXCLUDED.dag_name,
                dag_run_id    = EXCLUDED.dag_run_id,
                params_jsonb  = EXCLUDED.params_jsonb,
                status        = 'running',
                started_at    = NOW()
        """, (
            run_id,
            dag_run_id,
            parent_run_id,
            note,
            json.dumps(params or {}, ensure_ascii=False),
            git_commit,
            dag_name,
        ))
    conn.commit()


def update_registry_status(
    run_id: str,
    status: str,
    *,
    finished: bool = True,
) -> None:
    """Обновляет статус прогона.

    status: 'running' | 'success' | 'failed' | 'rolled_back'
    """
    if status not in ("running", "success", "failed", "rolled_back"):
        raise ValueError(f"invalid status: {status!r}")
    conn = get_conn()
    with conn.cursor() as cur:
        if finished and status != "running":
            cur.execute("""
                UPDATE public.experiment_registry
                SET status = %s, finished_at = NOW()
                WHERE run_id = %s
            """, (status, run_id))
        else:
            cur.execute("""
                UPDATE public.experiment_registry
                SET status = %s
                WHERE run_id = %s
            """, (status, run_id))
    conn.commit()


def update_registry_metrics(run_id: str, metrics: dict) -> None:
    """Дописывает/объединяет метрики прогона (jsonb merge).

    Рекомендую вызывать в самом конце прогона (task metrics_collector).
    """
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE public.experiment_registry
            SET metrics_jsonb = COALESCE(metrics_jsonb, '{}'::jsonb) || %s::jsonb
            WHERE run_id = %s
        """, (json.dumps(metrics, ensure_ascii=False), run_id))
    conn.commit()


def update_registry_note(run_id: str, note: str) -> None:
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE public.experiment_registry SET note = %s WHERE run_id = %s
        """, (note, run_id))
    conn.commit()


def update_registry_params(run_id: str, params: dict) -> None:
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE public.experiment_registry
            SET params_jsonb = COALESCE(params_jsonb, '{}'::jsonb) || %s::jsonb
            WHERE run_id = %s
        """, (json.dumps(params, ensure_ascii=False), run_id))
    conn.commit()


def get_registry_row(run_id: str) -> Optional[dict]:
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT run_id, dag_name, dag_run_id, parent_run_id,
                   started_at, finished_at, status, note,
                   params_jsonb, metrics_jsonb, git_commit
            FROM public.experiment_registry
            WHERE run_id = %s
        """, (run_id,))
        row = cur.fetchone()
        if not row:
            return None
        return {
            "run_id": row[0],
            "dag_name": row[1],
            "dag_run_id": row[2],
            "parent_run_id": row[3],
            "started_at": row[4],
            "finished_at": row[5],
            "status": row[6],
            "note": row[7],
            "params": row[8] if row[8] else {},
            "metrics": row[9] if row[9] else {},
            "git_commit": row[10],
        }


def list_registry_rows(
    *,
    limit: int = 50,
    order_by: str = "started_at DESC",
) -> list:
    """Возвращает последние прогоны (по умолчанию) для UI/сравнения."""
    if not order_by.replace(" ", "").isalnum() or ";" in order_by:
        raise ValueError(f"unsafe order_by clause: {order_by!r}")
    conn = get_conn()
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT run_id, dag_name, started_at, finished_at,
                   status, note, params_jsonb, metrics_jsonb
            FROM public.experiment_registry
            ORDER BY {order_by}
            LIMIT %s
        """, (limit,))
        cols = ["run_id", "dag_name", "started_at", "finished_at",
                "status", "note", "params", "metrics"]
        return [dict(zip(cols, row)) for row in cur.fetchall()]