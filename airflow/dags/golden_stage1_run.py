"""DAG: golden_stage1_run — разовый полный пересчёт золотой Stage 1.

Запускается вручную через "Trigger DAG w/ config". Параметр:
    note (опц.): заметка о причине пересчёта золотой Stage 1.

Архитектура: Airflow работает в WSL2, а Stage 1 выполняется Windows-Python
через WSL-interop (BashOperator, без SSH). См. dags/_wsl_interop.py.

Что делает DAG:
    1. Синглт BashOperator-таска `golden_stage1_run` через WSL-interop вызывает
       `/mnt/d/VENV/LLM/Scripts/python.exe -m airflow.experiments.runner
        --stage stage1_sort --subformat all --variant default
        --run-id golden_stage1_<ts>` (cwd = проект на /mnt/d).

    2. Внутри runner'а variant `variants/stage1_sort/all/default/entry.py`
       вызывает `stages.stage1_sorting.run_stage_1_sort()` (существующий код
       без правок), чистит прошлые выходы, снимает snapshot в
       `gold_stage1.documents` + копирует `Sorted/` → `Sorted_golden/`.

    3. Записывает строку в `public.experiment_registry` (dag_name=
       'golden_stage1_run') с метриками прогона.

Параметр `--run-id` = `golden_stage1_<ts_nodash>` (Airflow macro).
Схему `exp_...` НЕ создаёт: прогон не изолируется, только audit-запись +
snapshot в gold_stage1.

После успеха:
    gold_stage1.documents == snapshot из public.documents
    D:\\FileOrganizer\\Sorted_golden\\ == копия D:\\FileOrganizer\\Sorted\\
    public.experiment_registry: новая строка status=success

Пул: extractors (Stage 1 не использует GPU).
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

from _wsl_interop import runner_bash_command


DEFAULT_ARGS = {
    "owner": "experiments",
    "depends_on_past": False,
    "retries": 0,
}

with DAG(
    dag_id="golden_stage1_run",
    default_args=DEFAULT_ARGS,
    description="Golden Stage 1: full re-sort + snapshot to gold_stage1 (WSL-interop)",
    schedule=None,
    start_date=datetime(2026, 7, 1),
    catchup=False,
    tags=["golden", "stage1"],
    params={
        "note": "",  # optional human-readable reason for re-running golden
    },
    dagrun_timeout=timedelta(hours=4),
) as dag:

    golden_stage1_run = BashOperator(
        task_id="golden_stage1_run",
        bash_command=runner_bash_command(
            stage="stage1_sort",
            subformat="all",
            variant="default",
            run_id="golden_stage1_{{ ts_nodash }}",
            params={"note": "{{ params.note }}"},
        ),
        pool="extractors",
    )