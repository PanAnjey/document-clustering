"""DAG: golden_stage1_run — разовый полный пересчёт золотой Stage 1.

Запускается вручную через "Trigger DAG w/ config". Параметр:
    note (опц.): заметка о причине пересчёта золотой Stage 1.

Что делает DAG:
    1. Синглт SSH-таска `golden_stage1_run` вызывает
       `python -m airflow.experiments.runner --stage stage1_sort
       --subformat all --variant default --run-id golden_stage1_<ts>`
       на Windows-цели (`windows_ssh` connection).

    2. Внутри runner'а variant `variants/stage1_sort/all/default/entry.py`
       вызывает `stages.stage1_sorting.run_stage_1_sort()` (существующий
       код без правок), чистит прошлые выходы, снимает snapshot в
       `gold_stage1.documents` + копирует `Sorted/` → `Sorted_golden/`.

    3. Записывает строку в `public.experiment_registry` (dag_name=
       'golden_stage1_run') с метриками прогона.

Параметр `--run-id`:
    `golden_stage1_<ts_nodash>` (Airflow macro). Не создаёт схему
    `exp_golden_stage1_...` — этот прогон НЕ изолируется в схеме; только
   _audit-запись в experiment_registry + snapshot в gold_stage1.

После успеха:
    gold_stage1.documents == [182k строк из 182k файлов SourceFiles/]
    D:\\FileOrganizer\\Sorted_golden\\ == [копия D:\\FileOrganizer\\Sorted\\]
    public.experiment_registry: новая строка status=success

Пулы:
    preflight (для DAG-level precheck, если будет добавлен)
    extractors — для золотой Stage 1

Не идёт в `gpu` пул: Stage 1 не использует GPU.
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator


DEFAULT_ARGS = {
    "owner": "experiments",
    "depends_on_past": False,
    "retries": 0,
}

PROJECT_DIR = r"D:\Yandex.Disk\PYTHON\NLTK\Кластеризация"
PYTHON_EXE = r"D:\VENV\LLM\Scripts\python.exe"

with DAG(
    dag_id="golden_stage1_run",
    default_args=DEFAULT_ARGS,
    description="Golden Stage 1: full re-sort of 182k source files + snapshot to gold_stage1",
    schedule=None,
    start_date=datetime(2026, 7, 1),
    catchup=False,
    tags=["golden", "stage1"],
    params={
        "note": "",  # optional human-readable reason for re-running golden
    },
    dagrun_timeout=timedelta(hours=4),
) as dag:

    golden_stage1_run = SSHOperator(
        task_id="golden_stage1_run",
        ssh_conn_id="windows_ssh",
        command=(
            f'cd /d "{PROJECT_DIR}" && '
            f'"{PYTHON_EXE}" -m airflow.experiments.runner '
            f'--stage stage1_sort --subformat all --variant default '
            f'--run-id "golden_stage1_{{{{ ts_nodash }}}}" '
            f'--param note="{{{{ params.note }}}}"'
        ),
        cmd_timeout=7200,
        pool="extractors",
        do_xcom_push=True,
    )