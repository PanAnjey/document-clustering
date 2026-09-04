"""Helper: команда BashOperator для запуска Windows-Python через WSL-interop.

Airflow работает нативно в WSL2 Ubuntu, а вся тяжёлая работа пайплайна
(GPU cuda:0/cuda:1, COM Word, Aspose JVM, venv D:\\VENV\\LLM, PostgreSQL)
выполняется на Windows. WSL-interop позволяет вызвать Windows .exe напрямую
по пути /mnt/d/... — без SSH, без сети, без авторизации.

Все DAG-и эксперимента используют `runner_bash_command()` для формирования
команды к `airflow.experiments.runner` (диспетчер вариантов). Так как DAG-файлы
читаются Airflow напрямую с диска в UTF-8, кириллица в PROJECT_DIR_WSL
безопасна (в отличие от передачи через PowerShell→wsl).

Замечание про `python -m airflow.experiments.runner` на Windows:
    В venv D:\\VENV\\LLM пакет apache-airflow НЕ установлен, поэтому `import
    airflow` резолвится в ЛОКАЛЬНУЮ папку проекта airflow/ (namespace package),
    и `airflow.experiments.runner` находится корректно, если cwd = PROJECT_DIR.
"""

# Путь к Windows-Python (venv пайплайна) — как его видит WSL через /mnt.
WIN_PYTHON = "/mnt/d/VENV/LLM/Scripts/python.exe"

# Корень проекта — как его видит WSL через /mnt (кириллица допустима в файле DAG).
PROJECT_DIR_WSL = "/mnt/d/Yandex.Disk/PYTHON/NLTK/Кластеризация"


def _q(s: str) -> str:
    """Оборачивает путь в двойные кавычки для bash."""
    return '"' + s + '"'


def runner_bash_command(
    stage: str,
    subformat: str,
    variant: str,
    run_id: str,
    params: dict | None = None,
) -> str:
    """Строит bash-команду запуска варианта через WSL-interop.

    Args:
        stage, subformat, variant: координаты варианта (см. variants/).
        run_id: идентификатор прогона (можно с Jinja-макросом, напр.
                'golden_stage1_{{ ts_nodash }}').
        params: dict доп. параметров --param key=value (значения могут
                содержать Jinja-выражения, напр. '{{ params.note }}').

    Returns:
        Строка bash_command для BashOperator. Значения Jinja Airflow отрендерит.
    """
    parts = [
        f'cd {_q(PROJECT_DIR_WSL)} &&',
        f'{_q(WIN_PYTHON)} -m airflow.experiments.runner',
        f'--stage {stage}',
        f'--subformat {subformat}',
        f'--variant {variant}',
        f'--run-id "{run_id}"',
    ]
    if params:
        for key, val in params.items():
            parts.append(f'--param {key}="{val}"')
    return " ".join(parts)


def script_bash_command(script_relpath: str, args: list[str] | None = None) -> str:
    """Строит bash-команду запуска произвольного .py скрипта проекта через WSL-interop.

    В отличие от runner_bash_command (которая запускает airflow.experiments.runner
    с variants framework), эта функция вызывает скрипт напрямую — полезно для
    автономных runners в stage2_scripts/, уже содержащих всю логику (pool, БД,
    откат) и не зависящих от variants/<stage>/<subformat>/<variant>/entry.py.

    Args:
        script_relpath: путь к скрипту относительно корня проекта
            (напр. 'stage2_scripts/stage2_excel_xlsx.py').
        args: список аргументов командной строки (напр. ['extract']).

    Returns:
        Строка bash_command для BashOperator. Значения Jinja Airflow отрендерит.
    """
    parts = [
        f'cd {_q(PROJECT_DIR_WSL)} &&',
        f'{_q(WIN_PYTHON)} {script_relpath}',
    ]
    if args:
        parts.extend(args)
    return " ".join(parts)