"""airflow.experiments — инфраструктура исследовательских прогонов на Airflow.

Назначение пакета — предоставить тонкий слой изоляции экспериментов:

1. **Схемы PostgreSQL на эксперимент**: `exp_<run_id>` — изолированный набор
   таблиц `documents`, `documents_<fmt>`, `text_embeddings_<fmt>`,
   `image_embeddings_<fmt>` — всего того, что сейчас лежит в `public`.
   Хелпер `db.create_exp_schema(run_id)` создаёт схему и пустые таблицы,
   `clone_gold_to_exp(run_id)` копирует золотой выход Stage 1
   (метаданные файлов + format_type) из `gold_stage1.documents` в схему прогона.

2. **Каталог прогона на диске**: `D:\\FileOrganizer\\Experiments\\<run_id>\\` —
   сюда складываются `Clusters/`, `CentralDocuments/`, `report.txt`.
   Источник `D:\\FileOrganizer\\SourceFiles\\` и золотой `Sorted_golden\\`
   не мутируются между прогонами. Перед extract-подэтапом конкретного формата
   ленивая копия `Sorted_golden/<fmt>/` клонируется в
   `Experiments/<run_id>/Sorted/<fmt>/` (см. `paths.exp_sorted_subformat_dir`).

3. **Реестр прогонов**: общая таблица `public.experiment_registry` с
   заметками, параметрами, метриками каждого прогона (см. `registry.py`).

4. **Каталог вариантов**: `variants/<stage>/<subformat>/<variant_name>/entry.py`
   с единым контрактом `run(run_id, subformat, params) -> dict`. Выбор варианта
   для конкретного прогона — по Airflow Variable `variant.<stage>.<subformat>`
   (по умолчанию `default`) или override в `dag_run.conf` (см. `variant_loader.py`).

5. **Запуск варианта из Airflow**: SSHOperator на connection `windows_ssh`
   вызывает `python -m airflow.experiments.runner --stage ... --subformat ...
   --variant ... --run-id ...` (см. `runner.py`).

Пакет НЕ заменяет existing code (extractors/, pipelines/, stages/) —
в Блоках D-G существующие обработчики оборачиваются в варианты `default`.
Пакет НЕ трогает `web_server.py` — удаление web_server будет отдельным
предложением после того, как каркас run_experiment DAG заработает.
"""
from .db import (
    get_conn,
    create_exp_schema,
    drop_exp_schema,
    clone_gold_to_exp,
    set_search_path,
    SCHEMA_PREFIX,
)
from .paths import (
    experiments_root,
    exp_dir,
    exp_clusters_dir,
    exp_central_dir,
    exp_report_path,
    exp_log_path,
    exp_sorted_subformat_dir,
    gold_sorted_dir,
    ensure_exp_dirs,
)
from .registry import (
    insert_registry_row,
    update_registry_status,
    update_registry_metrics,
    get_registry_row,
    list_registry_rows,
)
from .variant_loader import (
    resolve_variant,
    load_variant_module,
    VariantNotFound,
)

__all__ = [
    "get_conn", "create_exp_schema", "drop_exp_schema", "clone_gold_to_exp",
    "set_search_path", "SCHEMA_PREFIX",
    "experiments_root", "exp_dir", "exp_clusters_dir", "exp_central_dir",
    "exp_report_path", "exp_log_path", "exp_sorted_subformat_dir",
    "gold_sorted_dir", "ensure_exp_dirs",
    "insert_registry_row", "update_registry_status", "update_registry_metrics",
    "get_registry_row", "list_registry_rows",
    "resolve_variant", "load_variant_module", "VariantNotFound",
]