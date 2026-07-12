# variants/ — каталог вариантов обработки

Каждая гипотеза по обработке = отдельный подкаталог здесь. Airflow
варианты выбираются по Airflow Variable `variant.<stage>.<subformat>`
(по умолчанию `default`) или через override в `dag_run.conf`.

## Структура

```
variants/
├── stage1_sort/
│   ├── default/                  # золотой (текущий путь классификации)
│   │   └── entry.py
│   └── alternative_pdf_classifier_v1/   # гипотеза: больше подклассов PDF
│       └── entry.py
├── stage2_extract/
│   ├── pdf_text/
│   │   ├── default/              # PyMuPDF4LLM/fitz
│   │   │   └── entry.py
│   │   └── v2_regex/             # гипотеза: другая обработка
│   │       └── entry.py
│   ├── pdf_tables_fin/
│   │   └── default/
│   │       └── entry.py
│   ├── word_docx/
│   │   ├── default/              # текущий Aspose 8 воркеров
│   │   │   └── entry.py
│   │   ├── aspose_16w/           # гипотеза: больше воркеров
│   │   │   └── entry.py
│   │   └── pandoc/               # гипотеза: Pandoc вместо Aspose
│   │       └── entry.py
│   └── ...
├── stage2_summarize/
│   ├── pdf_tables_fin/
│   │   └── default/              # Qwen3.5-4B для финансового топика
│   │       └── entry.py
│   └── ...
├── stage2_embed/
│   └── default/                  # один вариант, разные параметры (batch/GPU device)
│       └── entry.py
└── stage3_cluster/
    ├── default_hdbscan_min95_sim85/
    │   └── entry.py
    └── alt_kmeans/               # гипотеза: другая кластеризация
        └── entry.py
```

## Контракт варианта

Каждый `entry.py` экспортирует ОДНУ функцию:

```python
def run(run_id: str, subformat: str, params: dict) -> dict:
    """Обрабатывает подформат для конкретного прогона.

    Args:
        run_id:    идентификатор прогона, напр. 'exp_20260712_143022_abcd'.
                   Схема PostgreSQL прогона: `exp_<run_id>`.
                   Каталог прогона: `D:\\FileOrganizer\\Experiments\\<run_id>\\`.
        subformat: подформата, напр. 'pdf_tables_fin'.
        params:    словарь параметров из Airflow (dag_run.conf['params'])
                   или CLI `--param key=value`. Значения — JSON-декодированные.

    Returns:
        {
            "status": "ok" | "error",
            "metrics": {...},         # опционально, для public.experiment_registry.metrics_jsonb
            "error":   "...",         # обязателен при status="error"
            "elapsed": float,         # опционально, иначе считает runner
        }
    """
```

## Что вариант может делать внутри `run()`

- Читать из БД-схемы прогона: использовать
  `from airflow.experiments.db import get_conn, set_search_path; set_search_path(conn, run_id)` —
  после этого unqualified `SELECT/INSERT INTO documents ...` идут в схему прогона.
- Читать/писать в каталог прогона: `from airflow.experiments import paths; paths.exp_dir(run_id)`,
  `paths.exp_sorted_subformat_dir(run_id, subformat)`, `paths.exp_clusters_dir(run_id)`.
- Логировать: `from logger_utils import logger; logger.info(...)`.
- Импортировать общие утилиты проекта (`extractors.*`, `pipelines.*`, `embeddings_engine.*`,
  `llm_summarizer.*` и т.д.) — они доступны из `sys.path` (PROJECT_ROOT уже добавлен runner'ом).
- Возвращать метрики — они попадут в `experiment_registry.metrics_jsonb`.

## Что вариант НЕ должен делать

- Мутирует `D:\\FileOrganizer\\SourceFiles\\` или
  `D:\\FileOrganizer\\Sorted_golden\\` — это immutable-источник.
  Для extract-подэтапа есть `paths.exp_sorted_subformat_dir(run_id, subformat)` —
  это приватная копия прогона, её можно двигать (move-to-failed и т.д.).
- Писать в `public.documents` — для прогона есть `exp_<run_id>.documents`,
  для золотого Stage 1 — `gold_stage1.documents`. `public.documents` остаётся
  для legacy-демо-режима и будет демонтирован в поздних блоках.
- Поднимать LLM/эмбеддинг-модели в каждом вызове —
  Airflow-pool `gpu` (slots=1) гарантирует один таск за раз; моделей
  по-прежнему загружает Airflow-вызов на Windows-хосте.

## Добавление новой гипотезы

1. Положить новый подкаталог `variants/<stage>/<subformat>/<variant_name>/entry.py`,
   реализующий контракт. Импортировать общие утилиты — приветствуется.
   Свою логику — никого не трогать.
2. Если хочется временно переключить default на новый вариант —
   выставить Airflow Variable `variant.<stage>.<subformat>=<variant_name>`.
   Или передать override через `dag_run.conf['overrides']`.
3. Trigger DAG `run_experiment` с note, описывающим гипотезу.
4. Анализ результатов — Airflow UI + `public.experiment_registry.metrics_jsonb`
   + файлы в `Experiments/<run_id>/Clusters/`.
5. Если хуже — `rollback_experiment` DAG по `run_id`, забываем.
6. Если лучше — оставляем как candidate. Когда принято решение —
   включить в `golden_pipeline` (правка DAG-файла + git commit с человеко-читаемым
   обоснованием).

## Заглушки Block A

`variants/*/.../entry.py` — заглушки, **ничего не делают**, возвращают
`{"status":"ok","metrics":{"stub":true}}`. Реальные тела:
- **Block D**: stage2_extract (по одному подформату за раз, после вашего ОК).
- **Block E**: stage2_summarize + topic_gate для финансовых подформатов.
- **Block F**: stage2_embed (nomic-embed).
- **Block G**: stage3_cluster (HDBSCAN) + move-to-clusters + report.