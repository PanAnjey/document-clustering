# airflow.experiments

Инфраструктура исследовательских прогонов на Airflow.

Каждый прогон эксперимента =
- **DAG run** в Airflow с параметрами и заметкой заметьте
- **PostgreSQL-схема** `exp_<run_id>` с собственной копией таблиц
  `documents`, `documents_<fmt>`, `text_embeddings_<fmt>`,
  `image_embeddings_<fmt>` (см. `db.create_exp_schema`)
- **директория** `D:\FileOrganizer\Experiments\<run_id>\` с
  `Sorted/`, `Clusters/`, `CentralDocuments/`, `report.txt` (см. `paths.py`)
- **строка в** `public.experiment_registry` с заметкой, параметрами,
  метриками прогона (см. `registry.py`)

## Установка

Применить миграции и снять snapshot золотого Stage 1:

```powershell
cd D:\Yandex.Disk\PYTHON\NLTK\Кластеризация
& "D:\VENV\LLM\Scripts\python.exe" -m airflow.experiments.migrations.apply --snapshot
```

После этого в базе есть:
- `public.experiment_registry`, `public.golden_state`, `public.format_taxonomy`
- `public._experiments_migrations` (лог применённых .sql миграций)
- `gold_stage1.documents` — snapshot 182k файлов с их `format_type`

## Структура каталога

```
airflow/experiments/
├── __init__.py              # экспорт публичных хелперов
├── README.md                # этот файл
├── db.py                    # connection + schema helpers
├── paths.py                 # путевые хелперы файловой системы
├── registry.py              # CRUD для experiment_registry
├── variant_loader.py        # выбор варианта по Variable / overrides
├── runner.py                # CLI: python -m airflow.experiments.runner
└── migrations/
    ├── __init__.py
    ├── 001_init.sql         # DDL таблиц реестра + схема gold_stage1
    └── apply.py             # применяет .sql + опциональный snapshot
```

## Контракт варианта обработки

См. `variants/README.md`.

## Изоляция прогонов

- `D:\FileOrganizer\SourceFiles\` — **immutable**. Никогда не мутируется.
- `D:\FileOrganizer\Sorted_golden\` — золотой выход Stage 1, общий на все
  прогоны. Не мутируется между прогонами.
- `D:\FileOrganizer\Experiments\<run_id>\Sorted\<fmt>\` — ленивая копия
  подформата из `Sorted_golden` перед extract. Эту копию вариант extract
  может двигать (move-to-failed и т.д.) — SrcS остаётся нетронутым.

## Откаты

- **Откат одного прогона**: `drop_exp_schema(run_id)` + удаление
  `Experiments\<run_id>\` + UPDATE registry `status='rolled_back'`.
  Источник и `gold_stage1` не трогаются.
- **Частичный откат подформата** (делается внутри `rollback_stage(run_id, stage='stage2', subformat=...)`):
  удалить `Experiments\<run_id>\Sorted\<fmt>\`,
  `Experiments\<run_id>\Clusters\<fmt>\`, и truncate соответствующих строк
  в `exp_<run_id>.documents_<fmt>` / `text_embeddings_<fmt>` / etc.,
  затем пересоздать копию из Sorted_golden.

См. `airflow/dags/rollback_experiment.py` (Блок H, появится позже).

## Что Блок A НЕ делает

- Не запускает никаких реальных обработчиков. Вариантов `default/` в
  `variants/*/` — заглушки (возвращают `{"status":"ok","metrics":{}}`),
  реальные тела приедут в Блоках D-G.
- Не удаляет `web_server.py` и не трогает существующий код `stages/`,
  `extractors/`, `pipelines/`. Это позже.
- Не запускает Airflow DAGs. Их каркас — Блок C.