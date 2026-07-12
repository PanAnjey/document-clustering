# airflow_setup/

Конфигурация Airflow, которая не хранится в git (Variables, Connections, Pools)
— импортируется через CLI Airflow на хосте, где крутится scheduler.

## Pools

Импорт:

```bash
airflow pools import airflow_setup/pools.json
```

Доступные пулы:

| Имя пула      | Slots | Назначение |
|---------------|-------|------------|
| `gpu`         | 1     | Qwen3.5-4B (LLM суммаризация + cluster refinement) на `cuda:0`, nomic-embed на `cuda:1`. Один таск за раз. |
| `extractors`  | 1     | CPU-bound экстракторы (Aspose JVM / Pandoc / Tesseract / PyMuPDF). По вашей `_stage2_busy` конвенции: строго один подэтап за раз. |
| `clustering`  | 1     | HDBSCAN / кластеризация Stage 3 + LLM refinement кластеров. |
| `preflight`   | 4     | Pre-flight проверки зависимостей. |

## Variables

Импорт:

```bash
airflow variables import airflow_setup/variables_default.json
```

Все Variables имеют вид `variant.<stage>.<subformat>` = `default`.
Позже при добавлении новой гипотезы выставляется в имя нового варианта.

## Connections

Два connection'а нужно создать вручную (через UI Airflow или CLI):

### `windows_ssh` — Windows-машина с проектом

```bash
airflow connections add windows_ssh \
    --conn-type ssh \
    --conn-host <windows-host-ip> \
    --conn-login <windows-user> \
    --conn-password '<windows-password>' \
    --conn-port 22
```

Целевая машина должна:
- Иметь SSH-сервер (например, OpenSSH Server, встроенный в Windows 10/11).
- Иметь Python на `D:\VENV\LLM\Scripts\python.exe`.
- Иметь доступ к `D:\FileOrganizer\`, `D:\MODELS\Transformers\`,
  `D:\Yandex.Disk\Aspose\Aspose.Words for Java`. COM-Word фолбэк работает
  только на Windows — Linux-ноду использовать нельзя.

### `postgres_experiments` — PostgreSQL для experiment_registry

Не обязателен в Airflow (большинство DB-операций выполняется через SSHRunner
на Windows-цели, который сам подключается к `localhost:5432`),
но удобен для PythonOperator'ов, которые читают `experiment_registry`:

```bash
airflow connections add postgres_experiments \
    --conn-type postgres \
    --conn-host <postgres-host> \
    --conn-login postgres \
    --conn-password postgres \
    --conn-port 5432 \
    --conn-schema file_organizer_db
```

## Применение конфигурации (run at Airflow scheduler host)

```bash
# 1. Pools
airflow pools import airflow_setup/pools.json

# 2. Variables (default variant'ы)
airflow variables import airflow_setup/variables_default.json

# 3. Connections (создать вручную через UI/CLI, см. выше)

# 4. Smoke-test/smoke-cleanup pools/variables
airflow pools list
airflow variables list | head -20
```

## Где что лежит

```
airflow_setup/
├── pools.json
├── variables_default.json
└── connections.md            # этот файл
```

## Версионирование

- `pools.json` — слоты не меняются между экспериментами, только при
  принципиальном изменении топологии (например, добавлении второго GPU).
- `variables_default.json` — служит «базовым state». Если выставить другой
  вариант через UI Airflow (override), это локальное состояние schedul'ера
  и в git не коммитится. Текущий state можно выгрузить через
  `airflow variables export /tmp/varz.json` для аудита.
- `connections.md` — параметры коннекшенов чувствительные (пароли в
  `--conn-password`); хранить их в keytab/secret manager, не в git.