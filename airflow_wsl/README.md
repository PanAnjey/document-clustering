# Airflow нативно в WSL2 (Ubuntu) — установка и запуск

Стенд кластеризации: **Airflow работает в WSL2 Ubuntu**, а вся тяжёлая работа
(GPU cuda:0/cuda:1, COM Word, Aspose JVM, venv `D:\VENV\LLM`, PyMuPDF, Tesseract)
выполняется на **Windows** через **WSL-interop** — Airflow BashOperator напрямую
вызывает `/mnt/d/VENV/LLM/Scripts/python.exe` (без SSH, без сети, без авторизации).

```
┌──────────── WSL2 Ubuntu 26.04 ────────────┐
│  Airflow 2.10.5 (venv /opt/airflow-venv,   │
│    Python 3.12 через uv)                    │
│    - scheduler (nohup)                      │
│    - webserver :8080 (nohup)                │
│  PostgreSQL 18 (метаданные Airflow, systemd)│
│         │ BashOperator                       │
│         │ /mnt/d/VENV/LLM/Scripts/python.exe │
└─────────┼───────────────────────────────────┘
          │ WSL-interop (kernel binfmt)
          ▼
┌──── Windows (та же машина) ────┐
│  venv D:\VENV\LLM (torch cu132)│
│  PostgreSQL file_organizer_db   │
│  GPU / COM / Aspose             │
│  D:\FileOrganizer\ (данные)     │
└─────────────────────────────────┘
```

Доступ к UI: **http://localhost:8080** из браузера Windows (WSL2 localhost-forwarding), логин `admin` / `admin`.

---

## Что уже установлено (эта сессия)

1. **WSL2 + Ubuntu 26.04** (`wsl --install`, systemd включён через `/etc/wsl.conf`).
2. **uv** (`/root/.local/bin/uv`) + **Python 3.12.13** (Ubuntu несёт 3.14, слишком новый для Airflow).
3. **venv** `/opt/airflow-venv` с `apache-airflow==2.10.5` + провайдеры `ssh`, `postgres`
   (constraints для Python 3.12).
4. **PostgreSQL 18** в WSL (systemd-сервис `postgresql`), БД `airflow` / роль `airflow`.
5. **AIRFLOW_HOME** = `/opt/airflow`, конфиг в `/opt/airflow/airflow.env`:
   - LocalExecutor, метаданные в локальном Postgres,
   - `DAGS_FOLDER=/mnt/d/Yandex.Disk/PYTHON/NLTK/Кластеризация/airflow/dags` (проект на /mnt/d),
   - example-DAG-и выключены.
6. **pools** (gpu/extractors/clustering/preflight) + **62 variables** импортированы.
7. **admin**-пользователь Airflow создан.

## Запуск / остановка

Airflow запускается через nohup-скрипты (systemd-сервисы для webserver давали
краш-цикл на WSL — systemd глушил gunicorn-монитор через ~16с; foreground/nohup
работает стабильно).

```bash
# запуск (scheduler + webserver)
wsl -d Ubuntu -u root /opt/airflow/start_airflow.sh

# остановка
wsl -d Ubuntu -u root /opt/airflow/stop_airflow.sh
```

Логи: `/opt/airflow/logs/scheduler.out`, `/opt/airflow/logs/webserver.out`.

PostgreSQL стартует автоматически (systemd). Airflow scheduler/webserver —
запускать вручную скриптом после старта WSL (автозапуск — TODO, см. ниже).

## Проверка

```bash
# из WSL
curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8080/health   # 200
# из Windows PowerShell
(Invoke-WebRequest http://localhost:8080/health -UseBasicParsing).StatusCode  # 200
```

## Файлы в этом каталоге (копии рабочих, для воспроизводимости)

- `airflow.env` — shell env (source в скриптах и интерактивно). Живёт в `/opt/airflow/airflow.env`.
- `start_airflow.sh` / `stop_airflow.sh` — запуск/остановка (nohup). Живут в `/opt/airflow/`.
- `airflow.systemd.env` — env для systemd (оставлен на случай возврата к systemd).
- `airflow-scheduler.service` / `airflow-webserver.service` — systemd-юниты
  (НЕ используются из-за краш-цикла webserver; оставлены для справки/доработки).

## Как это соединяется с проектом

DAG-и лежат в `<проект>/airflow/dags/` (на /mnt/d, читаются Airflow напрямую).
Ключевой хелпер — `airflow/dags/_wsl_interop.py`: строит команду BashOperator
`cd "<проект>" && "/mnt/d/VENV/LLM/Scripts/python.exe" -m airflow.experiments.runner ...`.

На Windows-venv пакет apache-airflow НЕ установлен, поэтому `import airflow`
там резолвится в локальную папку проекта `airflow/` (namespace package), и
`airflow.experiments.runner` находится корректно при cwd = корень проекта.

## TODO: автозапуск при старте WSL

Сейчас scheduler/webserver запускаются вручную (`start_airflow.sh`). Варианты
автозапуска (не реализовано):
- `[boot] command=/opt/airflow/start_airflow.sh` в `/etc/wsl.conf` (проверить
  выживаемость nohup-процессов);
- доработать systemd-юниты (разобраться, почему systemd глушит webserver ~16с:
  вероятно `KillMode`/cgroup + gunicorn-монитор);
- Windows Task Scheduler → `wsl -d Ubuntu -u root /opt/airflow/start_airflow.sh` при логоне.

## Диагностика проблем этой сессии

- **Ubuntu Python 3.14 несовместим с Airflow** → поставлен Python 3.12 через uv.
- **Кириллица в пути ломается** при передаче команд PowerShell→wsl→bash →
  файлы копируются через `/mnt/c` + `tr -d '\r'`, пути раскрываются shell-glob.
- **systemd глушит webserver ~16с** → перешли на nohup.
- **WSL-interop подтверждён**: `/mnt/d/VENV/LLM/Scripts/python.exe` из WSL bash
  запускает Windows-Python; `airflow tasks test` прогнал stub-вариант до SUCCESS.