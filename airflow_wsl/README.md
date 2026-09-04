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

## Автозапуск при логоне Windows (Task Scheduler)

Реализован через Windows Task Scheduler, триггер `At user logon`. Содержимое:

- `start_airflow_windows.cmd` — обёртка на одну команду `wsl.exe -d Ubuntu -u root -- /opt/airflow/start_airflow.sh`.
  Запущенный WSL дистрибутив стартует systemd → PostgreSQL (метаданные Airflow),
  затем `start_airflow.sh` поднимает scheduler + webserver через nohup и завершается (~2-3с).
- `airflow_autostart.xml` — описатель задачи Task Scheduler 2.0 (триггер LogonTrigger,
  principal InteractiveToken/LeastPrivilege, ExecutionTimeLimit PT5M).
- `register_task.ps1` — PowerShell script, импортирующий XML в Task Scheduler.

### Регистрация (выполняется 1 раз)

Требует elevated PowerShell (Task Scheduler API не даёт не-админу регистрировать задачи даже в `\\Users\<user>`).

**Вариант 1 — скриптом (рекомендуется):**
```powershell
# 1) ПКМ по PowerShell -> Run as administrator
# 2) Разрешить запуск скриптов (если ещё не разрешено):
Set-ExecutionPolicy -Scope Process Bypass -Force
# 3) Зарегистрировать:
D:\Yandex.Disk\PYTHON\NLTK\Кластеризация\airflow_wsl\register_task.ps1
```

**Вариант 2 — одной командой из elevated PowerShell:**
```powershell
Register-ScheduledTask `
    -TaskName "AirflowWSL_Autostart" `
    -TaskPath "\Кластеризация" `
    -Xml (Get-Content "D:\Yandex.Disk\PYTHON\NLTK\Кластеризация\airflow_wsl\airflow_autostart.xml" | Out-String) `
    -User "$env:USERDOMAIN\$env:USERNAME" `
    -Force
```

**Вариант 3 — через GUI:**
1. `Win+R` → `taskschd.msc`
2. Action → Import Task… → выбрать `airflow_autostart.xml`
3. (На вкладке General проверить имя пользователя)

### Проверка

```powershell
# Состояние задачи (не-elevated OK)
Get-ScheduledTask -TaskName "AirflowWSL_Autostart" -TaskPath "\Кластеризация" `
    | Select-Object TaskName, TaskPath, State, @{n='Trigger';e={$_.Triggers[0].GetType().Name}} | Format-List

# Ручной запуск (не-elevated OK): в UI "Run" или из CLI
Start-ScheduledTask -TaskName "AirflowWSL_Autostart" -TaskPath "\Кластеризация"

# Проверка Airflow
(Invoke-WebRequest http://localhost:8080/health -UseBasicParsing).StatusCode  # 200
```

### Удаление

```powershell
# Из elevated PowerShell:
Unregister-ScheduledTask -TaskName "AirflowWSL_Autostart" -TaskPath "\Кластеризация" -Confirm:$false
```

### Почему не `[boot] command=` в `/etc/wsl.conf`

Альтернативно можно дописать `command=/opt/airflow/start_airflow.sh` в секции `[boot]` `/etc/wsl.conf` — тогда Airflow стартует при *первом* запуске WSL кем угодно. Недостатки:
- WSL не запускается автоматически при загрузке Windows (требуется внешний триггер);
-Boot command запускается синхронно — boot WSL задерживается на ~2-3с на старте;
- Если пользователь сам открывает терминал WSL после старта Airflow — повторный запуск пройдёт через `pkill` в start_airflow.sh, что сбрасывает процессы.

Выбранный путь (Task Scheduler ONLOGON) явно запускает WSL один раз при логоне, Airflow поднимается через `start_airflow.sh`, последующие интерактивные запуски WSL не повторяют запуск Airflow.

### Почему не systemd-юниты

В этой сессии webserver падал через ~16с при запуске через systemd (KillMode/cgroup конфликт с gunicorn-монитором). nohup-запуск стабилен. Можно вернуться к systemd после фикса этого конфликта — тогда автозапуск переезжает на `wsl.conf [boot] systemd=true` + `systemctl enable airflow-{scheduler,webserver}` и Task Scheduler нужен только чтобы `wsl --boot` при логоне.

## Диагностика проблем этой сессии

- **Ubuntu Python 3.14 несовместим с Airflow** → поставлен Python 3.12 через uv.
- **Кириллица в пути ломается** при передаче команд PowerShell→wsl→bash →
  файлы копируются через `/mnt/c` + `tr -d '\r'`, пути раскрываются shell-glob.
- **systemd глушит webserver ~16с** → перешли на nohup.
- **WSL-interop подтверждён**: `/mnt/d/VENV/LLM/Scripts/python.exe` из WSL bash
  запускает Windows-Python; `airflow tasks test` прогнал stub-вариант до SUCCESS.
