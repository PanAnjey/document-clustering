# Airflow в Docker — руководство по развёртыванию

Стенд для исследовательских прогонов пайплайна кластеризации.
**Топология:** Airflow крутится в Linux-контейнерах, а всю тяжёлую работу
(GPU, извлечение, эмбеддинги, кластеризация) выполняет **Windows-хост** —
контейнеры дотягиваются до него по SSH (`host.docker.internal`).

```
┌─────────────────── Docker Desktop (Linux VM) ───────────────────┐
│  airflow-metadb (postgres)   ← метаданные Airflow                │
│  airflow-scheduler                                                │
│  airflow-webserver (UI :8080)                                     │
│         │ SSHOperator (windows_ssh)                               │
└─────────┼─────────────────────────────────────────────────────────┘
          │ host.docker.internal:22
          ▼
┌──────── Windows-хост (эта машина) ────────┐
│  sshd (уже Running)                        │
│  D:\VENV\LLM\Scripts\python.exe            │
│  PostgreSQL file_organizer_db :5432        │
│  GPU cuda:0 / cuda:1, COM, Aspose          │
│  D:\FileOrganizer\ (данные)                │
└────────────────────────────────────────────┘
```

---

## Предусловия (выполняется ОДИН раз, нужны права администратора)

Текущая сессия **без прав администратора**, поэтому эти шаги делает пользователь.

### 1. Установить Docker Desktop

- Скачать: https://www.docker.com/products/docker-desktop/
- При установке выбрать **WSL 2 backend** (рекомендуется) — установщик сам
  предложит поставить WSL2, если его нет.
- На этой машине виртуализация в BIOS включена (`HypervisorPresent=True`),
  но присутствует VMware (адаптеры VMnet1/VMnet8). Docker Desktop с WSL2
  сосуществует с VMware через Windows Hypervisor Platform — если возникнет
  конфликт, включить компонент Windows «Платформа гипервизора Windows»
  (Windows Hypervisor Platform) в «Программы и компоненты».
- После установки перезапустить машину, запустить Docker Desktop, дождаться
  статуса **Engine running**.

Проверка (в новом терминале):
```powershell
docker --version
docker compose version
```

### 2. Разрешить SSH-подключения от контейнеров к Windows-хосту

`sshd` уже запущен. Нужно убедиться, что:

- **Firewall** пропускает входящие на порт 22 из сети Docker. Обычно правило
  "OpenSSH SSH Server (sshd)" уже создано установкой OpenSSH. Проверить:
  ```powershell
  Get-NetFirewallRule -DisplayName "*OpenSSH*" | Format-Table DisplayName, Enabled, Direction
  ```
- **Default shell** для OpenSSH = `cmd.exe` (по умолчанию так и есть). DAG-и
  используют синтаксис `cd /d "..." && "python.exe" ...` — это команды cmd.exe.
  Если раньше меняли на PowerShell — вернуть на cmd, либо переписать команды
  в DAG. Проверить:
  ```powershell
  Get-ItemProperty "HKLM:\SOFTWARE\OpenSSH" -Name DefaultShell -ErrorAction SilentlyContinue
  ```
  (Если ключа нет — используется cmd.exe, это правильно.)
- Проверить, что по SSH можно войти под нужным пользователем и запустить python:
  ```powershell
  ssh Anjey@localhost "cd /d D:\Yandex.Disk\PYTHON\NLTK\Кластеризация && D:\VENV\LLM\Scripts\python.exe --version"
  ```

---

## Развёртывание (после установки Docker)

Все команды — из папки `airflow_docker/`.

### 1. Создать `.env` из шаблона и заполнить секреты

```powershell
cd D:\Yandex.Disk\PYTHON\NLTK\Кластеризация\airflow_docker
Copy-Item .env.example .env
notepad .env
```

Заполнить:
- `AIRFLOW_CONN_WINDOWS_SSH` — `ssh://<user>:<password>@host.docker.internal:22`
  (спецсимволы пароля URL-энкодить: `@`→`%40`, `:`→`%3A`).
- `AIRFLOW_CONN_POSTGRES_EXPERIMENTS` — обычно оставить как есть
  (`postgresql://postgres:postgres@host.docker.internal:5432/file_organizer_db`).
- При желании поменять `_AIRFLOW_WWW_USER_PASSWORD`.

### 2. Инициализация (миграция БД + admin + импорт pools/variables)

```powershell
docker compose up airflow-init
```
Дождаться `=== init done ===`. Это разово:
- мигрирует метаданные Airflow,
- создаёт пользователя `admin`,
- импортирует пулы из `../airflow_setup/pools.json`
  (gpu=1, extractors=1, clustering=1, preflight=4),
- импортирует Variables из `../airflow_setup/variables_default.json`
  (`variant.<stage>.<subformat>=default`).

### 3. Поднять Airflow

```powershell
docker compose up -d
```
Открыть **http://localhost:8080** (admin / см. пароль в `.env`).

### 4. Проверить связку с Windows-хостом

В UI Airflow → Admin → Connections должны быть `windows_ssh` и
`postgres_experiments` (заведены из env-переменных автоматически).

Быстрый тест SSH-связки — Trigger DAG `golden_stage1_run` с параметром
`note="smoke"` (осторожно: полный прогон пересортирует источник; сейчас
SourceFiles/ пуст, так что прогон безопасен и просто ничего не отсортирует).

---

## Управление

| Действие | Команда |
|---|---|
| Поднять | `docker compose up -d` |
| Логи scheduler | `docker compose logs -f airflow-scheduler` |
| Остановить (метаданные сохраняются) | `docker compose down` |
| Полный сброс метаданных Airflow | `docker compose down -v` |
| Пересоздать после правки .env | `docker compose up -d --force-recreate` |
| Перечитать pools/variables | `docker compose run --rm airflow-init` |

DAG-и подхватываются автоматически из `../airflow/dags` (bind-mount, read-only).
После добавления нового DAG-файла — подождать ~30 сек (scheduler пересканирует).

---

## SSH по ключу (надёжнее пароля)

1. Сгенерировать ключ (если нет) и положить публичный в
   `C:\Users\<user>\.ssh\authorized_keys` (или в
   `C:\ProgramData\ssh\administrators_authorized_keys` для админов).
2. Смонтировать приватный ключ в контейнер и указать его в connection.
   Проще — задать connection через UI: Admin → Connections → `windows_ssh`,
   Conn Type = SSH, Host = `host.docker.internal`, Login = `<user>`,
   в Extra: `{"key_file": "/opt/airflow/config/id_rsa", "no_host_key_check": true}`,
   а сам ключ положить в `airflow_docker/config/id_rsa`.
3. Тогда из `.env` убрать пароль из `AIRFLOW_CONN_WINDOWS_SSH`
   (оставить `ssh://<user>@host.docker.internal:22`).

---

## Провайдеры и версия

- Образ: `apache/airflow:2.10.5-python3.11` (LocalExecutor).
- SSH/Postgres провайдеры доустанавливаются при старте через
  `_PIP_ADDITIONAL_REQUIREMENTS` (медленнее старт, проще поддержка).
  Для прода — собрать кастомный образ Dockerfile'ом с pre-installed
  провайдерами.
- Хочешь Airflow 3.x — обнови образ и учти, что webserver разделён на
  `apiserver` + `dag-processor`; docker-compose придётся адаптировать.

---

## Частые проблемы

- **Контейнер не достукивается до Windows sshd** → проверить firewall (порт 22),
  что `host.docker.internal` резолвится (в новых Docker Desktop — из коробки).
- **SSH-команды падают с синтаксисом** → default shell OpenSSH должен быть
  `cmd.exe` (DAG-и используют `cd /d ... &&`).
- **Кириллица в пути проекта** (`Кластеризация`) → DAG-и обращаются к путям
  Windows через SSH-строки; сам bind-mount использует относительный путь
  `../airflow/dags`, кириллица в нём допустима для Docker Desktop.
- **Postgres не доступен из контейнера** → PostgreSQL на Windows должен слушать
  не только `localhost`, но и внешний интерфейс, ИЛИ доступ идёт через
  `host.docker.internal`; проверить `listen_addresses` в postgresql.conf и
  `pg_hba.conf` (разрешить подключения из подсети Docker).