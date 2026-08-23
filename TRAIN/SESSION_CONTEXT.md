# Сессия: Аудит /Rotate + Airflow orchestration

**Дата:** 6 июля 2026
**Задачи:**
1. Аудит ошибочных `/Rotate` флагов в PDF-сканах
2. Коррекция ориентации в пайплайне (`utils/rotate_fix.py`)
3. Переход с web_server на Apache Airflow (VMware VM)

---

## Исходные данные

| Параметр | Значение |
|----------|----------|
| Windows-хост | `DMS-MAL2`, 2× RTX PRO 4000 + RTX 5060 Ti |
| Python venv | `D:\VENV\LLM\Scripts\python.exe` (Python 3.12, torch) |
| Проект | `D:\Yandex.Disk\PYTHON\NLTK\Кластеризация` |
| PDF сканы (mart) | `D:\FileOrganizer\SourceFiles_mart\SourceFiles` — 24,155 PDF |
| PDF сканы (июнь) | `F:\BEELINE\DOC\NONFORMAL_06` — 104,104 PDF |
| VMware | Workstation 17.x, `C:\Program Files (x86)\VMware\VMware Workstation\` |
| Tesseract | `C:\Program Files\Tesseract-OCR\tesseract.exe` |

---

## Часть 1: Аудит /Rotate флагов

### Контекст

Обнаружено, что PDF `28571_Однолинейная схема` (2382×1684 PNG, extracted from PDF) при извлечении через `extract_pdf_images.py` получил `/Rotate=90` из PDF, но контент в mediabox уже был upright. PyMuPDF применяет `/Rotate` автоматически при `get_pixmap()` → PNG повёрнут, хотя визуально документ upright.

**Корневая проблема:** софт сканеров/мобильных приложений иногда добавляет ошибочный `/Rotate` к PDF, даже если контент уже upright.

### Аудит SourceFiles_mart (24,155 PDF)

**Скрипт:** `D:\FileOrganizer\TRAIN\qwen_orient\_audit_rot_dir.py`

| Категория | Кол-во | % |
|-----------|--------|---|
| rot=0, text≥100 | 21,018 | 87.0% |
| rot=0, text<100 (scan) | 2,614 | 10.8% |
| rot≠0, text≥100 (text PDF) | 382 | 1.6% |
| rot≠0, text<100 (scan, suspect) | 139 | 0.6% |
| Ошибки | 2 | 0.0% |

**Tesseract OSD на 139 suspects:** 118 CORRECT, **19 WRONG_ROTATE**, 2 ERROR

### Аудит NONFORMAL_06 (104,104 PDF)

**Скрипт:** `D:\FileOrganizer\TRAIN\qwen_orient\_audit_rot_dir2.py` (с таймаутом на файл + чекпоинтингом)

| Категория | Кол-во | % |
|-----------|--------|---|
| rot=0, text≥100 | 90,705 | 87.1% |
| rot=0, text<100 (scan) | 9,846 | 9.5% |
| rot≠0, text≥100 | 1,570 | 1.5% |
| rot≠0, text<100 (suspect) | 525 | 0.5% |
| Ошибки | 1,458 | 1.4% |

**Tesseract OSD на 525 suspects:** 424 CORRECT, **95 WRONG_ROTATE**, 6 ERROR

**Паттерн:** 87 из 95 — `pdf_rot=270, osd=270` (A4 portrait-сканы с ошибочным `/Rotate=270`)

### Результаты аудита

| Директория | Всего PDF | WRONG_ROTATE | % |
|------------|-----------|--------------|---|
| SourceFiles_mart | 24,155 | 19 | 0.08% |
| NONFORMAL_06 | 104,104 | 95 | 0.09% |

Доля ошибочных `/Rotate` стабильна — **~0.09%**.

### Файлы аудита

| Файл | Назначение |
|------|------------|
| `_audit_rot_dir.py` | Аудит с таймаутом + чекпоинтинг (Stage 1 + Stage 2) |
| `_audit_rot_dir2.py` | Версия с `ProcessPoolExecutor` + `threading.Timer` для жёсткого kill |
| `_stage2_osd.py` | Отдельный Stage 2 (Tesseract OSD на suspects) |
| `_rot_audit_NONFORMAL_06_summary.json` | Сводка аудита |
| `_rot_audit_NONFORMAL_06_suspects.json` | Список подозрительных PDF |
| `_rot_audit_NONFORMAL_06_verdicts.json` | Вердикты OSD |

---

## Часть 2: Коррекция ориентации в пайплайне

### Решение: `utils/rotate_fix.py`

**Новый модуль:** `D:\Yandex.Disk\PYTHON\NLTK\Кластеризация\utils\rotate_fix.py`

Функция `render_pdf_first_page_upright(pdf_path, output_path, zoom=2.0)`:

1. `/Rotate=0` → рендер как обычно (99.5% файлов, быстро)
2. `/Rotate≠0`, текстовый PDF → рендер с `/Rotate` (легитимный landscape)
3. `/Rotate≠0`, скан (text<100):
   - Рендер с `/Rotate` → Tesseract OSD
   - OSD orientation=0 → `/Rotate` корректен, сохраняем
   - OSD orientation≠0, conf≥2.0 → `/Rotate` ошибочен, повторный рендер **без** `/Rotate` (inverse rotation matrix)
   - OSD conf<2.0 → не можем определить, сохраняем как есть

### Интеграция в пайплайн

| Файл | Изменение |
|------|-----------|
| `utils/rotate_fix.py` | Новый модуль (160 строк) |
| `pipelines/pdf_scan_pipeline.py:66` | `_generate_first_page_image()` → `render_pdf_first_page_upright()` |
| `pipelines/pdf_tables_pipeline.py:106` | `_generate_first_page_image()` → `render_pdf_first_page_upright()` |
| `extractors/pdf_extractor.py:58` | Fallback path → `render_pdf_first_page_upright()` |

### Тест на 95 WRONG_ROTATE файлах

**Скрипт:** `_test_rotate_fix.py`

| Категория | Кол-во | % | Описание |
|-----------|--------|---|----------|
| FIXED | 55 | 57.9% | `/Rotate` ошибочен → повторный рендер без `/Rotate` → OSD 0° |
| STILL_WRONG | 9 | 9.5% | Оба рендера non-zero (скан действительно повёрнут, нужен VLM) |
| LOW_CONF | 31 | 32.6% | OSD conf<2.0 (почти пустые страницы) |
| ERROR | 0 | 0% | — |

**Время:** 131 сек на 95 файлов = 1.4 сек/файл (только для 0.5% подозрительных)

**Производительность в пайплайне:** ~12 мин накладных на 104K файлов (только 0.5% подозрительных проверяются через OSD)

### Остаток — 40 файлов (9 STILL_WRONG + 31 LOW_CONF)

Для них Tesseract OSD не помогает. Кандидаты для VLM-модели Qwen3-VL+LoRA (99.83% accuracy) — отдельная задача интеграции инференса в пайплайн.

---

## Часть 3: Переход на Apache Airflow

### Мотивация

`web_server.py` (90K строк) нестабилен: правки в одной роли (API/UI/состояние) ломают другую. Данные (БД + диск) становятся недоступны при падении UI. Человек в цикле — вынужденная мера для исследования, продакшен будет автономным.

### Архитектура: Airflow в VMware VM, задачи на Windows-хосте

```
VM (Ubuntu 26.04, VMware)            Windows-хост (GPU)
┌──────────────────────┐             ┌──────────────────────┐
│ Airflow scheduler    │             │ torch, transformers  │
│ Airflow webserver    │──SSH───────▶│ Qwen3-VL, embeddings │
│ Metadata DB (SQLite) │             │ 33K файлов на D:\F:\ │
│ DAG код              │             │ PostgreSQL            │
└──────────────────────┘             └──────────────────────┘
```

Airflow в VM управляет (scheduler + webserver + UI). Тяжёлые задачи (GPU, файлы) выполняются на Windows-хосте через `SSHOperator`.

### Установка

| Компонент | Детали |
|-----------|--------|
| VM | Ubuntu 26.04 LTS Server, 4 vCPU, 8 GB RAM, 50 GB диск |
| VMware | Workstation 17.x, NAT network |
| IP | 192.168.5.130 |
| Python | 3.12 (из PPA deadsnakes, отдельно от системного 3.14) |
| Airflow | 2.10.5 в venv `~/airflow_venv` |
| SSH provider | `apache-airflow-providers-ssh==4.0.0` |
| Metadata DB | SQLite (`~/airflow/airflow.db`) |
| Executor | SequentialExecutor (LocalExecutor для SQLite) |

### SSH-подключение VM → Windows

- Windows OpenSSH Server установлен (`Add-WindowsCapability`, порт 22)
- Airflow connection `windows_ssh`: `ssh://Anjey:ZXmew861@192.168.5.1:22`
- Парольная аутентификация (ключи для non-admin Windows требуют `administrators_authorized_keys`)

### systemd-сервисы (автозапуск)

| Сервис | Unit-файл | Порт |
|--------|-----------|------|
| `airflow-scheduler` | `/etc/systemd/system/airflow-scheduler.service` | 8793 (internal API) |
| `airflow-webserver` | `/etc/systemd/system/airflow-webserver.service` | 8080 (UI) |

**Критичный нюанс:** в unit-файле нужно `Environment=PATH=/home/anjey/airflow_venv/bin:...` — иначе `SequentialExecutor` не находит `airflow` бинарь для запуска задач (`FileNotFoundError: [Errno 2] No such file or directory: 'airflow'`).

**Управление:**
```bash
sudo systemctl restart airflow-scheduler
sudo systemctl status airflow-scheduler
sudo journalctl -u airflow-scheduler -f
```

### Airflow UI

- URL: `http://192.168.5.130:8080`
- Логин: `admin` / Пароль: `admin`
- `load_examples = False` в `airflow.cfg` (строка 124, секция `[core]`)
- Example DAGs удалены из БД через `_clean_examples.py`

### DAGs

#### `test_windows_ssh` (тестовый)

Проверка SSH → Windows + torch + CUDA. Две задачи:
- `test_echo` — `echo CONNECTED_FROM_AIRFLOW && hostname && whoami`
- `test_python` — `python -c "import torch; print(torch.cuda.device_count())"` → `CUDA: 2 GPUs`

#### `stage1_sort_files` (этап 1 пайплайна)

**Файл DAG:** `/home/anjey/airflow/dags/stage1_sort_files.py`
**CLI:** `D:\Yandex.Disk\PYTHON\NLTK\Кластеризация\cli_stage1.py`

```
count_files → sort_files        (always)
count_files → check_unzip → unzip_archives  (optional, if unzip=True)
```

**Задачи:**
1. `count_files` — dry-run: подсчёт файлов в source (без очистки, без сортировки)
2. `sort_files` — очистка целевых директорий + сортировка файлов
3. `check_unzip` — PythonOperator: проверка `params.unzip`, `AirflowSkipException` если False
4. `unzip_archives` — распаковка `.zip` архивов (только если `unzip=True`)

**Параметры (через "Trigger DAG w/ config" в UI):**
- `source_dir` — исходная директория (по умолчанию `D:\FileOrganizer\SourceFiles`)
- `unzip` — `True/False`: распаковывать zip-архивы (по умолчанию `False`)

**Очистка перед сортировкой (`cli_stage1.py --clean`, по умолчанию):**
- Sorted/* (все подкатегории) — 20 директорий
- ErrorFiles/, FailedExtraction/, Extracted/, Embeddings/, Clusters/, CentralDocuments/, PipelineData/
- БД: `db.clear_all()`
- НЕ удаляет: SourceFiles/, Logs/

**Результат тестового запуска:**
- Очистка: 20 директорий, 40,169 файлов удалено, БД очищена
- Сортировка: 73 файла в SourceFiles (все .zip → 0 отсортировано, ожидаемо)
- Распаковка (unzip=True): 73/73 zip OK, 181 файл в `D:\FileOrganizer\unpacked_zip\`

### CLI runners

| Файл | Назначение |
|------|------------|
| `cli_stage1.py` | Очистка + сортировка файлов (этап 1) |
| `cli_unzip.py` | Распаковка .zip в `D:\FileOrganizer\unpacked_zip\<zip_stem>\` |

### Распаковка .zip (`cli_unzip.py`)

- Каждый zip → отдельная директория `unpacked_zip/<zip_stem>/`
- `\\?\` extended-length path prefix для обхода Windows MAX_PATH (260)
- Защита от zip-бомб (max 1024 MB uncompressed per zip)
- Защита от path traversal (zip-slip)
- Очистка `unpacked_zip/` перед распаковкой
- **Результат:** 73/73 zip OK, 181 файл, 0 ошибок (было 27 ошибок до `\\?\` prefix)

---

## Незавершённые задачи

1. **DAG для этапа 2** (извлечение текста/изображений) — не создан
2. **DAG для этапа 3** (кластеризация) — не создан
3. **Интеграция VLM Qwen3-VL+LoRA в пайплайн** — для 40 файлов где Tesseract OSD не помогает (9 STILL_WRONG + 31 LOW_CONF)
4. **Pre-prep обучение LANCZOS** — скрипты готовы (`train_lora_qwen3vl_prep.py`), не запускались (отложено до анализа ошибок)
5. **Анализ 1 ошибки Qwen3-VL+LoRA** — `28571_Однолинейная схема` (чертёж ГОСТ, вертикальный текст в угловом штампе → pred=270 вместо true=0)
6. **Адаптация `qwen_quick_test_v2_lora.py` для Qwen3-VL** — обновить MODEL_ID и adapter_path
7. **Перенос Qwen3-VL LoRA адаптера** в `D:\MODELS\Transformers\Qwen3-VL-8B-Instruct-Orient-LoRA\`
8. **Распакованные файлы** в `D:\FileOrganizer\unpacked_zip\` — ожидают решения пользователя

---

## Технические находки

### 1. PyMuPDF и /Rotate
PyMuPDF применяет `/Rotate` автоматически при `get_pixmap()`. Если `/Rotate` ошибочен (контент уже upright, флаг говорит "поверни") — PNG получается повёрнутым. Решение: inverse rotation matrix через `page.rotation_matrix.invert()` для повторного рендера без `/Rotate`.

### 2. Tesseract OSD confidence
`orientation_conf < 2.0` — почти пустые страницы, OSD не определяет ориентацию. Порог `OSD_CONF_THRESHOLD=2.0` в `rotate_fix.py` — пропускает такие файлы (оставляют рендер с `/Rotate`).

### 3. Windows MAX_PATH (260 символов)
`zipfile.extractall()` не использует `\\?\` prefix → файлы с путём >260 теряются. Решение: ручная распаковка с `_long_path()` → `\\?\D:\...` prefix. Имена ZIP от ЭДО (счета-фактуры) могут быть >200 символов.

### 4. Airflow SequentialExecutor + PATH
`SequentialExecutor` запускает задачи через `subprocess.check_call(["airflow", ...])`. В systemd-сервисе PATH не включает venv → `FileNotFoundError: 'airflow'`. Решение: `Environment=PATH=/home/anjey/airflow_venv/bin:...` в unit-файле.

### 5. Airflow DAGs и Windows-пути в docstrings
Python docstrings с `D:\FileOrganizer\Unpacked\` → `\U` интерпретируется как unicode escape → `SyntaxError`. Решение: `\\` в docstrings или raw strings. Airflow парсит DAG-файлы на Linux VM, но пути в строках — Windows.

### 6. Airflow params и trigger
`airflow dags trigger --conf '{"unzip": true}'` из CLI не работает (парсинг аргументов). Решение: внутренний Python API `Client.trigger_dag(dag_id, conf={...})` через `_trigger2.py`.

### 7. SQLite + SequentialExecutor
SQLite не поддерживает параллелизм → `parallelism=1` → только одна задача одновременно. Для продакшена лучше мигрировать на PostgreSQL (уже есть на хосте). Пока терпимо для одной машины.

---

## Файлы сессии

### Новые файлы

| Файл | Назначение |
|------|------------|
| `utils/rotate_fix.py` | Коррекция ошибочного `/Rotate` в PDF-сканах |
| `cli_stage1.py` | CLI runner этапа 1 (очистка + сортировка) |
| `cli_unzip.py` | CLI runner распаковки .zip архивов |
| `_audit_rot_dir.py` | Аудит /Rotate с таймаутами |
| `_audit_rot_dir2.py` | Версия с ProcessPoolExecutor + threading.Timer |
| `_stage2_osd.py` | Tesseract OSD на suspects |
| `_test_rotate_fix.py` | Тест rotate_fix на 95 WRONG_ROTATE |

### Изменённые файлы

| Файл | Изменение |
|------|-----------|
| `pipelines/pdf_scan_pipeline.py:66` | `_generate_first_page_image()` → `render_pdf_first_page_upright()` |
| `pipelines/pdf_tables_pipeline.py:106` | `_generate_first_page_image()` → `render_pdf_first_page_upright()` |
| `extractors/pdf_extractor.py:58` | Fallback path → `render_pdf_first_page_upright()` |

### Airflow (в VM 192.168.5.130)

| Путь | Назначение |
|------|------------|
| `/etc/systemd/system/airflow-scheduler.service` | systemd unit (enabled) |
| `/etc/systemd/system/airflow-webserver.service` | systemd unit (enabled) |
| `/home/anjey/airflow/dags/test_windows_ssh.py` | Тестовый DAG |
| `/home/anjey/airflow/dags/stage1_sort_files.py` | DAG этапа 1 |
| `/home/anjey/airflow/airflow.cfg` | `load_examples = False` (строка 124) |

---

## Сессия 2: Airflow Stage 2 DAGs + сортировка с верификацией сигнатур + архивы

**Дата:** 8 июля 2026
**Задачи:**
1. Создать DAGs для этапа 2 (извлечение текста/изображений) — по одному на каждую группу форматов
2. Автономные скрипты извлечения без посредников (`stage2_processing.py`, `cli_stage2.py`)
3. Полный откат пайплайна (отдельный DAG)
4. Верификация сигнатур на этапе 1 — проверка magic bytes для всех форматов
5. Обработка архивов на этапе 1 — распаковка, префикс имён, сортировка распакованных файлов
6. Исправление кодировки имён файлов в ZIP (cp866)

---

### Часть 4: Архитектура Stage 2 — автономные DAGs

#### Эволюция решения

| Итерация | Подход | Проблема |
|----------|--------|----------|
| 1 | CLI-посредник `cli_stage2.py` + фабрика DAG `stage2_dags.py` | Лишний слой, скрипт в проекте |
| 2 | Base64-кодирование Python-скриптов внутри DAG | Невозможно редактировать, нет подсветки |
| 3 | Per-format скрипты `stage2_{fmt}.py` + DAG вызывает их через SSH | Решение принято |

**Финальная архитектура:**

```
VM (Airflow) --SSH--> Windows-хост
┌──────────────────────┐       ┌──────────────────────────────────┐
│ DAGs (21 шт)         │       │ stage2_scripts/ (21 шт)          │
│   stage2_pdf_text    │──SSH──│   stage2_pdf_text.py extract     │
│   stage2_word_docx   │──SSH──│   stage2_word_docx.py rollback   │
│   ...                │       │   ... (автономные, без посредников)│
│                      │       │                                   │
│ full_rollback        │──SSH──│ full_rollback.py (корень проекта)  │
└──────────────────────┘       └──────────────────────────────────┘
```

Каждый скрипт `stage2_scripts/stage2_{fmt}.py` **полностью автономен**:
- multiprocessing.Pool с worker, напрямую вызывающим `PipelineRegistry.get(fmt).extract(fp)`
- Запись в БД (update_extraction_batch)
- Откат (очистка БД + .npy + возврат файлов из FailedExtraction/ErrorFiles)
- Перемещение ошибочных файлов
- Без импортов `stage2_processing.py`, `process_file`, `cli_stage2.py`

#### DAGs (23 шт на VM)

| DAG ID | Задачи | Описание |
|--------|--------|----------|
| `stage2_{fmt}` (21 шт) | `rollback_gate → rollback → extract_gate → extract` | Извлечение для одного формата |
| `full_rollback` | `full_rollback` | Полный сброс пайплайна (очистка Sorted/ + БД) |
| `stage1_sort_files` | (существующий) | Этап 1 |

**Параметры `stage2_{fmt}` (Trigger DAG w/ config):**
- `skip_extract` (bool, default False) — только rollback
- `skip_rollback` (bool, default False) — пропуск rollback перед extract

#### Изменение часового пояса Airflow

`airflow.cfg` изменён:
```ini
default_timezone = Europe/Moscow    # было utc
default_ui_timezone = Europe/Moscow # было UTC
```
Календарь в UI теперь показывает московское время (UTC+3).

---

### Часть 5: Верификация сигнатур на этапе 1

#### Проблема

Файлы сортировались **по доверию к расширению** — без проверки magic bytes. Файл `photo.jpg`, который на самом деле PNG — попадал в `Sorted/Image_Jpg/` без проверки.

#### Решение: `sorting/verify_signature.py`

19 сигнатур (magic bytes):

| Сигнатура | → Расширение | → Категория |
|-----------|-------------|-------------|
| `%PDF` | `.pdf` | pdf |
| PKCS#7 | `.pdf` | error |
| `PK\x03\x04` (ZIP) | `.docx`/`.xlsx`/`.odt`/`.ods`/`.odp` | word/excel |
| `D0 CF 11 E0` (OLE2) | `.doc`/`.xls` | word/excel |
| `FF D8 FF` | `.jpg` | image |
| `89 PNG` | `.png` | image |
| `GIF87a`/`GIF89a` | `.gif` | image |
| `BM` | `.bmp` | image |
| `II*\x00`/`MM\x00*` | `.tif` | image |
| `RIFF...WEBP` | `.webp` | image |
| `{\rtf` | `.rtf` | word |
| `<?xml` (BOM варианты) | `.xml` | xml |

Функции:
- `verify_file_signature(file_path)` → `(detected_ext, category)` — проверка magic bytes
- `rename_by_signature(file_path)` → `(new_path, category)` — переименование если расширение не совпадает

Расширения без надёжной сигнатуры (`.txt`, `.csv`, `.svg`, `.xsd`, `.xsl`, `.wsdl`, `.jfif`) — доверяются расширению.

#### Интеграция в `_sort_worker()`

```python
# ШАГ 1: verify_signature → переименование если расширение не соответствует
f, sig_category = rename_by_signature(f)
if sig_category == 'error':  # PKCS#7 или неизвестная сигнатура
    move_file_to_error(f)
# ШАГ 2: диспетчер по проверенному расширению → процессор → Sorted/
```

---

### Часть 6: Обработка архивов на этапе 1

#### Новый модуль: `sorting/archive_processor.py`

**Поддерживаемые форматы:** `.zip`, `.gz`, `.rar`

**Поток обработки:**
1. Создать директорию `Sorted/zip/<archive_stem>/`
2. Распаковать всё содержимое (даже из вложенных директорий) — в один уровень
3. Добавить префикс (часть имени архива до первой точки) + точка + имя файла: `prefix.file.pdf`
4. Каждый файл проходит те же процессоры что и обычные файлы: сигнатура → `pdf_processor`/`text_processor`/etc → целевая папка
5. Мультимедиа (mp3/mp4/avi/...) → `Sorted/multimedia/`
6. Архив → `Sorted/zip/` (рядом с поддиректорией)
7. Отчёт → `Logs/archives_report.txt` (перезаписывается при каждом запуске)
8. Поддиректория распаковки удаляется (файлы перемещены в целевые папки)

**Защита:**
- Zip-бомба: max 1024 MB uncompressed per zip
- Path traversal (zip-slip): только имя файла, без пути
- Коллизия имён: `_1`, `_2` суффиксы

**Кодировка имён в ZIP:**
Windows-утилиты (WinRAR, 7-Zip, Explorer) создают архивы с cp866 (DOS Russian) именами, но не устанавливают UTF-8 флаг (`0x800`). Функция `_decode_zip_filename()`:
1. Если UTF-8 флаг установлен → имя корректно
2. Если нет → закодировать обратно в cp437, декодировать как cp866
3. Fallback: cp1251
4. Последний fallback — как есть

**Отчёт `archives_report.txt`:**
- Кодировка: UTF-8 with BOM (`utf-8-sig`)
- Группировка по имени архива
- Без временных меток
- Указание целевой папки для каждого файла

Пример:
```
================================================================================
Отчёт по архивам
Архивов обработано: 1
Всего файлов: 2

--------------------------------------------------------------------------------
Архив: Отчёт_2024.zip
Файлов: 2
--------------------------------------------------------------------------------
  [sorted    ] Отчёт.договор.pdf
              → Sorted/PDF_Text/Отчёт.договор.pdf
  [multimedia] Отчёт.видео.mp4
              → Sorted/multimedia/Отчёт.видео.mp4
```

#### Интеграция в `run_stage_1_sort()`

Архивы обрабатываются **до** пула multiprocessing (последовательно, с общим отчётом):
```python
# ШАГ 1: process_archives(source_path) — последовательно, общий отчёт
# ШАГ 2: _run_sort_pool(non_archive_files) — параллельная сортировка
# ШАГ 3: _cleanup_empty_dirs() — удалить пустые поддиректории
```

---

### Часть 7: Полный откат (full_rollback)

#### DAG `full_rollback` (на VM)

Отдельный автономный DAG — не привязан к этапам или форматам. Вызывает `full_rollback.py` на Windows через SSH.

#### Скрипт `full_rollback.py` (в корне проекта)

- Удаляет `Sorted/` **целиком** (создаётся пустой, поддиректории — при сортировке)
- Удаляет: `ErrorFiles/`, `FailedExtraction/`, `Extracted/`, `Embeddings/`, `Clusters/`, `CentralDocuments/`, `PipelineData/`
- Очищает БД (`db.clear_all()`)
- Удаляет `Logs/archives_report.txt`
- НЕ удаляет: `SourceFiles/`, `Logs/`

---

### Часть 8: Изменения в config.py

**FORMAT_TARGETS** — новые категории:
```python
"zip":        "Sorted/zip",
"multimedia": "Sorted/multimedia",
```

**XML — упрощённые имена:**
```python
# Было:
"xml_xml":    "Sorted/XML_Xml",
"xml_xsd":    "Sorted/XML_Xsd",
# Стало:
"xml_xml":    "Sorted/xml",
"xml_xsd":    "Sorted/xsd",
"xml_xsl":    "Sorted/xsl",
"xml_wsdl":   "Sorted/wsdl",
```

---

### Часть 9: Очистка при полном откате и сортировке

`full_rollback.py` и `cli_stage1.py:_clean_targets()`:
- Удаляют `Sorted/` целиком (вместо отдельных поддиректорий)
- Создают пустой `Sorted/`
- Поддиректории создаются процессорами только при наличии файлов
- После сортировки `_cleanup_empty_dirs()` удаляет пустые поддиректории

---

### Изменённые файлы

| Файл | Изменение |
|------|-----------|
| `stages/stage1_sorting.py` | Архивы до пула; верификация сигнатур; _cleanup_empty_dirs |
| `config.py` | FORMAT_TARGETS: +zip, +multimedia, XML имена упрощены |
| `cli_stage1.py` | _clean_targets: удаляет Sorted/ целиком, очищает archives_report.txt |
| `full_rollback.py` | Удаляет Sorted/ целиком, создаёт пустой, очищает отчёт архивов |
| `airflow.cfg` (VM) | default_timezone = Europe/Moscow, default_ui_timezone = Europe/Moscow |

### Новые файлы

| Файл | Назначение |
|------|------------|
| `sorting/verify_signature.py` | Проверка magic bytes для всех форматов |
| `sorting/archive_processor.py` | Обработка архивов: распаковка, префикс, сортировка, отчёт |
| `full_rollback.py` | Полный откат пайплайна (скрипт в корне проекта) |
| `stage2_scripts/stage2_{fmt}.py` (21 шт) | Автономные скрипты извлечения (прямой вызов pipeline) |
| `airflow/dags/stage2_{fmt}.py` (21 шт) | DAGs этапа 2 (на VM) |
| `airflow/dags/full_rollback.py` | DAG полного отката (на VM) |
| `airflow/dags/stage1_sort_files.py` | DAG этапа 1 (на VM, изменён) |

### Удалённые файлы

| Файл | Причина |
|------|---------|
| `cli_stage2.py` | Посредник — заменён автономными скриптами |
| `airflow/dags/_ssh_helpers.py` | Base64-кодирование — не используется |
| `airflow/dags/_generate_dags.py` | Старый генератор DAG-файлов |
| `airflow/dags/stage2_dags.py` | Фабрика DAG — заменена на per-format файлы |
| `extract_pdf_images.py` | Тестовый скрипт |
| `test-results/` | Артефакты тестов |
| `web_test*.log` | Логи тестов |

---

### Технические находки (Session 2)

#### 8. ZIP-кодировка имён файлов (cp866 vs cp437)
Windows-утилиты (WinRAR, 7-Zip, Explorer) создают ZIP-архивы с именами файлов в cp866 (DOS Russian), но не устанавливают UTF-8 флаг (`0x800`) в `ZipInfo.flag_bits`. Python по умолчанию декодирует имена как cp437 (DOS Latin US) → крокозябры. Решение: `_decode_zip_filename()` — проверка флага, если нет → `encode('cp437').decode('cp866')`.

#### 9. Airflow DAG парсинг на Linux, скрипты на Windows
DAG-файлы парсятся на VM (Ubuntu), но вызывают скрипты на Windows через SSH. Кавычки и слеши нужно экранировать: `cd /d "D:\path" && "D:\VENV\LLM\Scripts\python.exe" script.py`.

#### 10. Airflow timezone
`default_ui_timezone = UTC` → календарь показывает вчерашний день (UTC vs Moscow UTC+3). Решение: `default_ui_timezone = Europe/Moscow` в `airflow.cfg` + `kill -HUP <gunicorn_master_pid>`.

#### 11. Полный откат vs откат формата
Полный откат (`full_rollback`) — глобальная операция (очистка всех директорий + БД), не привязана к этапу или формату. Откат формата (`stage2_{fmt}.py rollback`) — только для одного формата (очистка в БД по `format_type` + возврат файлов из FailedExtraction/ErrorFiles).

#### 12. Поддиректории Sorted/ создаются только при наличии файлов
При полном откате `Sorted/` удаляется целиком и создаётся пустой. При сортировке процессоры создают поддиректории через `mkdir(parents=True, exist_ok=True)` только при перемещении файла. После сортировки `_cleanup_empty_dirs()` удаляет пустые.
