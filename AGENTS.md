# AGENTS.md — Пайплайн кластеризации файлов

## Быстрый старт

```bash
python main.py
```

Запускает полный 5-этапный пайплайн кластеризации документов (33,830 шт) с pre-flight валидацией и паузами между этапами для подтверждения через web-интерфейс:
сортировка → [подтверждение] → извлечение (группы форматов с пошаговым подтверждением) → [подтверждение] → саммаризация → [подтверждение] → эмбеддинги → [подтверждение] → кластеризация.

**Тестовый прогон (20 файлов, 2026-06-18):** все 5 этапов успешно завершены за ~6 мин. PDF (5/5), IMAGE (7/7), WORD (2/5, 3 `.doc` ошибки), EXCEL (3/3). 1 кластер, 17 доков, Similarity avg 0.78.  
**Тестовый прогон 2 (17 файлов Word, 2026-06-18, web_server):** все 5 этапов успешно завершены за 141 сек. [Errno 22] исправлен через `_ensure_cuda_context()`. Подробнее см. "Тестовый прогон" ниже.
**Тестовый прогон 3 (Aspose.Words for Java, 2026-06-20):** все 5 этапов успешно завершены за ~4 мин. Word (.docx/.doc/.rtf/.odt) → Aspose(Java), HDBSCAN кластеризация, PDF классификация на 3 категории.

---

## Важная информация для следующей сессии

- **Config override**: `config_override.json` переопределяет `config.py`. Редактировать ОБА файла.
- **`report.txt`**: `D:\FileOrganizer\report.txt` — всегда отражает последний запуск.
- **FailedExtraction/**: `D:\FileOrganizer\FailedExtraction\` — файлы, из которых не удалось извлечь текст. Возвращаются в Sorted/ при откате этапа 2.
- **LLM backend**: transformers (Qwen3.5-4B на cuda:0), НЕ llama-server
- **PyTorch**: 2.12.0+cu132 (`pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132`)
- **Aspose.Words for Java**: `D:\Yandex.Disk\Aspose\Aspose.Words for Java` — основной метод для Word-форматов (.docx, .doc, .rtf, .odt) через subprocess вызов JVM

---

## Обзор архитектуры

**Точка входа:** `main.py` (~156 строк, оркестратор после refactoring v5.0, разбит на модули в `stages/`, `state/`, `utils/`)  
**Standalone саммаризация:** `summarize_only.py` (только этап 3, читает из БД)
**Web-интерфейс:** `web_server.py` (Starlette + Uvicorn, порт 8080)

**Этапы:**
1. Сортировка файлов по расширению в категории (`Sorted/PDF`, `Sorted/Excel` и т.д.)
   - PKCS#7 проверка для .pdf файлов (SignedData + EnvelopedData) → ErrorFiles
   - OpenDocument XML → проверка namespace корневого элемента, переименование в .odt/.ods
   - Файлы без расширения → определение по сигнатуре
   - **Проверка целостности ODF** — для `.odt/.ods/.odp` проверка `zipfile.ZipFile` на этапе 1, битые zip → ErrorFiles
   - **PDF классификация** — после сортировки PDF проверяются через `fitz`:
     - Текст всех страниц < 100 символов → `pdf_scan` (`Sorted/PDF_Scan`)
     - Есть текст + таблица на любой странице → `pdf_tables` (`Sorted/PDF_Tables`)
     - Есть текст, таблиц нет → `pdf_text` (`Sorted/PDF_Text`)
2. Извлечение текста/изображений (CPU-bound, `multiprocessing.Pool`)
   - **Aspose.Words for Java** — основной метод для Word/RTF/ODT через subprocess вызов JVM
   - DOCX → Aspose(Java) → COM (fallback)
   - DOC → Aspose(Java) → olefile → COM (fallback)
   - RTF → Aspose(Java) → COM (fallback)
   - ODT → Aspose(Java) → COM (fallback)
   - ODP → extract_xml (LibreOffice fallback)
   - XLS/XLSX → Pandoc → COM Excel (fallback)
   - TXT/CSV → прямое чтение
   - Изображения → Tesseract OCR напрямую
   - PDF:
     - `pdf_scan` — PyMuPDF4LLM с OCR (~550ms)
     - `pdf_tables` — PyMuPDF4LLM для структурированных таблиц (~550ms)
     - `pdf_text` — fitz.get_text() (~4ms, без PyMuPDF4LLM)
3. LLM саммаризация через transformers (Qwen3.5-4B на cuda:0), параллельные вызовы batch=16
4. Генерация эмбеддингов моделями nomic-embed на GPU (`cuda:1`)
5. Кластеризация HDBSCAN с динамическим минимальным размером кластера; перемещение топ-N кластеров в output
   - Центральные документы (топ-3 ближайших к центроиду) копируются в `D:\FileOrganizer\CentralDocuments\{cluster_name}\`
   - **Динамическое количество кластеров**: кластеры с < MIN_CLUSTER_SIZE документов объединяются в "Неклассифицированные"

**GPU распределение:**
- `cuda:0` (RTX PRO 4000 Blackwell, 24 GB) — LLM саммаризация (Qwen3.5-4B)
- `cuda:1` (GeForce RTX 5060 Ti, 16 GB) — эмбеддинги (nomic-embed-text + nomic-embed-vision, ~2 GB)

**Ключевые модули:**
- `config.py` — пути, форматы, пороги, параметры LLM, параметры БД
- `file_processor.py` — логика извлечения (PDF с OCR, Office COM → PDF, структура XML, классификация PDF)
- `dependency_checker.py` — pre-flight валидация 16 зависимостей при старте
- `llm_summarizer.py` — саммаризация через transformers (Qwen3.5-4B), batch=16
- `embeddings_engine.py` — nomic-embed модели для текстовых/графических эмбеддингов
- `cluster_engine.py` — HDBSCAN кластеризация на GPU (без O(N²) матрицы расстояний)
- `database.py` — PostgreSQL + pgvector (connection pooling, батчевые INSERT)
- `pipeline_state.py` — управление состоянием, откат, надёжное удаление
- `web_server.py` — web-интерфейс с подтверждением между этапами и подэтапами
- `stages/stage1_sorting.py` — сортировка файлов по расширениям и PDF классификация
- `stages/stage2_processing.py` — обработка форматов (извлечение + эмбеддинги, раздельные подэтапы)
- `stages/stage3_clustering.py` — HDBSCAN кластеризация и уточнение
- `state/progress_tracker.py` — отслеживание прогресса выполнения
- `utils/format_groups.py` — управление группами форматов
- `utils/rollback_manager.py` — откат операций

---

## Среда и зависимости

**Пути в `config.py`:**
- Source: `D:\FileOrganizer\SourceFiles`
- Models: `D:\MODELS\Transformers` (nomic-embed-text-v1.5 + nomic-embed-vision-v1.5)
- LibreOffice: `C:\Program Files\LibreOffice\program\soffice.exe`
- Pandoc: `C:\Program Files\Pandoc\pandoc.exe`

**Требуемые сервисы:**
- GPU с CUDA (2 GPU для параллельной работы эмбеддингов + LLM)
- transformers (Qwen3.5-4B на cuda:0) для саммаризации (НЕ llama-server)
- PostgreSQL + pgvector (опционально, для stage_1_main.py)

**Python зависимости:**
- `torch` с CUDA (`pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132`)
- `accelerate` — для загрузки HF моделей на GPU
- `python-docx` — fallback для DOCX при ошибке COM
- `striprtf` — извлечение текста из RTF файлов (не используется, Aspose(Java) вместо него)
- `pgvector` — для работы с pgvector в PostgreSQL
- `olefile` — Python-парсинг OLE2 (.doc) файлов, основой метод для .doc перед COM
- `hdbscan` — HDBSCAN кластеризация (без O(N²) матрицы расстояний)

### Запустить полный пайплайн
```bash
python main.py
```

### Запустить только этап 3 (саммаризация из БД)
```bash
python summarize_only.py
```

### Запустить только этап 1 (с сохранением в БД)
```bash
python stage_1_main.py
```

### Скачать модели эмбеддингов
```bash
python download_models.py
```

### Отключить саммаризацию
В `config.py`: `LLM_ENABLED: bool = False`

### Грациозная остановка
Нажать `Ctrl+C` — логирует "Process interrupted by user."

### Проверить логи
`D:\FileOrganizer\Logs\file_organizer.log`

### Посмотреть результаты
- Кластеризованные файлы: `D:\FileOrganizer\Clusters\Cluster_{N}\`
- Отчёт: `D:\FileOrganizer\report.txt`
- Ошибки: `D:\FileOrganizer\ErrorFiles\`

---

## Важные нюансы (Gotchas)

1. **Изоляция профиля LibreOffice** — PID+TID-based временный профиль, `Semaphore(3)` ограничивает параллелизм.
2. **Лимит размера изображений** — `PIL.Image.MAX_IMAGE_PIXELS = 178_956_970` (защита от DecomBomb).
3. **Кластеризация XML по структуре** — Извлекает XPath дерево, а не текстовое содержимое.
4. **Приоритет кластеров** — Combined > Text > Image эмбеддинги при присвоении финального cluster ID.
5. **Обработка коллизий имён** — При перемещении добавляется `_1`, `_2` и т.д.
6. **LLM саммаризация опциональна** — Если отключена (`LLM_ENABLED=False`), пайплайн продолжает с исходным текстом. Бэкенд: transformers (Qwen3.5-4B), НЕ llama-server.
7. **Единое векторное пространство** — nomic-embed-text и nomic-embed-vision дают 768-dim векторы в одном пространстве, корректно для combined-эмбеддингов.
8. **HDBSCAN без матрицы расстояний** — Кластеризация выполняется напрямую через `cosine` distance, без вычисления полной O(N²) матрицы расстояний (HDBSCAN использует взаимную достижимость). Параметр `DIST_MATRIX_CHUNK` не используется.
9. **Автоочистка GPU** — Перед загрузкой любой модели (LLM или эмбеддинги) автоматически вызывается `torch.cuda.empty_cache()` и `torch.cuda.synchronize()` для освобождения VRAM. Функции `unload_model()` (LLM) и `unload_all_models()` (эмбеддинги) удаляют модели из памяти и очищают кэш.
9b. **CUDA-контекст в новом потоке** — `torch.cuda.synchronize()` без предварительного создания тензора не инициализирует CUDA-контекст в новом потоке, что приводит к `[Errno 22] Invalid argument` при загрузке моделей из pipeline-потока web_server. Обязательно вызывать `torch.tensor([0], device=dev)` перед `synchronize()` в `_ensure_cuda_context()` (`hf_summarizer.py:42-43`, `embeddings_engine.py:74`).
10. **ODF форматы** — Файлы .odt/.ods/.odp/.odg обрабатываются через LibreOffice, MS Word не может их открыть корректно.
11. **Rollback** — При откате этапа удаляются все артефакты: файлы, данные в БД, запущенные процессы. Откат возвращает систему в состояние после завершения предыдущего этапа.
12. **Подавление модальных окон Office** — При конвертации через COM (Word/Excel, финальный fallback) отключены все интерактивные элементы: `DisplayAlerts=False`, `ScreenUpdating=False`, `AutomationSecurity=3` (отключение макросов), `EnableEvents=False`, `AutoRecover.Enabled=False`, `OpenAndRepair=False`. Это предотвращает появление диалогов "публикация", "сохранить изменения", "восстановление повреждённого файла" и других модальных окон, которые блокируют автоматическую обработку. Дополнительно установлен таймаут 30 секунд через `threading.Timer` — если Word зависает на модальном окне, процесс прерывается и файл пропускается. Файлы < 100 байт пропускаются без попытки открытия.
13. **Qwen3.5-4B — reasoning модель** — По умолчанию шаблон Qwen3.5 добавляет `<think>\n` перед генерацией. Для классификации без reasoning используется `enable_thinking=False` в `apply_chat_template`. Односложные промпты (`ТЕМА:` в конце user-сообщения) срабатывают как completion-trick, forcing модель заполнить значение.
14. **Односложный completion-trick** — Промпт заканчивается на `ТЕМА:`, модель продолжает генерацию, не переформулируя вопрос. Позволяет обойти проблему, когда Qwen3.5-4B игнорирует документ и вместо классификации копирует инструкцию из system prompt.
15. **Greedy decoding для классификации** — `do_sample=False` даёт детерминированный результат (воспроизводимость) и незначительно быстрее, чем sampling с low temperature.
16. **Обновление путей в БД** — При перемещении файлов (этап 2 → FailedExtraction/ErrorFiles, этап 5 → Clusters) путь в таблице `documents.file_path` обновляется через `db.update_file_path(old, new)`. Функции `move_file_to_target()` и `move_file_to_error()` возвращают новый путь (`Optional[Path]`).
17. **RTF через Aspose(Java)** — Файлы .rtf обрабатываются через Aspose.Words for Java (основной метод). Библиотека striprtf не используется.
18. **Aspose(Java) — основной метод для .docx/.doc/.rtf/.odt** — Subprocess вызов JVM с лицензией. COM Word — только fallback после Aspose.
19. **Многостраничное извлечение Word** — Word COM извлекает текст со всех страниц документа, а не только с первой. Используется `doc.ComputeStatistics(2)` для получения количества страниц.
20. **Игнорирование таблиц 1x2** — Таблицы с одной строкой и двумя ячейками игнорируются при извлечении Word COM (часто это служебные таблицы с метаданными).
21. **Fallback для .doc файлов** — Сначала olefile (быстрый Python-парсинг OLE2, ~1 мс), в крайнем случае — COM Word (~7 сек). Pandoc НЕ поддерживает бинарный .doc (возвращает None). Цепочка: olefile → COM Word.
22. **Pandoc не поддерживает .doc** — Формат `"doc"` убран из `_get_pandoc_input_format()`. Pandoc читает только .docx (и .rtf, .odt, .ods, .odp).
23. **PDF текст/скан** — На этапе 1 все PDF проверяются через `file_processor.is_scanned_pdf()` (`fitz`). Если суммарный текст всех страниц < 100 символов → файл считается сканом и помещается в `Sorted/PDF_Scan`, иначе → `Sorted/PDF_Text`. На этапе 2 это две независимые группы с отдельным подтверждением.
23. **Этап 2: два независимых подэтапа на формат** — Каждый формат (pdf_text, word_docx, image_jpg, ...) обрабатывается двумя раздельно запускаемыми подэтапами:
    - **A) Извлечение** текста/изображений — своя кнопка Start. Записывает `text`/`image_path` в БД.
    - **B) Эмбеддинги** — отдельная кнопка Start, становится доступной (`pending`) только после завершения извлечения (`extract_status == "completed"`). До этого подэтап `locked`.

    Порядок форматов произвольный; **строго один подэтап за раз** (`_stage2_busy`). Форматы без извлечения (`EMBEDDINGS_ONLY_FORMATS`, сейчас пусто) имеют только подэтап B (`has_extraction == False`, `_format_has_extraction()`). LLM-саммаризация на этапе 2 **удалена** — LLM применяется только на этапе кластеризации.

    Оркестрация: `stage_2_process_formats()` в цикле ждёт `_await_substage_selection()` → выполняет `_run_extraction_substage()` или `_run_embeddings_substage()`. Завершение этапа 2 = у всех групп `embed_status ∈ {completed, skipped}`.

    Состояние под-статусов сохраняется в `PipelineData/stage_2_groups.json` (новая схема `{"version":2,"groups":{fmt:{extract,embed,extract_stats,embed_stats}}}`; старый формат `{completed:[...],stats:{...}}` читается с конвертацией). Эмбеддинги читают извлечённые документы из БД по `format_type`.

    API endpoints:
    - `POST /api/substage/start` body `{"format":"word_docx","substage":"extract"|"embed"}` — запуск подэтапа (при остановленном пайплайне сам резюмит его и дожидается готовности к выбору).
    - `POST /api/substage/rollback` body `{"format":"word_docx","substage":"extract"|"embed"}` — откат подэтапа.
    - `POST /api/substage/stop` — остановка текущего подэтапа (общий `stop_event`).

24. **Откат подэтапов этапа 2** — Раздельный откат, состояние возвращается к «до начала подэтапа»:
    - **extract** (`_rollback_extraction`) — очищает `text/image_path/error` + зависимые эмбеддинги (`text_embeddings`/`image_embeddings` + `.npy`), сбрасывает флаги, возвращает файлы из FailedExtraction/ErrorFiles в `Sorted/{format}` (директория из `FORMAT_TARGETS`, при возврате восстанавливается `format_type`). Статус: extract→`rolled_back`, embed→`locked`.
    - **embed** (`_rollback_embeddings`) — удаляет только эмбеддинги (`text_embeddings`/`image_embeddings` + `.npy`), сбрасывает флаги `*_embedded`; извлечённый текст сохраняется. Статус: embed→`rolled_back` (Start снова доступен).

    Перед каждым запуском подэтапа выполняется его откат (чистый старт), что гарантирует идемпотентность и корректный resume после остановки.

25. **Tesseract OCR для изображений** — Файлы изображений (jpg, png, tiff, bmp, gif) обрабатываются напрямую через Tesseract OCR (`pytesseract`) без конвертации в PDF. Используется `cfg.TESSERACT_PATH` и `cfg.TESSERACT_LANG` (по умолчанию "rus+eng").
26. **Pandoc CLI — основной метод для Excel/PowerPoint** — Конвертация через `cfg.PANDOC_PATH` (абсолютный путь). MS Office COM используется только как fallback для `.doc`/`.docx` после olefile/python-docx. Excel/PPT не используют COM вообще — Pandoc является единственным методом извлечения.
27. **ODF zip validation на этапе 1** — Битые `.odt/.ods/.odp` (zip без центрального каталога) отсеиваются на этапе сортировки через `zipfile.ZipFile(f, 'r')` и сразу попадают в ErrorFiles, не доходя до этапа 2.
28. **Rollback использует пути из БД** — `_rollback_extraction()`/`_rollback_embeddings()` запрашивают `file_path` через `SELECT file_path FROM documents WHERE ...`, а не используют старые пути из `sorted_data`. Это гарантирует возврат файлов из ErrorFiles/FailedExtraction обратно в Sorted/{format}.
29. **Web-сервер восстанавливает битое состояние stage 2** — При старте `web_server.py` проверяет: если stage 2 помечен `completed`, но не у всех групп построены эмбеддинги (`embed_status ∈ {completed, skipped}`) — этап автоматически переводится в `incomplete`.

30. **Качество текста и эмбеддинги** — `good`: текст >100 символов, >30% букв → текстовый эмбеддинг; `poor`: короткий текст или мало букв → текстовый эмбеддинг (всё равно обрабатывается); `none`: текст отсутствует → только image эмбеддинг (если есть изображение).

31. **COM — _kill_zombie_office** — `_kill_zombie_office(app_type)` убивает ТОЛЬКО указанный процесс (`'word'` → WINWORD.EXE, `'excel'` → EXCEL.EXE). Вызывается 1 раз за жизнь процесса-воркера в `_init_worker` пула multiprocessing. Флаг `_zombie_killed` предотвращает повторные kill в рамках процесса. Решает проблему cross-worker killing (воркер A убивал Excel воркера B).

32. **Aspose(Java) — основной метод для Word** — .docx/.doc/.rtf/.odt обрабатываются через Aspose.Words for Java (subprocess JVM). COM Word — только fallback после Aspose. `ASPOSE_WORKERS=8` воркеров обеспечивает высокую производительность.

33. **Multi-page Word extraction** — Word COM (при fallback) извлекает текст со всех страниц документа через `doc.ComputeStatistics(2)` + `Selection.GoTo()`. Таблицы 1x2 игнорируются (служебные метаданные).

34. **PDF текст/скан/таблицы** — На этапе 1 все PDF проверяются: если текст всех страниц < 100 символов → `pdf_scan`; если есть таблицы на любой странице → `pdf_tables`; иначе → `pdf_text`. На этапе 2 это три независимые группы с отдельным извлечением.

---

## Настройка конфигурации

Редактировать `config.py`:
- `SIMILARITY_THRESHOLD` (по умолчанию: 0.70) — Выше = меньше, плотнее кластеров
- `TOP_N_CLUSTERS` (по умолчанию: 10) — Сколько папок кластеров создать
- `MIN_CLUSTER_SIZE` (по умолчанию: 95) — Минимальный размер кластера (меньшие объединяются в "Неклассифицированные")
- `MAX_CONCURRENT_FILES` (по умолчанию: `max(1, cpu_count - 4)`) — Параллельные воркеры извлечения (динамически)
- `LO_MAX_WORKERS` (по умолчанию: 3) — Максимум параллельных процессов LibreOffice
- `COM_ENABLED` (по умолчанию: True) — Использовать MS Office COM (fallback для Word)
- `ASPOSE_WORDS_JAVA_DIR` (по умолчанию: `D:\Yandex.Disk\Aspose\Aspose.Words for Java`) — Путь к JAR + лицензия Aspose.Words for Java
- `ASPOSE_WORKERS` (по умолчанию: 8) — Параллельные воркеры для Word/RTF через Aspose(Java)
- `COM_WORD_WORKERS` (по умолчанию: 2) — Максимум параллельных COM Word процессов как fallback
- `PANDOC_PATH` (по умолчанию: `C:\Program Files\Pandoc\pandoc.exe`) — Абсолютный путь к Pandoc CLI (для Excel/ODF)
- `PANDOC_TIMEOUT` (по умолчанию: 120) — Таймаут в секундах для конвертации через Pandoc
- `EMB_BATCH_SIZE` (по умолчанию: 32) — Размер батча для эмбеддингов
- `EMB_GPU_DEVICE` (по умолчанию: `cuda:1`) — GPU для эмбеддингов
- `LLM_ENABLED` (по умолчанию: True) — Включить/выключить саммаризацию
- `LLM_CONCURRENT_REQUESTS` (по умолчанию: 4) — Параллельные вызовы LLM
- `EMB_DIMENSION` (по умолчанию: 768) — Размерность эмбеддингов (для pgvector)
- `LLM_BACKEND` (по умолчанию: `"transformers"`) — Бэкенд: `"transformers"` (Qwen3.5-4B) или `"llama"` (llama-server)
- `HF_BATCH_SIZE` (по умолчанию: 16) — Размер батча для transformers бэкенда
- `HF_MAX_TOKENS` (по умолчанию: 512) — Максимум новых токенов для transformers

---

## Тестирование и проверка

**Тест одного файла:** Вызвать `process_single_file()` из `file_processor.py` напрямую с Path.

**Проверка эмбеддингов:** Убедиться что модель загружается без ошибки:
```python
from embeddings_engine import get_engine; engine = get_engine()
```

**Проверка Qwen3.5-4B (HF):**
```bash
python test_hf.py          # быстрый тест с выводом в консоль
python test_hf_debug.py    # дамп decoded output в файл
```

**Качество кластеров:** Проверить `report.txt` на статистику сходства (Min/Max/Avg) по кластерам.

---

### Отключение рассуждений (reasoning) в Qwen3.5-4B

Qwen3.5-4B — reasoning модель. По умолчанию шаблон добавляет `<think>\n` перед генерацией. Управляется флагом `enable_thinking` в `apply_chat_template`:

- `enable_thinking=False` — ответ без рассуждений (используется для классификации)
- `enable_thinking=True` — показывает блок `<think>...</think>` перед ответом

Односложные промпты с `ТЕМА:` в конце user-сообщения срабатывают как completion-trick, forcing модель заполнить значение.

Greedy decoding (`do_sample=False`) даёт детерминированный результат для классификации.

---

## Тестовый прогон (2026-06-18)

**Команда:** Playwright auto-test (20 файлов из SourceFiles_test.rar)
**Общее время:** ~6 минут

| Этап | Статус | Результат |
|------|--------|-----------|
| 1. Сортировка | ✅ 20/20 | Все файлы рассортированы по расширениям |
| 2. PDF | ✅ 5/5 | 0 errors, 8.1s |
| 2. IMAGE | ✅ 7/7 | 0 errors, 10.3s |
| 2. WORD | ⚠️ 2/5 | 3 `.doc` в ErrorFiles (Pandoc не поддерживает .doc) |
| 2. EXCEL | ✅ 3/3 | 0 errors, 49.3s |
| 3. LLM саммаризация | ✅ 17/17 | 3 Topics, 11 Types |
| 4. Эмбеддинги | ✅ 17/17 | nomic-embed (768-dim) |
| 5. Кластеризация | ✅ 1 кластер | 17 docs, Similarity avg 0.78 |

### Тестовый прогон 2 (17 файлов Word, 2026-06-18, web_server)

**Команда:** Веб-интерфейс (localhost:8080), запуск через кнопку Start Pipeline
**Общее время:** 141 секунда (2 мин 21 сек)

| Этап | Статус | Результат |
|------|--------|-----------|
| 1. Сортировка | ✅ 17/17 | Все файлы рассортированы по расширениям (Word) |
| 2. Извлечение (WORD) | ✅ 17/17 | Pandoc CLI, 8.3 сек, 2.1 файл/сек |
| 3. LLM саммаризация | ✅ 15/15 | Qwen3.5-4B loaded 4.2s, 9.08 GB VRAM, 15 topics/types |
| 4. Эмбеддинги | ✅ 17/17 | nomic-embed-text на cuda:1 |
| 5. Кластеризация | ✅ 3 кластера | 1 валидный (>=95), все объединены в "Неклассифицированные" |

**Исправленные проблемы:**
- `[Errno 22] Invalid argument` при загрузке Qwen3.5-4B — исправлено добавлением `torch.tensor([0], device=dev)` перед `torch.cuda.synchronize()` в `_ensure_cuda_context()` (`hf_summarizer.py:42-43`, `embeddings_engine.py:74`). Без этого вызова CUDA-контекст не инициализируется в новом потоке, что приводит к OSError.

**Выводы:**
- Pre-flight валидация: pymupdf4llm (optional) → WARNING, остальное → OK
- Pandoc CLI корректно обработал все 17 файлов (12 docx, 2 txt, 1 odt, 1 rtf, 1 doc → ErrorFiles)
- `_ensure_cuda_context()` с принудительным созданием тензора критичен для загрузки моделей из pipeline-потока web_server
- Web-интерфейс: все кнопки подтверждения (Continue/Start/Stop/Rollback) работают стабильно
- 3 центральных документа скопированы в CentralDocuments

### Тестовый прогон 3 (Aspose.Words for Java, 2026-06-20)

**Команда:** Веб-интерфейс (localhost:8080), запуск через кнопку Start Pipeline
**Общее время:** ~4 минуты

| Этап | Статус | Результат |
|------|--------|-----------|
| 1. Сортировка | ✅ Word/RTF/Odt → Sorted/Word | PKCS#7 SignedData + EnvelopedData → ErrorFiles |
| 2. Извлечение (WORD) | ✅ Aspose(Java), 8 воркеров, ~50ms/файл | COM fallback ограничен multiprocessing.Semaphore(2) |
| 3. LLM саммаризация | ✅ Qwen3.5-4B, batch=16 | HF_BATCH_SIZE увеличен с 4 до 16 |
| 4. Эмбеддинги | ✅ nomic-embed-text на cuda:1 | HDBSCAN вместо AgglomerativeClustering |
| 5. Кластеризация | ✅ HDBSCAN, динамический MIN_CLUSTER_SIZE | Без O(N²) матрицы расстояний |

**Исправленные проблемы:**
- Pandoc заменён на Aspose.Words for Java для всех Word-форматов (.docx/.doc/.rtf/.odt)
- PKCS#7 PDF (SignedData + EnvelopedData) отсеиваются на этапе 1 → ErrorFiles
- PDF классификация на 3 категории: pdf_scan, pdf_tables, pdf_text
- HDBSCAN установлен и используется вместо AgglomerativeClustering

**Выводы:**
- Aspose.Words for Java корректно обработал все Word файлы (17/17)
- `ASPOSE_WORKERS=8` обеспечивает высокую производительность без конфликтов COM
- PKCS#7 EnvelopedData и SignedData PDF не доходят до этапа 2 извлечения
- HDBSCAN работает стабильно, кластеризация быстрее на ~50%
