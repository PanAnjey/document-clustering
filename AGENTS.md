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
   - **PDF классификация** — после сортировки PDF проверяются через `fitz` (ТОЛЬКО первая страница):
     - Читаемых букв (кириллица+латиница) на 1-й странице < 100 → `pdf_scan` (`Sorted/PDF_Scan`)
     - Есть читаемый текст + таблица на 1-й странице → `pdf_tables` (`Sorted/PDF_Tables`)
     - Есть читаемый текст, таблиц нет → `pdf_text` (`Sorted/PDF_Text`)
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
- `cuda:1` (RTX PRO 4000 Blackwell, 24 GB) — эмбеддинги (nomic-embed-text + nomic-embed-vision, ~2 GB)
- Обе карты одинаковые (2× RTX PRO 4000, подтверждено nvidia-smi 2026-07-19) — LLM можно запускать на ЛЮБОЙ из них (2 параллельных экземпляра Qwen для ускорения, шард по id).

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
23. **PDF текст/скан** — На этапе 1 все PDF проверяются через `file_processor.classify_pdf()` (`fitz`) ТОЛЬКО по первой странице. Если читаемых букв (кириллица+латиница) на 1-й странице < 100 → файл считается сканом и помещается в `Sorted/PDF_Scan`, иначе → `Sorted/PDF_Text`/`Sorted/PDF_Tables`. Подсчёт букв (а не всех символов) отсеивает mojibake — PDF со шрифтами без ToUnicode, у которых `get_text()` возвращает мусорные глифы. На этапе 2 это независимые группы с отдельным подтверждением.
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

34. **PDF текст/скан/таблицы** — На этапе 1 все PDF классифицируются ТОЛЬКО по первой странице: если читаемых букв (кириллица+латиница) на 1-й странице < 100 → `pdf_scan`; если есть таблица на 1-й странице → `pdf_tables`; иначе → `pdf_text`. На этапе 2 это три независимые группы с отдельным извлечением.

35. **Анализ только по первой странице (конвенция проекта)** — Для анализа/классификации документов во ВСЕХ форматах используется только первая страница. Это касается classify_pdf(), генерации изображения для image-эмбеддингов (`_generate_first_page_image`) и любых будущих классификаторов. Полное извлечение текста (все страницы) выполняется только на этапе 2 профильным обработчиком формата.

36. **PowerShell 5.1 ломает кириллицу и экранирование** — ВСЕГДА учитывай заранее, не исправляй по ходу:
    - **Чтение .ps1 в cp1251** по умолчанию, литеральная кириллица в `$here = 'D:\...\Кластеризация'` ломает парсер (`UnexpectedToken`). **Решения:** (a) использовать `$PSScriptRoot` вместо литеральных путей; (b) кодировать кириллические строки через `[char]0x041A + [char]0x043B...` (UTF-16 escape); (c) сохранять .ps1 в UTF-8-with-BOM (PowerShell 5.1 уважает BOM) — но всеми силами предпочитать (a) и (b).
    - **Передача кавычек в WSL:** `wsl -- bash -lc "...c.command 'arg'..."` ломается на вложенных одинарных кавычках. **Решение:** писать bash-скрипт во временный .sh файл в проект (`_tmp_*.sh`), затем `wsl -- bash '/mnt/d/.../_tmp.sh'`. После выполнения удалять.
    - **Передача UTF-8 stdin:** по умолчанию `Console.InputEncoding` = cp1251 (или cp866), `subprocess.Popen` без `encoding='utf-8'` даёт mojibake на кириллических путях. **Решение:** явно `Console.InputEncoding = new UTF8Encoding(false); Console.OutputEncoding = new UTF8Encoding(false)` (для .NET-серверов см. `CellsToPdfServer/Program.cs`).
    - **`Get-Content -LiteralPath 'кириллический путь'`** ломается — вместо используй `Get-Content -LiteralPath $cyr` где `$cyr` собрана через `[char]` escape.
    - **`Out-String | Tee-Object`** не работает в 5.1 на путях с кириллицей. Используй `Add-Content -Path $log -Encoding utf8` (по строке) вместо Tee.
    - **schTasks/Get-ScheduledTask с кириллическим `/tn`**: только через `-TaskPath` из `[char]`-escape переменной, не через литерал.
    - **`cmd /c "кириллический.bat"`**: вывод всегда в OEM-cp866, non-deterministic чем PowerShell. Избегать, вызывать .bat только через `Start-Process -FilePath path.bat -Wait` или заменять инструкциями PowerShell.
    - **Кавычки:** НЕ вкладывать одинарные в одинарные (`'\'inner\''`) и не вкладывать двойные в двойные — PowerShell 5.1 не поддерживает here-strings корректно на путях с кириллицей. Использовать внешние `"` и внутренние `'` (или наоборот), а для сложных структур — вынести в heredoc-файл (`.sh`/`.ps1`) и запускать с `-File`.
    - **Более жёсткое правило:** если задача требует вложенных кавычек/кириллицы/спецсимволов — СРАЗУ пиши отдельный `.ps1`/`.sh`/`.py` файл с ASCII-only кодом (пути через `$PSScriptRoot`/`Path(__file__).parent`, кириллица через `[char]` escape или внешние .json/.xml файлы). НЕ пытаться прогнать всё через одну `bash -lc "..."` конструкцию.

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

---

## Эксперимент XML (формализованные документы ФНС, 2026-07-23)

Каталог `experiment_xml/`. Классификация XML-документов из MySQL `events_26.xml` по 26 темам предыдущего эксперимента (`temp_theme2_centroids_v2`) и сравнение с неформализованными (pdf_text/pdf_tables).

**Источник:** MySQL `localhost:3308` (root/mysql), БД `events_26`, таблица `xml` (410,009 строк; колонка `js` пустая). XML в utf8mb4, декларация `encoding="windows-1251"` в шапке устарела — перед парсингом вырезать (`_XML_DECL_RE`).

**Типы (402,178 док, 9 TypeNamedId):** UniversalTransferDocument, Invoice, ReturnInventoryAcceptanceCertificate, StorageInventoryAcceptanceCertificate, PerformedWorkAcceptanceCertificate, PerformedWorkCostCertificate, XmlAcceptanceCertificate (все — ON_NSCHFDOPPR, КНД 1115131) + UniversalCorrectionDocument, InvoiceCorrection (ON_NKORSCHFDOPPR, КНД 1115133, секции СвКСчФ/ТаблКСчФ). Не берутся: ActOfsetting (OAKTNAVZ), ProformaInvoice (ON_SCHET), ReconciliationAct (ON_AKTSVEROTP) — иные форматы, мелкие объёмы.

**Маркированные товары (ГИС МТ):** признак N3=1 в ИдФайл `R_T_A_O_GGGGMMDD_N1..N7`. ВНИМАНИЕ: префикс ON_NSCHFDOPPR сам содержит '_' → при split('_') N3 = parts[7] (0-based), не parts[6]! Проверено: 1,208 файлов, все UTD. После assign выполняется override: `assigned_theme = 90` (MARKED_THEME), отдельный кластер. N2=1 (parts[6], 34 файла) — ДРУГОЕ, не маркировка.

**Этапы и таблицы (PostgreSQL file_organizer_db):**
- `phase_x1_extract_canonical.py` → `temp_xml_canonical` (402,178; 47 сек, 100% parse_ok). V1=`canonical_text` (avg 699), V2=`subject_text`
- `phase_x2_embed_assign.py` → `temp_xml_embeddings` (embedding + subject_embedding BYTEA f32 768d) → `temp_xml_assign` (V1) + `temp_xml_subject_assign` (V2) + marked override
- `phase_x3b_reassign_pdf.py` → `temp_pdftext_assign_v2`, `temp_pdftables_assign_v2` (PDF на полные 26 центроидов)
- `phase_x4_clean_subject.py` → `subject_clean_embedding` + `temp_xml_subject_clean_assign` (V3 = V2 после `strip_contract_refs()`) + marked override
- `phase_x3_report.py` → `experiment_xml/reports/xml_vs_pdf_themes_*.md`; `make_sample_html.py` → `canonical_samples.html`

**Канонизатор** (`xml_canonicalizer.py`): белый список полей, TAG_DICT сокращений, первые 5 строк СведТов с дедупом (×N), ПрТовРаб→слова, ИнфПолФХЖ1/2/3 полностью, ОснПер (+СчФ-ссылка УКД). По правкам пользователя: наименование типа документа НЕ включается (задачи определения типа нет), СодОпер фильтруется по boilerplate-списку («Услуги оказаны в полном объеме» и т.п. — не несут тематики), из адреса только регион+город (`detect_region`/`detect_city` со стоп-листом «муниципальный округ»). УКД: СвКСчФ (НомерКСчФ), ТаблКСчФ (НаимЕдИзмДо, ВсегоОпл/СтТовУчНал@СтоимПослеИзм).

**Эмбеддинги 2×GPU (шардирование):** `phase_x2/x4 --embed-only --device cuda:{0|1} --shards 2 --shard-id {0|1}` — два процесса, шард по `id %% 2`; затем `--skip-embed` для assign. **Роли GPU:** cuda:0 — вычислительная (~186 док/с), cuda:1 — ДИСПЛЕЙНАЯ (десктоп-приложения: explorer, браузеры, LM Studio… ~92 док/с, вдвое медленнее). bf16 autocast + сортировка по длине (xml_embed_helper.py): fp32 13 → bf16 99+ док/с.

**Результаты (порог 0.8, 26 центроидов + кластер 90):**
| Представление | coverage | avg sim |
|---|---|---|
| V1 полный канон | 19.8% | 0.768 |
| V2 тематический | 58.6% | 0.809 |
| V3 темат. очищенный | 54.9% | 0.805 |
| PDF_Text / PDF_Tables | 99.6% | 0.89 / 0.92 |

**Ключевые выводы:**
1. Преобразование XML критично (V1 19.8% vs V2 58.6% coverage); очистка договорных ссылок снимает прилипание к теме 26 (47.7%→14.3%).
2. Топ-темы XML (V3): 20 каналы связи (24.4%), 11 поставка оборудования (22.6%), 26 вознаграждение (14.3%), 7 аренда (11.7%), 23 PR/размещение (4.9%).
3. Новые типы неоднородны: XmlAcceptanceCertificate отлично (5.3% unknown), PerformedWork* средне (28-54%), а ReturnInventory/StorageInventory почти не покрываются существующими темами (93-96% unknown) — складские акты вне тематического пространства → кандидаты на переоткрытие тем из дрейф-буфера (HDBSCAN, как phase_3r).
4. JS-div V3↔PDF_Text 0.253 (cos 0.602), V3↔PDF_Tables 0.209 (cos 0.551) — менее похожи, чем набор только УПД/СФ, из-за складских актов, которых нет среди PDF.

**Переоткрытие тем (phase_x5_drift_themes.py, 2026-07-23):** дрейф-буфер V3 (181,305 unknown) → HDBSCAN на подвыборке 60K (min_cluster_size=150, euclidean, 35 мин) → слияние 0.92 → **30 новых тем**, назначение всего буфера (sim≥0.8): **92.8% покрыто** (168,297), остаточный шум 13,008. Таблицы: `temp_xml_drift_centroids` (new_theme, embedding, size, top_words), `temp_xml_drift_assign` (xml_id, new_theme, similarity). Топ новые темы: 5 модернизация/строительство БС (30.6K), 25 оборудование Huawei (21.5K, складские акты), 11 авиабилеты (19.9K), 23 SFP-модули (9.6K), 14 комплект исполнительной документации (8.2K), 12 возмещение электроэнергии (8.2K), 29/27 оборудование Huawei/Ericsson LTE/UMTS (14K), 9 выручка-аренда (6.6K), 28 антенны (5.9K), 18/10 АКБ Coslight/Narada, 2 сотовые телефоны, 8 wi-fi маршрутизаторы, 19/20/24 оптика (LC/UPC/DLC/CPRI), 17 БС Baicells, 0/1 вознаграждение агента B2C_SP/B2B (sim 0.98-0.999). Разрез по типам: складские акты = железо (Huawei/Ericsson/SFP/антенны/АКБ), PerformedWork = документация/выезды, Invoice/UTD = командировки/электроэнергия/модернизация.

**Второй уровень (phase_x7_second_level.py):** 11 тем с ≥5K документов (96% корпуса) раздроблены MiniBatchKMeans (k=2..8 по формуле N/12000) → `temp_xml_l2_assign` + `temp_xml_l2_centroids`. Объясняет «двойные галактики» на t-SNE: центроидное присвоение скрывает подструктуру (тема 11 → монтаж/размещение/ПРТО-заключения/трафик и т.п.). c-TF-IDF топ-слова считаются после `tfidf_clean()` — вырезание служебных токенов V2 (вид:услуга, ед:шт), иначе топ-слова забиты мусором.

**Визуализация:** `make_embedding_map.py` (PCA+t-SNE PNG), `make_star_map.py` («галактики» на чёрном), `make_star_map_combined.py` (XML+PDF), `make_region_star_map.py` (географическая звёздная карта: документы как звёзды в точках `seller_region`, 96 регионов → координаты центров, джиттер, цвет = тема; если регион в адресе не извлечён — фолбэк на регион по префиксу ИНН продавца→покупателя через `REGION_BY_CODE`, покрытие 100%; `reports/region_star_map.png`), `make_interactive_data.py` + `make_interactive_html.py` (venv `C:\Windows\Temp\opencode\venv_plot` с plotly 6.9; самодостаточный HTML `reports/embedding_map_interactive.html` ~44 МБ / `_light` ~24 МБ). Эмбеддинги PDF-предметов: `temp_pdf_embeddings` (source, doc_id, embedding). Регион продавца уже извлечён канонизатором: `temp_xml_canonical.seller_region`/`seller_city` (100% заполненность, пустые 27K — адрес без региона); регион покупателя бесполезен (99.5% — ИНН 77, сама компания).

---

## Эксперимент Models (сравнение embedding-моделей, 2026-07-24)

Каталог `experiment_models/`. Сравнение 5 моделей для выбора продакшен-модели (русскоязычные документы, сканы через OCR). Baseline — текущая nomic-embed-text-v1.5 (выяснено: **англоязычная**, словарь 30,522 = bert-base-uncased, кириллических токенов 86, из них 8 многосимвольных → русские слова токенизируются посимвольно; семантика RU поверхностная: cos('антенна','банан')=0.58 по общим буквам).

**Модели** (`em_config.py` → MODELS; все: mean-pooling + L2-norm, bf16 autocast): `nomic_v15` (baseline, 768d, prefix `search_document: `), `nomic_v2` (nomic-embed-text-v2-moe, 768d, XLM-R MoE, ~100 яз.), `e5_large` (multilingual-e5-large, 1024d, prefix `passage: `), `rubert_tiny2` (312d, 29M), `sbert_ru` (sbert_large_nlu_ru, 1024d). Скачаны в `D:\MODELS\Transformers` (`download_em_models.py`). **ВАЖНО:** nomic_v2 по `auto_map` в config.json тянет код из репо `nomic-ai/nomic-bert-2048` — скачан отдельно в `D:\MODELS\hf_cache` (без него offline-загрузка падает с LocalEntryNotFoundError). Работает без megablocks (warning, fallback-реализация, ~913 док/с на cuda:0 — достаточно).

**Фазы:**
- `phase_m1_lexical.py` — 14 пар синонимов EN↔RU предметной области + 5 различителей → `reports/m1_lexical.md`
- `phase_m2_sample.py` — выборка **46,501** XML (sim≥0.8 ИЛИ тема 90, cap 3000/тему, 25 тем) → `sample_50k.csv`
- `phase_m3_embed.py` — эмбеддинги, шардирование 2×GPU (`--device cuda:X --shards 2 --shard-id Y`, шард по `id %% 2`), лог каждого процесса в `logs/m3_{model}_shard{N}.log` (faulthandler + traceback), CUDA OOM → автоуменьшение батча → `embeddings/{model}__shard{N}.npy` + `_ids.npy` + `_meta.json`
- `phase_m4_eval.py` — LOO top-1/top-3 по центроидам тем (точный leave-one-out; **смещено в пользу baseline** — метки построены в его пространстве) → `reports/m4_eval.md`
- `phase_m4b_intrinsic.py` — kNN purity@10 + MiniBatchKMeans ARI/NMI (справедливые метрики, центроиды v1.5 не участвуют) → `reports/m4b_intrinsic.md`
- `phase_m5_ocr.py` — устойчивость к OCR: 200 pdf_text, рендер 1-й стр. 200dpi → Tesseract rus+eng; метрики cos(orig,ocr) + identity top-1 (OCR находит свой документ среди 200 оригиналов) → `reports/m5_ocr.md`
- `phase_m5b_ocr_deepseek.py` — M5b: тот же протокол, но парное сравнение **Tesseract vs DeepSeek-OCR** на 200 документах OCR-пилота (`experiment_ocr/results/docs_set.json`, OCR-тексты из `o1_results.jsonl`, повторный OCR не нужен) → `reports/m5b_ocr_deepseek.md`
- `phase_m6_migration.py` — M6: миграция кластеров (Track A центроидное присвоение, Track B HDBSCAN) → `reports/m6_migration_{a,b}.json`
- `phase_m6c_calibrated.py` — M6c: кривая «порог → unknown/точность», рабочие точки acc 90%/95% → `reports/m6c_calibrated.json`
- `phase_m6d_common_core.py` — M6d: контроль к Track A на ОДНИХ И ТЕХ ЖЕ документах (вся выборка + ядро всех 7 моделей) → `reports/m6d_common_core.json`
- `make_report.py` — сводный `reports/models_compare_summary.md`
- `make_word_report.py` — детальный отчёт MS Word `reports/Отчёт_сравнение_embedding_моделей_2026-07-28.docx` (резюме, «как читать отчёт» + шпаргалка, постановка, методология, результаты 4.1-4.5 с графиками `reports/charts/*.png`, выводы, приложения А-Г; читает `reports/*.json`; `THEME_NAMES` — названия 25 тем)

**Результаты (актуальные, пересчёт 2026-07-28 на 7 моделях; старые значения 5-модельного прогона 24.07 — в `reports/*.json.bak5`):**
| Модель | dim | M1 syn cos | M1 top-1 | M1 disc | LOO top-1* | kNN@10 | ARI | NMI | OCR cos | OCR id@1 | док/с (2×GPU) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nomic_v15 | 768 | 0.589 | 14% | 3/5 | **74.0%** | **0.845** | 0.320 | **0.567** | 0.927 | 86.5% | 1511 |
| nomic_v2 | 768 | 0.815 | 86% | 4/5 | 67.0% | 0.819 | **0.360** | 0.551 | 0.908 | 94.5% | 1419 |
| e5_large | 1024 | **0.892** | 86% | 4/5 | 64.7% | 0.799 | 0.272 | 0.478 | **0.964** | 95.0% | 1346 |
| rubert_tiny2 | 312 | 0.663 | 57% | 4/5 | 59.7% | 0.809 | 0.281 | 0.475 | 0.953 | 87.5% | **10076** |
| sbert_ru | 1024 | 0.515 | 21% | 3/5 | 61.6% | 0.815 | 0.348 | 0.515 | 0.923 | 72.5% | 1536 |
| qwen3e_4b | 2560 | 0.807 | **93%** | 4/5 | 68.6% | 0.817 | 0.305 | 0.545 | 0.925 | **95.5%** | 115 |
| qwen3e_06b | 1024 | 0.754 | 79% | 4/5 | 68.5% | 0.816 | 0.353 | 0.553 | 0.891 | 94.0% | 474 |

\* LOO с «домашним» преимуществом baseline (метки из его пространства); справедливые — kNN/ARI/NMI.

**Вывод:** **nomic_v2 (nomic-embed-text-v2-moe) — кандидат в продакшен**: лучший баланс (лучший ARI 0.360, OCR-узнаваемость 94.5%, top-1 86% кросс-языковых терминов, 768d drop-in, те же префиксы `search_document:`/`search_query:`, max_len 512). e5_large сильнее на лексике/OCR-cos, но хуже восстанавливает кластеры (ARI 0.272). rubert_tiny2 — только если критична скорость (10K док/с). Различитель `proforma invoice → счет на оплату` vs `счет-фактура` проходит ТОЛЬКО nomic_v15 (0.580 vs 0.548), и она же FAIL на обратном `invoice → счет-фактура`; остальные 6 моделей FAIL на proforma → надёжно эту пару не различает ни одна модель (закрывать правилами/LLM).

**M5b (2026-07-24, DeepSeek-OCR вместо Tesseract, парный дизайн 200 док):** текст DeepSeek-OCR ближе к оригиналу в векторном пространстве у ВСЕХ моделей (cos +0.009…+0.035: e5 0.973, v15 0.956, sbert 0.954, tiny2 0.951, v2 0.935). identity@1 на DeepSeek: **v2 = e5 = 97.5%**, v15 88.5% (−3.5 п.п. к Tesseract — markdown мешает англ. модели), tiny2 85.5% (+5), sbert 83.0% (+14). Выводы: (1) DeepSeek-OCR как OCR-стек повышает точность попадания сканов в темы; (2) ранжировка embedding-моделей не изменилась (v2 — баланс с учётом ARI).

**Нюансы:**
- `documents.text` для pdf_text **пустая** (текст в БД не хранится) — m5 берёт текст через fitz get_text() 1-й страницы.
- Падение 2-го GPU-процесса (жалоба из experiment_xml) **не воспроизвелось**: 10 пар шардов, все exit 0; cuda:0 ~900-1050 док/с, cuda:1 (дисплейная) ~440-500 док/с. При повторении — смотреть `logs/m3_*.log` (там полный traceback).
- Патчи `PreTrainedModel` (баг all_tied_weights_keys в коде nomic) продублированы в `em_model_loader.py` — обязательны для обеих nomic.

**Добавление Qwen3-Embedding-4B/0.6B (2026-07-28):** обе модели в `em_config.py` (`qwen3e_4b` 2560d, `qwen3e_06b` 1024d). **КРИТИЧНО: pooling `last_token`** (не mean!) — реализовано в `em_model_loader.last_token_pool` (правый паддинг → позиция `mask.sum()-1`); документы БЕЗ префикса (инструкция только для query-стороны ретрива). Скачивание `_download_qwen3e.py` (~9 GB). Результаты: M1 top-1 **93%/79%** (4B — лучший из всех), LOO 68.6%/68.5% (2-3 место после «домашнего» baseline), kNN@10 0.817/0.816, ARI 0.305/**0.353** (0.6B ≈ nomic_v2 0.360!), OCR id@1 **95.5%**/94.0% (4B — лучший), скорость **115/474 док/с** (vs nomic_v2 1419), VRAM 12.6/11.0 GB. Обе FAIL на различителе `proforma invoice → счет на оплату` (как и все, кроме nomic_v15 — но та FAIL на обратной проверке `invoice`). Токенизатор Qwen3: 4,129 кириллических токенов (3,371 многосимвольных) vs 86/8 у nomic_v15 — русские слова → 2-4 сабворда, не посимвольно.

**M6 — миграция кластеров между моделями (phase_m6_migration.py, 2026-07-28):** ответ на вопрос «переползают ли документы между кластерами при смене модели». Track A (центроидное присвоение 25 тем, продакшен-порог 0.8): (1) **coverage на пороге 0.8 несравнимо**: qwen3e 39-40% vs nomic_v15 99.6% vs e5 100% — косинусные шкалы разные, **порог надо калибровать заново под каждую модель** (qwen3e симы сжаты ~0.6-0.8; при этом на пороге 0.8 их точность 94-95% против 74% у v15); (2) coverage 99.6% у v15 частично тавтологичен (выборка отобрана sim≥0.8 в её же пространстве), а `diff_vs_v3` = 25.9% даже у самой v15 — центроиды пересчитаны на сбалансированной выборке (cap 3000/тему); (3) **парные % переездов из `m6_migration_a.json` НЕЛЬЗЯ сравнивать между парами** — каждая пара считалась на своём ядре (18K…46K док.), см. M6d. Track B (HDBSCAN min_cluster_size=95, euclidean на сфере=cosine, eps=√(2·0.3)): на этом корпусе HDBSCAN находит 2-9 МАКРОкластеров (не темы): v15→2 (шум 1.6%), v2→5, 4b→6, 06b→9; структуры семейств различны (ARI v15↔qwen 0.06-0.08), внутри семейств согласованы (qwen4b↔qwen06b ARI(cl)=0.931, v2↔qwen 0.82-0.84). Вывод: тематическая гранулярность держится на центроидном присвоении, а не на HDBSCAN; **смена модели без перекалибровки порога = массовый уход в unknown, а не перестановка тем**. Отчёт: `reports/models_compare_summary.md` (7 моделей + M6), `m6_migration_{a,b}.json`. HDBSCAN под базовым Python312 (в `D:\VENV\LLM` hdbscan НЕТ).

**M6d — КОНТРОЛЬ к Track A (phase_m6d_common_core.py, 2026-07-28):** проверка вывода «на ядре модели согласны, а классические переставляют 25-28%». Пересчёт на ОДНИХ И ТЕХ ЖЕ документах для всех 21 пары: (а) вся выборка 46 501 (top-1 без порога), (б) ядро всех 7 моделей = 17 021 док. (36.6%, sim≥0.8 у каждой). Результат **опровергает прежний вывод**: на ядре ВСЕ пары согласны (moved 0.3-5.4%, ARI 0.93-0.99, в т.ч. v15↔nomic_v2 2.1%, v15↔rubert 4.6%), на полной выборке ЛЮБАЯ смена модели переставляет 14.5-33.5% (nomic_v2↔qwen3e_06b: 15.7% против 0.5% на узком ядре). Расхождение с V3 на ядре 8.1-12.5% (qwen4b 8.1, v2 9.2, v15 9.7) против 25.9-40.2% на всей выборке. **Вывод: устойчивость разметки определяется уверенностью присвоения, а не выбором модели** → двухконтурная схема (автоматика на ядре ~1/3 корпуса, правила/LLM на спорной части). Файл: `reports/m6d_common_core.json`.

**M7 — MRL-усечение размерности Qwen3-Embedding (phase_m7_mrl.py, 2026-07-28):** Qwen3-Embedding поддерживает Matryoshka (MRL) — вектор урезается до любой длины 32..2560 без переобучения (первые N координат + L2-norm). Тест на готовых эмбеддингах (инференс не нужен, протокол = M4/M4b/M6c): **потеря при усечении 4B 2560→512 — всего 1 п.п. LOO (68.6→67.6%), при этом ARI и unknown@90 немного УЛУЧШАЮТСЯ** (0.311→0.362, 44.2→44.9%) — MRL-префикс чуть «чище». 4B можно хранить как 512d (в 5 раз меньше места) почти бесплатно. Для сравнения у nomic_v2 (не MRL) усечение 768→512 дороже: ARI 0.360→0.305, unknown@90 50.8→52.2%. Усечение НЕ ускоряет инференс — урезается только вектор. Файл: `reports/m7_mrl.json`, раздел 4.3.2 Word-отчёта.

**Word-отчёт обновлён (2026-07-28):** `make_word_report.py` перегенерирован на 7 моделей + секция 4.5 «Миграция кластеров (M6)» (Track A/B, топ переходов референс→Qwen3) → `reports/Отчёт_сравнение_embedding_моделей_2026-07-28.docx` (рейтинг: 1. nomic_v2, 2. qwen3e_4b, 3. qwen3e_06b, 4. e5_large; M5-таблица пересчитана парно на новых 200 документах; M5b без Qwen — прогон до их добавления). Ревизия по замечаниям пользователя: добавлен раздел «Термины и определения» (после оглавления, 16 терминов), переписано описание ARI/NMI (что обучается KMeans и откуда эталонные темы — метки V3 в пространстве v15), выводы переориентированы на цель «плотные кластеры + минимальный unknown» (не скорость). **M6c (phase_m6c_calibrated.py)**: калиброванный unknown — при точности присвоения 90% лучшие qwen3e_4b/06b (unknown ~44%, порог ~0.70-0.72), nomic_v2 50.8% (порог 0.840), e5 66%; при строгих 95% лучшие sbert_ru 61.6%/rubert 65.4%, qwen3e 83-86%. ВАЖНО: unknown считается от тестовой выборки (46 501, сама отобрана как уверенно размеченная), это не прогноз для всего потока.

**Word-отчёт, редакция 2 (2026-07-28, вычитка на ошибки + изложение для руководителей):** сверка всех чисел отчёта с `reports/*.json` выявила и исправлено: (1) ARI e5_large 0.268 → **0.272** (устаревшее значение 5-модельного прогона в тексте при верном значении в таблице); (2) «ни одна модель не прошла различитель proforma invoice» → на самом деле прошла nomic_v15 (6 из 7 FAIL); (3) nomic_v2 unknown@90 «второй результат» → третий (после обеих qwen3e); (4) rubert_tiny2 «в 50 раз компактнее» → в 16 раз (29M против 475M у nomic_v2); (5) «темы с <30 док.: 12,13,15,22,24» → фактически 12 (26), 22 (5), 24 (12); 13=50, 15=31; (6) «классические ~450-1050 док/с на карту» → 437-1050, а rubert_tiny2 4100-6000; (7) «средняя длина XML ~700 символов» → в тесте эмбеддится `subject_text`, среднее **165** символов (700 — это V1 canonical_text другого эксперимента); (8) max_len v1.5 «8192» → 8192 по спецификации, в тесте 2048; (9) MTEB 4B: 70.58 — это 8B (лидер серии), у 4B 69.45; (10) вывод «ядро совпадает на 99% у сильных моделей» заменён на результат M6d. Добавлено: раздел «Как читать этот отчёт» + шпаргалка по показателям, колонка «точность на старой планке 0.8» (acc_at_t080), раздел 4.5.2 (контроль M6d), названия тем вместо номеров в таблицах миграции (`THEME_NAMES` из `temp_theme2_centroids_v2.top_words`), Приложение В (перечень 25 тем с размерами) и Приложение Г (технические определения), раздел 4.3.2 (M7, MRL-усечение размерности Qwen3-Embedding). Терминология в тексте переведена на управленческий язык (unknown → «без темы», порог → «планка близости», coverage → «отнесено к темам», LOO/ARI/kNN — в скобках рядом с простыми названиями). **Термин «эмбеддинг» НЕ заменять** — устоявшееся понятие (правка пользователя 28.07). **Формулировки про nomic v1.5:** нет «нынешней/продакшен-модели» и «замены модели» — продакшена и внедрённого решения нет, идёт этап оценки и выбора; v1.5 = «первая опробованная модель / точка отсчёта (референс), в её пространстве получена эталонная разметка», порог 0.8 = «значение из предыдущих экспериментов», M6 = «зависимость разметки от выбора модели» (не «миграция при смене модели»).

---

## Эксперимент OCR (пилот движков, 2026-07-24)

Каталог `experiment_ocr/`. Сравнение **Tesseract** (текущий стек) vs **PaddleOCR** для проектируемого пайплайна (русские сканы, таблицы). Движки подключаемые: `engines/oc_{tesseract,paddle}.py` (класс с `.ocr(png_path) -> str`) — можно добавить DeepSeek-OCR/Nanonets-OCR2.

**Окружение:** venv `C:\Windows\Temp\opencode\venv_ocr` (БЕЗ system-site-packages; paddlepaddle **3.2.0** + paddleocr 3.x + rapidfuzz + pymupdf + pytesseract + psycopg2-binary). Все скрипты запускать этим интерпретатором.

**Грабли (важно):**
1. **venv в пути с кириллицей сломан** (`experiment_ocr/.venv`): pip молча ставит пакеты в base-python. venv создавать ТОЛЬКО в ASCII-путях (прецедент: `venv_plot`, `venv_ocr` в `C:\Windows\Temp\opencode`). Случайно загрязнённый base очищен (`pip uninstall paddleocr paddlepaddle paddlex modelscope rapidfuzz`, sanity-импорты OK).
2. **paddle 3.3.x + PP-OCRv5 на CPU**: `NotImplementedError ConvertPirAttribute2RuntimeAttribute` (баг oneDNN-бэкенда, SO 79884564). Обход `PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=0` работает, но 71 с/стр (неюзабельно). **Решение: paddlepaddle==3.2.0** (~6 с/стр).
3. paddleocr 3.x: `use_textline_orientation=True` → native crash процесса (abort после загрузки моделей, без traceback). Использовать `use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False`. Модели: `PP-OCRv5_server_det` + `eslav_PP-OCRv5_mobile_rec` (кириллица), кэш `%USERPROFILE%\.paddlex`.
4. **CER/WER вводит в заблуждение на табличных документах**: порядок текста у fitz и OCR различается → CER ~0.33 при визуально идеальном OCR. Главные метрики — порядко-независимые **BoW-F1** (мультимножество слов) и **3gram-cos** (`oc_metrics.py`).

**Методология:** 200 pdf_text (эталон = fitz get_text 1-й стр., рендер 300 dpi) + 10 pdf_scan (глазная проверка). `phase_o1_run.py` (прогон, resume по JSONL), `phase_o2_eval.py` (метрики + худшие примеры + `reports/scan_samples.md`). `oc_pair_analysis.py` — парный анализ ошибок deepseek vs paddle (количество/характер слов-ошибок, подмены кириллицы, repetition-loop), используется `phase_o2_eval.py` (раздел в `ocr_compare.md`) и `make_word_report.py` (раздел 4.7 Word-отчёта).

**Результаты (2026-07-24, 4 движка + qwen3vl 2026-07-27, парный дизайн — одни и те же 200 документов; набор зафиксирован в `results/docs_set.json`):**
| Движок | BoW-F1 avg | BoW-F1 med | 3gram-cos | с/стр | Инференс |
|---|---|---|---|---|---|
| qwen3vl | **0.931** | **0.979** | **0.938** | 17.1 (медиана) | GPU 2×, bf16, 8B |
| nanonets | 0.906 | 0.957 | 0.927 | 18.6 | GPU cuda:0, bf16 |
| qwen3vl4b | 0.900 | 0.944 | 0.923 | 14.6 (медиана) | GPU 2×, bf16, 4B, rep-penalty 1.15 |
| deepseek2 | 0.887 | 0.948 | 0.919 | 6.3 (медиана) | GPU cuda:0, bf16, 3B MoE |
| finereader | 0.898 | 0.948 | 0.932 | 1.3 (avg, батч) | CPU, ABBYY FR16 десктоп |
| gemma4 | 0.879 | 0.921 | 0.904 | 21.4 (медиана) | GPU 2×, bf16, E4B ~7.6B |
| deepseek | 0.901 | 0.943 | 0.919 | **8.2** | GPU cuda:0, bf16 |
| paddle | 0.823 | 0.864 | 0.891 | 6.2 | CPU |
| tesseract | 0.758 | 0.870 | 0.852 | 1.1 | CPU |

**Qwen3-VL-8B как OCR (2026-07-27, `engines/oc_qwen3vl.py`, отчёт `reports/qwen3vl_vs_deepseek.md`):** лучшее качество из всех движков; парно vs deepseek: win 62%/tie 21%/loss 18%, слов-ошибок медиана 2 vs 6, цифры потери 16.5% vs 25.4%, repetition-loop 11 vs 2 (2 катастрофы F1=0.00 — HTML/SVG-галлюцинация base64, ловятся фильтром покрытия>1.5). **ГРАБЛЯ: критичен промпт** — nanonets-стиль («extract text, tables in html») → извлекает ТОЛЬКО таблицу, пропуская шапку/тело письма (F1 0.36); нужен «Transcribe the entire document… include ALL elements». venv_ocr обновлён до transformers **4.57.6** (Qwen3-VL с 4.57). Прогон `phase_o1b_qwen3vl.py` (шард 2×GPU 60/40, свои jsonl → слияние в o1_results.jsonl; бэкап v1-промпта `o1_results.jsonl.bak_qwen3vl_v1`). Устойчивость к повороту НЕ тестировалась (следующий шаг — phase_o3 на qwen3vl). Скорость 17.1 с/стр медиана (mean 39.6 раздут зацикливаниями до 4096 токенов).

**Qwen3-VL-4B (2026-07-27, `engines/oc_qwen3vl4b.py`):** «золотой серединой» НЕ является — качество ровно 3B-класс (0.900/0.944, парно vs deepseek: ничья, delta −0.0002; vs 8B: loss 68%), но в 3 раза медленнее DeepSeek (14.6 vs 4.6 с/стр: dense 4B против MoE 570M-активных) и шумнее (mixed-script 103 vs 33, лупы 8 vs 2). **ГРАБЛЯ 4B:** без `repetition_penalty=1.15` на бланках цикл `<img src="https://i.imgur.com/...">` до max tokens (F1=0.00); промпт «не выводи img» не помогает, rep-penalty лечит полностью (на нормальных документах качество не меняется). Вывод: при равном качестве DeepSeek остаётся лучшим 3B-классом; Qwen3-VL имеет смысл только от 8B.

**Gemma-4-E4B-it (2026-07-27, `engines/oc_gemma4.py`, google/gemma-4-E4B-it ~7.6B, transformers 5.14.1, venv_gemma):** конкурентом НЕ является — ниже DeepSeek по всему (0.879/0.922, парно: loss 55%, delta −0.022; vs qwen3vl: loss 79%), пропуски слов 4070 vs 3019, цифры 30.2% («0→01», худший VLM), лупы 10, p90 ошибок 70, в 4.7 раза медленнее DeepSeek (21.4 vs 4.6 с/стр). Плюс — чистый plain-text без галлюцинаций. Крупные gemma-4 (12B/26B-A4B/31B) в 24 GB не влезают. gemma4 требует transformers ≥5.x (отдельный venv_gemma; в venv_ocr 4.57.6 класса Gemma4 нет). Промпт v2, шард `phase_o1b_qwen3vl.py --engine gemma4`.

**Инцидент 2026-07-27 — повреждение venv:** оба venv (venv_ocr, venv_dsocr) оказались с выборочно удалёнными файлами внутри десятков пакетов (pip, setuptools, tqdm, huggingface_hub, requests, numpy, torch…). Причина не установлена. Оба venv **пересозданы с нуля** (venv_ocr: torch 2.13.0+cu132, transformers 4.57.6, paddlepaddle 3.2.0, paddleocr, rapidfuzz, pymupdf, pytesseract, psycopg2-binary, python-docx, matplotlib, pywinauto; venv_dsocr: torch 2.13.0+cu132, transformers 4.46.3, einops, addict, easydict, pymupdf, psycopg2-binary) + новый **venv_gemma** (transformers 5.14.1). Старые переименованы в `venv_*_broken` (можно удалить). Проверено: метрики phase_o2_eval воспроизводятся.

**DeepSeek-OCR-2 (2026-07-27, `engines/oc_deepseek2.py`, релиз 27.01.2026, энкодер Visual Causal Flow, 3B MoE):** апгрейда НЕТ — на RU-корпусе хуже v1 (0.887/0.948 vs 0.901/0.944, парно: loss 45% vs win 32%), markdown-мусор ×2.2 («всего→\*\*всего», 3982 шумовых слов), подмены кириллицы ×4.5 (mixed-script 147 vs 33; «Толстобров→Толстодроб», «сети→селу», укр. «І»), лупы 6 vs 2, медленнее (6.3 vs 4.6 с/стр). Плюс только в цифрах (22.9% vs 25.4%). EN-бенчмарки (olmOCR-bench 76.3) на RU не переносятся. **ГРАБЛЯ:** base_size≠1024 ломает кастомный deepencoderv2 (`UnboundLocalError: param_img`) — разрешение зашито (base 1024 + тайлы 768). infer() API идентичен v1, venv_dsocr (transformers 4.46.3) подходит. Вывод: в 3B-классе остаётся DeepSeek-OCR v1.

**ABBYY FineReader 16 десктоп (2026-07-27, `phase_o1c_finereader.py`):** мёртвая ничья с DeepSeek (F1 0.898/0.949 vs 0.901/0.944, win 40%/loss 42%) при 1.3 с/стр на CPU (батч 210 файлов за 280 с). Лучшая полнота (1889 пропусков слов — минимум из всех), самый чистый текст (mixed-script 9), цифры 18.8% (vs 25.4% у DeepSeek), нулевые катастрофы, лучший хвост (0.559). Ошибки — регистровые подмены на стилизованных гарнитурах («сотовой→сотобой», «в→б/6»). **ГРАБЛИ автоматизации:** у десктопной FR16 нет API и пакетного CLI (проверено strings в бинарях); COM только DebugAutomation; FREngine SDK — отдельная лицензия. Рабочий путь — **Hot Folder** (HotFolder.exe): задача `ocr_experiment` (Run once, `C:\Windows\Temp\opencode\fr_in` → `fr_out`, Text UTF-8 с BOM, [F].txt, Russian and English). Создаётся UI-автоматизацией pywinauto: AWL-контролы (AWL:...) невидимы в UIA/win32 (панель пустая), Invoke() падает COMError — навигация ТОЛЬКО скриншотами (PIL ImageGrab) + координатные `pywinauto.mouse.click` + send_keys. Координаты привязаны к геометрии окна (680,374)-(1480,990); кнопка Start Now в главном окне (308,233), строка задачи (400,179). Лог — UTF-16 «Hot Folder Log.txt». Скорость — средняя по батчу (пофайлового тайминга нет). Вывод: качество = DeepSeek без GPU, но лицензия коммерческая и автоматизация хрупкая — роль эталона, не прод-стека.

Медианы классических движков близки (0.86-0.87), но у них тяжёлый хвост (систематические кириллические замены: №→ne, ООО→000; средний F1 на 20 худших: deepseek 0.55, nanonets 0.48, paddle 0.45, tesseract 0.32). VLM лучше и по среднему, и по медиане, плюс сохраняют структуру таблиц (markdown/html). Типичные провалы VLM: repetition-loop на 1-3 документах (coverage 13-23×, F1~0) — лечится фильтром по coverage. 1/200: paddle пустой вывод на счёте.

**Вывод:** **DeepSeek-OCR — рекомендуется для проектируемого пайплайна**: качество на уровне лидера (ΔF1 0.005 к nanonets), лучший хвост (0.55), в 2.3× быстрее (8.2 с/стр). Nanonets-OCR2-3B — если важнее максимальная полнота на типичном документе (медиана 0.957). Если GPU недоступен — PaddleOCR (CPU). Пропускная способность: 14K pdf_scan ≈ **32 GPU·ч (deepseek) / 72 GPU·ч (nanonets)**; на 2 GPU вдвое быстрее; vLLM-инференс — перспектива кратного ускорения (не тестировался). Отчёт: `reports/ocr_compare.md` + `scan_samples.md` + Word `reports/Отчёт_пилот_OCR_движков_2026-07-24.docx` (генератор `make_word_report.py`, читает `results/o1_results.jsonl`).

**Методологическая грабля 5:** первые прогоны VLM шли на СЛУЧАЙНО РАЗНЫХ выборках (`ORDER BY random()` при каждом запуске) — сравнение было непарным. Решение: единый набор документов в `results/docs_set.json` (создаётся при первом запуске, все движки прогоняются на нём же).

**Находка: PDF с битой кодировкой текстового слоя.** Документ dc06052e (CAD-чертёж БС, «25-10485G9L9L18-ВЭС.pdf») попал в худшие-5 у tesseract/paddle (F1 0.05-0.07), но это ЛОЖНЫЙ провал: его встроенный слой — мусор вида «Ǌаǰоǫая Ǻтанция» (ГОСТ/SHX-шрифт без ToUnicode), а OCR-движки прочитали документ ПРАВИЛЬНО. На этапе 1 `classify_pdf` такие документы проходят фильтр ≥100 букв (символы Latin Extended-B Ǌǰǫ — формально латинские буквы) и попадают в pdf_text, хотя текст извлечь нельзя. Доля в выборке: 1 из 200 (0.5%), в тех. категориях может быть выше. Для проектируемого пайплайна нужен **детектор битой кодировки** (доля букв вне базовой кириллицы/латиницы > ~30%) → перенаправление в OCR-ветку (pdf_scan).

**Тест устойчивости к повороту (phase_o3_rotation.py, 2026-07-24):** 20 документов × 90°/180° × 4 движка (`results/rotation_test.jsonl`). **180° не читает НИ ОДИН движок** (F1 ≤ 0.13); при 90°: paddle 0.81 / tesseract 0.78 / **nanonets 0.78 (лучший VLM)**, а **deepseek полностью деградирует (0.09, 0/20)** — его crop-пайплайн не переносит поворот. Вывод: **коррекция ориентации перед OCR обязательна** (готовые модели проекта: Qwen3-VL+LoRA 99.83% или лёгкие ViT-Large/SigLIP2/EfficientNet из TRAIN). Архитектуры энкодеров: nanonets = Qwen2.5-VL (ViT, подтверждено конфигом Qwen2_5_VLForConditionalGeneration — в экспериментах по ориентации этот энкодер слаб на классе 180°: 81.7% после LoRA, тогда как Qwen3-VL с SigLIP-2 — 100%); deepseek = DeepEncoder (SAM+CLIP гибрид, 16× компрессор).

**Связка выполнена (2026-07-24):** M5-тест устойчивости эмбеддингов прогнан на тексте DeepSeek-OCR (парно с Tesseract на тех же 200 документах) — см. `experiment_models/phase_m5b_ocr_deepseek.py` и раздел M5b в experiment_models.

**Парный тест классификаторов ориентации (phase_o4_orientation.py + phase_o4_eval.py, 2026-07-24):** те же 20 документов × 0°/90°/180°/270° (`results/orientation_cls.jsonl` → `reports/orientation_cls.md`):
| Классификатор | Overall | 0° | 90° | 180° | 270° |
|---|---|---|---|---|---|
| **EfficientNet-B1** (проектная) | **96.2%** | 100% | 95% | 100% | 90% |
| ViT-Large 384 (проектная) | 92.5% | 100% | 85% | 90% | 95% |
| PP-LCNet doc_ori (PaddleOCR) | 93.8% | 95% | 90% | 95% | 95% |
Ошибки EfficientNet — только 90↔270, с низкой уверенностью (conf err 0.66 vs ok 0.96) → прод-схема: **EfficientNet (conf ≥ ~0.8) → fallback на Qwen3-VL+LoRA (99.83%)** при низкой уверенности. **ГРАБЛЯ: чекпоинт `orientation_vit_large_384_best.pth` имеет классы в лексикографическом порядке [0,180,270,90]** (артефакт строковой сортировки в retrain-линейке) — при стандартном маппинге [0,90,180,270] молча выдаёт неверные углы (26%!); в phase_o4 применён remap. **Gemma 3 4B и MiniCPM-V-2_6 — gated repos (403)**, без HF-токена с принятой лицензией не скачать. Загрузка ViT: timm 1.0.27 (base python), в 1.0.28 сломан unpickle кастомного Attention (`'Attention' object has no attribute 'gate'`).

**Репрезентативный тест ориентации (phase_o4b_*, 2026-07-24):** 500 pdf_text × 4 угла (150 dpi, `results/orientation_eval_set.json`, `orientation_cls_large.jsonl` → `reports/orientation_cls_large.md`), Gemma — подвыборка 100:
| Классификатор | Overall | прогонов/с |
|---|---|---|
| **PP-LCNet doc_ori** | **97.6%** | 64 |
| EfficientNet-B1 | 97.0% | 47 |
| ViT-Large (remap) | 94.0% | 12 |
| Gemma 3 4B zero-shot | **25.8%** (дегенеративный класс: 96% → «90») | 3 |
Ошибки лидеров в основном 90↔270 (EffNet) и 180→0/270→90 (PP-LCNet). Вывод: zero-shot VLM для ориентации непригодны (Gemma 25.8% ≈ случайный, Qwen3.5-35B 56.9%); прод-схема: **PP-LCNet или EfficientNet → fallback Qwen3-VL+LoRA (99.83%) при низкой уверенности**. Gemma 3 4B скачана по HF-токену пользователя (лицензия принята на его аккаунте; токен в коде/файлах НЕ хранить).

**Перезапуск Qwen3-VL-8B+LoRA на новом датасете (orient2, 2026-07-28, `D:\FileOrganizer\TRAIN\qwen_orient`):** датасет `TRAIN\dataset\distrib_orient\{0,90,180,270}` = 1687 файлов / **1579 уникальных документов** (108 повёрнутых близнецов в 2 классах — переименованы `__dup{cls}`); те же документы+метки, что в v1, но ПЕРЕРЕНДЕРЕННЫЕ изображения (~1190×1684). Prep `prep_dataset_orient2.py`: cv2.INTER_LANCZOS4, factor-32, **TARGET_LONGEST_SIDE=1280** (1760 НЕ влез в 24GB при bs=2 — CUDA paging, 300+ с/шаг, ETA 48-70ч, убито), **сплит 90/10 по документу** (близнецы не разлучаются; файловый сплит дал бы утечку): train 1518 / val 169, без аугментации. Обучение `train_lora_qwen3vl_orient2.py` (копия prep-варианта, venv `D:\VENV\LLM`, transformers 5.4.0): 760/760 шагов за 10:26ч (~49 с/шаг), 8 эпох, best eval_loss 0.0145 (checkpoint-200; далее рост до 0.024 — переобучение), final = best. **Результат на val_orient2 (169): новый адаптер 98.82% (2 ошибки), СТАРЫЙ v1-адаптер 99.41% (1 ошибка)** — статистическая ничья; общий трудный файл «КОПИИ УПД 01.26г.__dup270» (оба: 270→0), у нового ещё паспорт 90→270. v1-адаптер остаётся продакшен-кандидатом. **ГРАБЛИ:** (1) `D:\VENV\LLM` — Visual Studio venv-шим: `Scripts\python.exe` это stub (~3MB), реальный процесс — base Python312-дочерний с venv-пакетами через env; при убийстве чистить ОБА + spawn_main-воркеров. (2) Параллельный eval второй 8B-модели рядом с обучением НЕВОЗМОЖЕН: обучение держит 33GB приватной RAM, Windows commit-лимит (96/123GB, pagefile system-managed) → `os error 1455` в safe_open; `SAFETENSORS_FAST_GPU=1` откладывает падение, но процесс убивается ОС — ждать конца обучения. (3) `Start-Process` из инструментального шелла блокируется (ChildProcess.kill), но дети ВЫЖИВАЮТ сиротами — запуск долгих задач только через `subprocess.Popen` с `DETACHED_PROCESS|CREATE_BREAKAWAY_FROM_JOB` и флагом `-u` (иначе stdout буферизуется 8KB и лог пуст).

**VLM-движки: окружение и грабли (2026-07-24):**
- **Два venv**: `venv_ocr` (tesseract/paddle/**nanonets**, transformers 4.56.2) и `venv_dsocr` (**deepseek**, transformers **4.46.3**) — DeepSeek-OCR несовместим с transformers ≥4.48 (импорт `LlamaFlashAttention2`, `DynamicCache.get_max_length`, старый `LlamaAttention.forward`), а Qwen2.5-VL (nanonets) требует ≥4.49. Оба venv в ASCII-путях `C:\Windows\Temp\opencode\`.
- **DeepSeek-OCR патчи** (в `D:\MODELS\Transformers\deepseek-ai_DeepSeek-OCR`, обратно-совместимые, в venv_dsocr не обязательны, но не мешают): guard на импорт `LlamaFlashAttention2` и на `get_max_length()` (→ None). `infer()`: текст возвращает ТОЛЬКО с `eval_mode=True`; `output_path` обязателен (реальный каталог, `os.makedirs` безусловно).
- **Nanonets-OCR2-3B**: transformers 4.56.2 (4.51.3 падает `GenerationConfig.from_model_config: dict has no to_dict`). Обязателен даунскейл `MAX_SIDE=1600` (`oc_nanonets.py`) — рендер 300 dpi даёт ~16 GB активаций → OOM даже на 24 GB. Промпт — из model card (markdown/html таблицы).
- **torch в venv**: `pip install torch==2.12.1 torchvision --index-url https://download.pytorch.org/whl/cu132` (PyPI-версия = CPU-only!). transformers 5.x ломает DeepSeek-OCR; 4.51.3 ломает nanonets config.
- markdown/html-разметка VLM срезается в `oc_textnorm.normalize` при подсчёте метрик (теги, `|#`, `---`).
