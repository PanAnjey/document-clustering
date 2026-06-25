# SESSION_CONTEXT.md

## Cluster description
Кластеризация документов (33,830 шт) через Qwen3.5-4B (transformers) → nomic-embed → HDBSCAN кластеризация с динамическим минимальным размером кластера.

## Important info for the next session
- **Config override file**: `config_override.json` существует и переопределяет `config.py`. Редактировать ОБА файла.
- **`report.txt`**: `D:\FileOrganizer\report.txt` — всегда отражает последний запуск.
- **FailedExtraction/**: `D:\FileOrganizer\FailedExtraction\` — файлы, из которых не удалось извлечь текст. Возвращаются в Sorted/ при откате этапа 2.
- **LLM backend**: transformers (Qwen3.5-4B на cuda:0), НЕ llama-server
- **PyTorch**: версия 2.12.0+cu132 (установлена через `pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132`)
- **Aspose.Words for Java**: `D:\Yandex.Disk\Aspose\Aspose.Words for Java` — основной метод для Word-форматов (.docx, .doc, .rtf, .odt) через subprocess вызов JVM

---

## Сессия 2026-06-24: Полный рефакторинг пайплайна

### Ключевые исправления:

#### 1. Импорты и глобальное состояние
- `web_server.py`: импорты из `main` → `stages.stage2_processing`
- `main.py`: `PipelineState` из корневого `pipeline_state.py`
- `stage2_processing.py`: lazy import `stop_event`/`progress`/`pipeline`
- `progress_tracker.py`: добавлен `from logger_utils import logger`

#### 2. Этап 1
- `stage1_sorting.py`: `sys.path` для spawn-воркеров
- `excel_processor.py`: `.xls` в `EXCEL_EXT_MAP`
- `main.py`: пустой SourceFiles → fallback на Sorted/

#### 3. Этап 2 (подэтапы)
- `main.py`: `mark_completed`/`save_data`/`wait_for_confirmation`
- `web_server.py`: `select_substage_to_start` всегда
- `stage2_processing.py`: `_pending_substage`
- `web_server.py`: восстановление статусов из `load_stage2_groups()`

#### 4. process_file
- `extractors/__init__.py`: PipelineRegistry для всех форматов

#### 5. Пайплайны
- `pdf_text`: только fitz
- `pdf_scan`: PNG + Radon + Tesseract
- `word_docx`: Aspose(Java)

#### 6. Эмбеддинги
- text+image для pdf_scan
- формат-специфичные таблицы в БД
- `np.any(emb)` — без нулевых векторов

#### 7. Rollback
- очистка `PDF_Images`/`Office_PDF`
- формат-специфичные таблицы

#### 8. Radon transform
- `skimage.transform.radon()` — коррекция 90° + skew
- ~544с / 1975 файлов

### Тесты pdf_scan (1975 файлов)
| Метод | Время | Повороты 90° | Перекосы |
|-------|-------|--------------|----------|
| OpenCV HoughLines | 193с | 0 | 1 |
| OpenCV findContours | 183с | 0 | 57 |
| OpenCV boundingRect | 183с | ~118 | 1 |
| Radon transform | 544с | ~244 | ~62 |
