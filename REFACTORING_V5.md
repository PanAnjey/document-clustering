# File Organizer Pipeline v5.0 (Refactored)

## 📁 Новая структура проекта

```
D:\Yandex.Disk\PYTHON\NLTK\Кластеризация\
├── main.py                          # Оркестратор пайплайна (entry point)
├── config.py                        # Конфигурация системы
├── logger_utils.py                  # Настройка логирования
├── database.py                      # PostgreSQL + pgvector
├── cluster_engine.py                # Кластеризация HDBSCAN
├── embeddings_engine.py             # Генерация эмбеддингов
│
├── stages/                          # Модули этапов пайплайна
│   ├── __init__.py
│   ├── stage1_sorting.py            # Этап 1: Сортировка файлов
│   ├── stage2_processing.py         # Этап 2: Извлечение + эмбеддинги
│   └── stage3_clustering.py         # Этап 3: Кластеризация и уточнение
│
├── state/                           # Управление состоянием
│   ├── __init__.py
│   ├── pipeline_state.py            # Сохранение состояния на диск
│   └── progress_tracker.py          # Отслеживание прогресса
│
├── utils/                           # Вспомогательные утилиты
│   ├── __init__.py
│   ├── format_groups.py             # Управление группами форматов
│   └── rollback_manager.py          # Откат операций
│
├── sorting/                         # Обработчики сортировки
│   ├── pdf_processor.py
│   ├── text_processor.py
│   ├── excel_processor.py
│   ├── image_processor.py
│   └── xml_processor.py
│
├── extractors/                      # Извлечение текста из файлов
│   └── ...
│
├── pipelines/                       # Дополнительные пайплайны
│   └── ...
│
└── [остальные файлы и директории]
```

## 🎯 Основные изменения в v5.0

### 1. **Разделение ответственности**
- `main.py` стал оркестратором (~200 строк вместо ~2000)
- Каждый этап пайплайна вынесен в отдельный модуль
- Состояние и прогресс инкапсулированы в классы

### 2. **Улучшенная архитектура**
```python
# До рефакторинга:
main.py (2000 строк) - все функции вместе

# После рефакторинга:
main.py (orchestrator) → вызывает модули этапов
├── stages/stage1_sorting.py
├── stages/stage2_processing.py
└── stages/stage3_clustering.py
```

### 3. **Глобальные экземпляры**
- `get_pipeline_state()` - доступ к состоянию пайплайна
- `get_progress_tracker()` - отслеживание прогресса
- `get_format_groups_manager()` - управление группами форматов
- `get_rollback_manager()` - откат операций

### 4. **Совместимость**
Все старые вызовы работают через глобальные функции:
```python
# Старый код:
pipeline = PipelineState()
progress = ProgressTracker()

# Новый код (совместимый):
from state.pipeline_state import get_pipeline_state
from state.progress_tracker import get_progress_tracker

pipeline = get_pipeline_state()
progress = get_progress_tracker()
```

## 🚀 Запуск пайплайна

### Полный запуск:
```bash
python main.py
```

### Запуск конкретного этапа:
```bash
# Только этап 1 (сортировка)
python main.py --stage 1

# Только этап 2 (обработка форматов)
python main.py --stage 2

# Только этап 3 (кластеризация)
python main.py --stage 3
```

### Восстановление из последней точки:
```bash
python main.py --resume
```

### Очистка всех данных перед запуском:
```bash
python main.py --clear-all
```

## 📊 Структура этапов

### Этап 1: Сортировка файлов (`stage1_sorting.py`)
- **Вход:** Файлы в `SourceFiles/`
- **Выход:** Отсортированные файлы по форматам в `Sorted/{format}/`
- **Функции:**
  - `_sort_worker()` - обработка одного файла
  - `_run_sort_pool()` - параллельная сортировка
  - `run_stage_1_sort()` - основной метод этапа

### Этап 2: Обработка форматов (`stage2_processing.py`)
- **Вход:** Отсортированные файлы
- **Выход:** Документы с эмбеддингами в БД и `Embeddings/`
- **Подэтапы для каждого формата:**
  - A) Извлечение текста/изображений
  - B) Построение эмбеддингов (текст + vision)
- **Функции:**
  - `_run_extraction_substage()` - извлечение
  - `_run_embeddings_substage()` - эмбеддинги
  - `run_stage_2_process_formats()` - управление подэтапами

### Этап 3: Кластеризация (`stage3_clustering.py`)
- **Вход:** Документы с эмбеддингами
- **Выход:** Кластеры документов
- **Уровни кластеризации:**
  1. ТЕМА (верхнеуровневая)
  2. ПОДТЕМЫ (внутри каждой темы)
  3. УТОЧНЕНИЕ (LLM описание)

## 🧩 Модули состояния (`state/`)

### `pipeline_state.py`
- Сохранение состояния на диск (`pipeline_state.json`)
- Управление статусами этапов
- Поддержка восстановления после сбоя

```python
from state.pipeline_state import get_pipeline_state

pipeline = get_pipeline_state()
pipeline.set_stage_status(1, "completed", {"files": 100})
is_done = pipeline.is_stage_completed(2)
```

### `progress_tracker.py`
- Отслеживание прогресса выполнения
- Логирование каждые 10%
- Тайминг этапов

```python
from state.progress_tracker import get_progress_tracker

progress = get_progress_tracker()
progress.begin_stage(1, "Сортировка", total_files)
progress.update(completed_count)
result = progress.complete_stage()
```

## 🛠 Утилиты (`utils/`)

### `format_groups.py`
- Управление группами форматов для этапа 2
- Валидация подэтапов (извлечение → эмбеддинги)
- Web-интерфейс для выбора следующего действия

```python
from utils.format_groups import get_format_groups_manager, select_substage_to_start

manager = get_format_groups_manager()
manager.prepare_groups(sorted_data)
select_substage_to_start("pdf_text", "extract")  # Запуск из web-интерфейса
```

### `rollback_manager.py`
- Откат операций на этапах 2 и 3
- Возврат файлов в исходное состояние
- Очистка БД и .npy файлов

```python
from utils.rollback_manager import get_rollback_manager

manager = get_rollback_manager()
manager.rollback_embeddings("pdf_text", sorted_data)
manager.rollback_extraction("word_docx", sorted_data)
```

## 📝 Преимущества новой архитектуры

| Критерий | До v5.0 | После v5.0 |
|----------|---------|------------|
| **Размер main.py** | ~2000 строк | ~200 строк |
| **Тестируемость** | Низкая (глобальные переменные) | Высокая (инкапсуляция) |
| **Читаемость** | Средняя | Высокая |
| **Поддержка** | Сложная | Простая |
| **Масштабируемость** | Ограниченная | Хорошая |

## 🔧 Добавление нового этапа

1. Создать файл `stages/stage4_new.py`
2. Добавить функцию `run_stage_4_new(data: List[Dict]) -> List[Dict]`
3. Обновить `main.py`:
```python
from stages.stage4_new import run_stage_4_new

# В функции main():
if args.stage is None or args.stage == 4:
    result = await run_stage_4_new(data)
```

## 📌 Зависимости

- Python 3.8+
- PostgreSQL + pgvector
- PyTorch (для GPU эмбеддингов)
- Transformers (HuggingFace модели)
- NLTK, scikit-learn, hdbscan
- psycopg2-binary

## 📄 Лицензия

Проект разработан для внутреннего использования в File Organizer.
