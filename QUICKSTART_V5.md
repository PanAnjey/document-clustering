# 🚀 Быстрый старт после рефакторинга v5.0

## ✅ Проверка установки

```bash
# 1. Проверить структуру файлов
ls -la stages/ state/ utils/

# 2. Проверить импорты в main.py
python -c "from stages.stage1_sorting import run_stage_1_sort; print('OK')"
python -c "from stages.stage2_processing import run_stage_2_process_formats; print('OK')"
python -c "from stages.stage3_clustering import run_stage_3_cluster_refine; print('OK')"

# 3. Проверить состояние пайплайна
python -c "from state.pipeline_state import get_pipeline_state; ps = get_pipeline_state(); print(ps.get_summary())"
```

## 🎯 Запуск пайплайна

### Полный запуск:
```bash
python main.py
```

### Конкретный этап:
```bash
# Только сортировка (этап 1)
python main.py --stage 1

# Только обработка форматов (этап 2)
python main.py --stage 2

# Только кластеризация (этап 3)
python main.py --stage 3
```

### Восстановление:
```bash
# Продолжить с последней точки
python main.py --resume

# Очистить всё и начать заново
python main.py --clear-all
```

## 📊 Мониторинг прогресса

### Через консоль:
```bash
# Следить за логами
tail -f Logs/file_organizer.log
```

### Через код:
```python
from state.pipeline_state import get_pipeline_state
from state.progress_tracker import get_progress_tracker

pipeline = get_pipeline_state()
progress = get_progress_tracker()

# Статус этапов
print(pipeline.get_summary())

# Текущий прогресс
print(progress.get_current_progress())
```

## 🧪 Тестирование модулей

### Этап 1: Сортировка
```python
from stages.stage1_sorting import _sort_worker, _run_sort_pool
from pathlib import Path

# Тест одного файла
result = _sort_worker(Path("test.pdf"))
print(result)

# Тест пула (если есть файлы)
files = [Path("file1.pdf"), Path("file2.docx")]
results = _run_sort_pool(files, num_workers=4)
print(f"Обработано: {len(results)}")
```

### Этап 2: Обработка форматов
```python
from stages.stage2_processing import (
    run_stage_2_process_formats, 
    select_substage_to_start,
    _find_group
)

# Запуск этапа
sorted_data = [{"source": "test.pdf", "type": "pdf_text"}]
processed = await run_stage_2_process_formats(sorted_data)
print(f"Обработано: {len(processed)}")

# Выбор подэтапа из web-интерфейса
select_substage_to_start("pdf_text", "extract")
```

### Этап 3: Кластеризация
```python
from stages.stage3_clustering import run_stage_3_cluster_refine, _cluster_group
from cluster_engine import ClusterAnalyzer

# Тест кластеризации
data = [
    {"text_embedding": [0.1]*768},
    {"text_embedding": [0.2]*768},
]

analyzer = ClusterAnalyzer(similarity_threshold=0.85)
clustered = await run_stage_3_cluster_refine(data)
print(f"Кластеров: {len(clustered)}")
```

## 🛠 Управление состоянием

### Сохранение статуса этапа:
```python
from state.pipeline_state import get_pipeline_state

pipeline = get_pipeline_state()

# Установить статус
pipeline.set_stage_status(1, "completed", {"files": 100})

# Проверить завершение
if pipeline.is_stage_completed(2):
    print("Этап 2 завершен")
```

### Отслеживание прогресса:
```python
from state.progress_tracker import get_progress_tracker

progress = get_progress_tracker()

# Начало этапа
progress.begin_stage(stage_num=1, label="Сортировка", total_items=100)

# Обновление (вызывается в цикле)
progress.update(completed_count=50)

# Завершение
result = progress.complete_stage()
print(f"Этап завершен: {result}")
```

## 🔄 Откат операций

### Откат эмбеддингов:
```python
from utils.rollback_manager import get_rollback_manager

manager = get_rollback_manager()

# Откат для формата pdf_text
count = manager.rollback_embeddings("pdf_text", sorted_data)
print(f"Откатано файлов: {count}")
```

### Откат извлечения:
```python
# Откат для формата word_docx
count = manager.rollback_extraction("word_docx", sorted_data)
print(f"Возвращено файлов: {count}")
```

## 📝 Добавление нового этапа

1. **Создать файл:** `stages/stage4_new.py`

2. **Добавить функцию:**
```python
# stages/stage4_new.py
async def run_stage_4_new(data: List[Dict]) -> List[Dict]:
    """Этап 4: Новая функциональность."""
    
    # Ваша логика здесь
    
    return result
```

3. **Обновить main.py:**
```python
# В импортах
from stages.stage4_new import run_stage_4_new

# В функции main()
if args.stage is None or args.stage == 4:
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 4: NEW FEATURE")
    logger.info("=" * 60)
    
    result = await run_stage_4_new(data)
    
    if args.stage == 4:
        return 0
```

## 🐛 Решение проблем

### Проблема: `ImportError` при запуске
**Решение:** Проверить, что все модули в `__init__.py`:
```python
# stages/__init__.py
from .stage1_sorting import run_stage_1_sort
from .stage2_processing import run_stage_2_process_formats
from .stage3_clustering import run_stage_3_cluster_refine

__all__ = [
    "run_stage_1_sort",
    "run_stage_2_process_formats",
    "run_stage_3_cluster_refine",
]
```

### Проблема: `KeyError` в pipeline_state.json
**Решение:** Удалить файл и перезапустить:
```bash
rm pipeline_state.json
python main.py --clear-all
```

### Проблема: БД недоступна
**Решение:** Проверить подключение к PostgreSQL:
```sql
-- В psql
\c file_organizer_db
SELECT COUNT(*) FROM documents;
```

## 📚 Документация

- `REFACTORING_V5.md` - подробное описание изменений
- `REFACTORING_SUMMARY.md` - краткое резюме рефакторинга
- `main.py` - комментарии к каждому этапу
- Каждый модуль имеет docstring с описанием

## 🎓 Примеры использования

### Полная последовательность:
```python
import asyncio
from stages.stage1_sorting import run_stage_1_sort
from stages.stage2_processing import run_stage_2_process_formats
from stages.stage3_clustering import run_stage_3_cluster_refine

async def full_pipeline():
    # Этап 1
    sorted_data = await run_stage_1_sort()
    
    # Этап 2
    processed_data = await run_stage_2_process_formats(sorted_data)
    
    # Этап 3
    clustered_data = await run_stage_3_cluster_refine(processed_data)
    
    return clustered_data

# Запуск
result = asyncio.run(full_pipeline())
```

### Параллельный запуск этапов:
```python
import asyncio

async def parallel_stages():
    # Этапы 1 и 2 могут выполняться параллельно (если независимы)
    stage1, stage2 = await asyncio.gather(
        run_stage_1_sort(),
        some_other_task()
    )
    
    return stage1, stage2
```

---

**Версия:** 5.0  
**Дата:** 2026-06-07  
**Статус:** ✅ Готово к использованию
