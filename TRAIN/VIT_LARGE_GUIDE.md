# 🚀 Переход на Vision Transformer Large (307M параметров)

## 📊 Сравнение: Base vs Large

| Параметр | ViT Base | ViT Large | Изменение |
|----------|----------|-----------|-----------|
| **Параметры** | 86M | **307M** | +259% |
| **VRAM (обучение)** | ~15-20GB | **~25-30GB** | +10GB |
| **VRAM (инференс)** | ~10GB | **~15GB** | +5GB |
| **Время обучения** | ~4 часа | **~6 часов** | +2 часа |
| **Точность** | 97-98% | **98-99%** | +1-2% |
| **Batch size** | 32 | **16** | -50% |

---

## ✅ Обновлённые файлы

### 1. config.py - ключевые изменения

```python
# Модель Vision Transformer (LARGE версия)
MODEL_NAME: str = "vit_large_patch16_384"   # timm: vit_large_patch16_384
OUTPUT_CLASSES: int = 4                    # 0°, 90°, 180°, 270° (4 класса)
CLASS_NAMES = [0, 90, 180, 270]             # индекс -> угол ориентации

# Файлы моделей
MODEL_FILENAME: str = "orientation_vit_large_384.pth"
BEST_MODEL_FILENAME: str = "orientation_vit_large_384_best.pth"

# Параметры обучения (для ViT Large)
IMAGE_SIZE: int = 384                      # ViT Large с 384x384 для A4 сканов
BATCH_SIZE: int = 16                       # Уменьшен из-за большего размера модели
EPOCHS: int = 50                           # ViT требует больше эпох для обучения

# ViT-specific параметры
USE_VIT_LAYERS: str = "large"              # **LARGE** версия (307M параметров)
```

### 2. 2_train.py - изменения в build_model()

```python
def build_model(num_classes, device, num_gpus):
    """Строит модель Vision Transformer Large с attention-механизмом."""
    
    # Определяем размер входа из MODEL_NAME
    if "384" in C.MODEL_NAME:
        input_size = 384
    elif "224" in C.MODEL_NAME:
        input_size = 224
    else:
        input_size = 224
    
    # Загружаем предобученную ViT модель через timm (LARGE версия)
    if C.USE_VIT_LAYERS == "large":
        model = timm.create_model(f'vit_large_patch16_{input_size}', pretrained=True, num_classes=0)
        in_features = 1024  # vit_large имеет 1024 features
    else:
        model = timm.create_model(f'vit_base_patch16_{input_size}', pretrained=True, num_classes=0)
        in_features = 768  # vit_base имеет 768 features
    
    # ... (далее attention и классификатор)
```

### 3. 5_retrain.py - обновлённый префикс модели

```python
RETRAIN_FILE_PREFIX: str = "orientation_vit_large_retrain"
RETRAIN_MODEL_PREFIX: str = "orientation_vit_large_retrain"
```

---

## 🚀 Пошаговый план обучения с нуля

### Шаг 1: Подготовка датасета (если ещё не сделано)

```bash
cd D:\Yandex.Disk\PYTHON\NLTK\Кластеризация\TRAIN
python 1_prepare_dataset.py
```

**Ожидаемый вывод:**
```
Классы: ['0', '90', '180', '270']
Готово. train=XXXX val=YYYY
```

### Шаг 2: Проверка структуры датасета

```bash
dir D:\FileOrganizer\TRAIN\dataset\distrib\0
dir D:\FileOrganizer\TRAIN\dataset\distrib\90
dir D:\FileOrganizer\TRAIN\dataset\distrib\180
dir D:\FileOrganizer\TRAIN\dataset\distrib\270
```

**Цель:** Минимум 50-100 файлов на каждый класс (всего ≥200)

### Шаг 3: Обучение с нуля (ViT Large)

```bash
python 2_train.py
```

**Ожидаемое время:** ~6 часов на GPU (2x NVIDIA RTX PRO 4000)

**Ожидаемый вывод:**
```
[INFO] Доступно 2 GPU: NVIDIA RTX PRO 4000 (48GB) x2
Устройство: cuda (GPU count: 2)
[INFO] Загрузка модели с размером входа 384x384
Классы ImageFolder: ['0', '90', '180', '270'] (samples: XXXX train / YYYY val)
Вход: full/letterbox  |  ROTATION_AUG=True ...
Epoch  1/50  ...
```

### Шаг 4: Тестирование на новых данных

```bash
python 3_test_inference.py
```

**Результат:** `D:\FileOrganizer\TRAIN\models\inference_results.json`

---

## 📊 Ожидаемые результаты после обучения с нуля

| Метрика | ViT Base (ранее) | ViT Large (теперь) | Изменение |
|---------|------------------|--------------------|-----------|
| **Общая точность** | 97-98% | **98-99%** | +1-2% ✅ |
| **0° vs 180%** | ~96% | **~97-98%** | +1-2% ✅ |
| **Средняя уверенность** | ~90% | **~93-95%** | +3-5% ✅ |
| **Время обучения** | 4 часа | **6 часов** | +2 часа ⚠️ |
| **VRAM (обучение)** | ~15GB | **~25GB** | +10GB ⚠️ |

---

## 💡 Преимущества ViT Large

### ✅ Точность
- **+1-2% общая точность** на реальных документах
- **Лучшее распознавание мелкого текста** (<10pt)
- **Улучшенное различение 0° vs 180°** (заголовки, подписи видны чётче)

### ✅ Обобщающая способность
- Больше параметров → лучше обобщает на новые документы
- Меньше риск переобучения при большом датасете (>500 файлов)

### ✅ Качество для A4 сканов
- 384×384 + Large архитектура = максимальное качество
- Сохраняет детали текста даже при низком разрешении (150 DPI)

---

## ⚠️ Компромиссы ViT Large

| Параметр | Base | Large | Комментарий |
|----------|------|-------|-------------|
| **VRAM** | ~15GB | **~25GB** | Требует больше памяти |
| **Время обучения** | 4 часа | **6 часов** | +50% времени |
| **Batch size** | 32 | **16** | Меньше батч из-за памяти |
| **Инференс** | 0.15 сек | **0.25 сек** | +0.1 сек на изображение |

---

## 🎯 Критерии успеха обучения с нуля

После обучения модель должна показать:

| Метрика | Целевое значение | Как проверить | Статус |
|---------|------------------|---------------|--------|
| **Общая точность** | ≥98% | `3_test_inference.py` | ⬜ |
| **0° vs 180%** | ≥97% | `analyze_retrain_results.py` | ⬜ |
| **Средняя уверенность** | ≥93% | `inference_results.json` | ⬜ |

---

## 🔄 Полный цикл обучения с нуля

```bash
# 1. Подготовка датасета
cd D:\Yandex.Disk\PYTHON\NLTK\Кластеризация\TRAIN
python 1_prepare_dataset.py

# 2. Проверка структуры (минимум 50 файлов на класс)
dir D:\FileOrganizer\TRAIN\dataset\distrib\0
dir D:\FileOrganizer\TRAIN\dataset\distrib\90
dir D:\FileOrganizer\TRAIN\dataset\distrib\180
dir D:\FileOrganizer\TRAIN\dataset\distrib\270

# 3. Обучение с нуля (ViT Large)
python 2_train.py --epochs 50

# 4. Тестирование на новых данных
python 3_test_inference.py

# 5. Сравнение результатов
type D:\FileOrganizer\TRAIN\models\inference_results.json | more

# 6. (Опционально) Fine-tuning на ошибках
python 5_retrain_v2.py --epochs 30
```

---

## 📝 Чек-лист для успешного обучения:

```yaml
[ ] 1. Убедиться, что в config.py USE_VIT_LAYERS = "large"
[ ] 2. Проверить наличие минимум 50 файлов на каждый класс (всего ≥200)
[ ] 3. Запустить обучение с нуля: python 2_train.py --epochs 50
[ ] 4. Мониторить метрики в консоли (val_acc должен расти)
[ ] 5. Проверить историю обучения: type D:\FileOrganizer\TRAIN\models\train_history.json
[ ] 6. Протестировать на новых данных: python 3_test_inference.py
[ ] 7. Сравнить с предыдущей моделью (Base vs Large)
```

---

## 🎉 Итог

**ViT Large** является оптимальным выбором для вашего случая (**2x NVIDIA RTX PRO 4000 48GB VRAM**):

✅ **+1-2% точность** на реальных документах  
✅ **Лучшее распознавание мелкого текста** (<10pt)  
✅ **Улучшенное различение 0° vs 180°** (заголовки, подписи видны чётче)  
✅ **Адаптивность к качеству сканов** (работает даже на низком разрешении)  

Компромиссы:
⚠️ +2 часа времени обучения  
⚠️ +10GB VRAM  
⚠️ +0.1 сек на инференс  

Для вашего случая с **достаточными ресурсами GPU** - ViT Large является **оптимальным выбором**! 🚀
