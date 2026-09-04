# Сводка экспериментов по определению ориентации документов

## Дата: 2026-07-01
## Тестовый набор: 2991 файл

---

## Итоговая таблица результатов

| Метод | Accuracy | Ошибок | Время | Рекомендация |
|-------|----------|--------|-------|--------------|
| **Ensemble ViT+PaddleOCR** | **99.80%** | **6** | 234s | ⭐ **Production** |
| Ensemble ViT+EffNet+Tesseract | 99.80% | 6 | 349s | Альтернатива |
| ViT-Large 384 | 99.63% | 11 | 33.5s | Быстрый вариант |
| EfficientNet-B1 | 99.57% | 13 | 15.8s | Edge/embedded |
| PaddleOCR | 99.00% | 30 | 237s | Компонент ансамбля |
| Tesseract OSD | 96.12% | 116 | ~300s | Не рекомендуется |
| SigLIP2 (обученная) | 70.81% | 873 | 588s | Не рекомендуется |
| Qwen3.5-35B (llama.cpp) | 56.87% | 781 | 2000s+ | ❌ Не работает |
| SigLIP2 (zero-shot) | 20.33% | 2383 | 399s | Не работает |

---

## Ключевые выводы

### 1. Лучший метод: Ensemble ViT + PaddleOCR
- **99.80% accuracy** (6 ошибок из 2991)
- На 33,830 документах: ~68 ошибок (0.20%)
- Время: ~44 минуты на полный набор
- 98.76% файлов — согласие обеих моделей

### 2. ViT-Large 384 — лучшая одиночная модель
- **99.63% accuracy** (11 ошибок из 2991)
- Быстрый инференс: 33.5s
- Компактный размер: 1.2 GB

### 3. LLM модели не подходят для этой задачи
- **Qwen3.5-35B**: 56.87% (предсказывает только 0°)
- **SigLIP2 zero-shot**: 20.33% (хуже случайного угадывания)
- **SigLIP2 обученная**: 70.81% (значительно хуже ViT)
- Причина: не специализированы для определения ориентации

### 4. Основная проблема: симметричные документы
- Путаница 0° ↔ 180° — основной источник ошибок
- Решается через ансамблирование разных методов

---

## Per-class сравнение лучших методов

| Класс | ViT-Large | EfficientNet | PaddleOCR | Ensemble |
|-------|-----------|--------------|-----------|----------|
| 0° | 99.71% | 99.54% | 99.25% | **99.83%** |
| 90° | 99.30% | 99.82% | 98.77% | **99.65%** |
| 180° | 99.43% | 98.87% | 98.87% | **99.72%** |
| 270° | **100.00%** | **100.00%** | 98.22% | **100.00%** |

---

## Рекомендации

### Для production (максимальная точность)
**Ensemble ViT + PaddleOCR**
- Accuracy: 99.80%
- Время: 234s на 2991 файл
- На 33,830 документах: ~68 ошибок

### Для быстрого инференса
**Только ViT-Large**
- Accuracy: 99.63%
- Время: 33.5s на 2991 файл
- На 33,830 документах: ~125 ошибок

### Для edge/embedded устройств
**EfficientNet-B1**
- Accuracy: 99.57%
- Время: 15.8s на 2991 файл
- Размер: 26 MB

---

## Файлы отчётов

| Отчёт | Путь |
|-------|------|
| **Сводка** | `REPORT_SUMMARY.md` |
| **ViT-Large и EfficientNet** | `REPORT_VIT_EFFICIENTNET.md` |
| **PaddleOCR** | `REPORT_PADDLEOCR.md` |
| **SigLIP2** | `REPORT_SIGLIP.md` |
| **Qwen3.5-35B** | `REPORT_QWEN.md` |
| **Ансамбли** | `REPORT_ENSEMBLES.md` |
| **Сравнение всех методов** | `REPORT_ORIENTATION_COMPARISON.md` |

---

## Скрипты

| Скрипт | Назначение |
|--------|------------|
| `1_prepare_dataset.py` | Подготовка датасета |
| `2_train.py` | Обучение ViT-Large |
| `2_train_siglip.py` | Обучение SigLIP2 |
| `3_test_inference.py` | Инференс ViT-Large |
| `paddle_vs_vit.py` | Сравнение PaddleOCR с ViT |
| `ensemble_vit_paddle.py` | Ансамбль ViT + PaddleOCR |
| `ensemble_orientation.py` | Ансамбль ViT + EffNet + Tesseract |
| `siglip_orientation.py` | SigLIP2 zero-shot |
| `siglip_trained_vs_vit.py` | Сравнение обученной SigLIP2 с ViT |
| `qwen_server_orientation.py` | Qwen3.5-35B через llama-server |

---

## Модели

| Модель | Путь | Размер |
|--------|------|--------|
| ViT-Large | `D:\FileOrganizer\TRAIN\models\orientation_vit_large_384_best.pth` | 1.2 GB |
| EfficientNet-B1 | `D:\FileOrganizer\TRAIN\models_efficientnet\orientation_efficientnet_b1_best.pth` | 26 MB |
| SigLIP2 | `D:\FileOrganizer\TRAIN\models\orientation_siglip2_best.pth` | 4.5 GB |
| SigLIP2 base | `D:\MODELS\Transformers\siglip2-so400m-patch14-384` | 4.5 GB |
| Qwen3.5-35B | `D:\MODELS\STUDIO\HauhauCS\Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive\` | 21 GB |
