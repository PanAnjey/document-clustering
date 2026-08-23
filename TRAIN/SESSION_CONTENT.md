# Сессия: Дообучение VLM для классификации ориентации сканов документов

**Дата:** 4-6 июля 2026
**Задача:** Дообучение модели Qwen2.5-VL-7B-Instruct (позже — Qwen3-VL-8B-Instruct) для определения ориентации отсканированных документов (0°, 90°, 180°, 270°)

---

## Исходные данные

| Параметр | Значение |
|----------|----------|
| GGUF модель | `D:\MODELS\STUDIO\lmstudio-community\Qwen2.5-VL-7B-Instruct-GGUF\Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` |
| llama.cpp | `D:\llama.cpp` (с `llama-qwen2vl-cli.exe`) |
| Датасет | `D:\FileOrganizer\TRAIN\dataset\{distrib,val}\{0,90,180,270}\*.png` |
| Train (distrib) | 801/329/113/122 файлов по классам (дисбаланс) |
| Val | 346/114/71/68 файлов |
| GPU | 2× RTX PRO 4000 Blackwell, 24 GB каждая |
| Python venv | `D:\VENV\LLM\Scripts\python.exe` (Python 3.12, torch 2.11+cu130, transformers 5.4.0) |
| Предыдущий опыт | `D:\FileOrganizer\TRAIN\models\` — обученные ViT/MobileNet/SigLIP2/EfficientNet модели ориентации |

---

## Ключевое ограничение GGUF

**GGUF Q4_K_M нельзя дообучать напрямую** — это 4-битное квантование, несовместимое с backprop. llama.cpp `finetune` поддерживает только text-only LoRA, не multimodal/VL. Поэтому потребовался переход на HF-формат (bf16).

---

## Этап 1: Дообучение Qwen2.5-VL-7B-Instruct + LoRA

### 1.1 Скачивание базовой модели

- **Способ:** Прямой HTTPS-downloader (`download_https.py`) с HTTP Range resume, обход Xet protocol (который stall'ился на chunk-store RPC)
- **Результат:** 14 файлов, 15.46 ГБ → `D:\MODELS\hf_cache\hub\models--Qwen--Qwen2.5-VL-7B-Instruct\` (позже перенесено в `D:\MODELS\Transformers\Qwen2.5-VL-7B-Instruct\`)
- **Время:** ~27 мин при ~11 МБ/с

### 1.2 Подготовка датасета

**Скрипт:** `build_dataset.py`

- Семантика меток восстановлена из `qwen_vs_vit_report.json`: `XXX_rot090.png → true_angle=90` → метка = угол наклона скана CW относительно upright
- Аугментация: каждое изображение класса 0 поворачивается на 90/180/270 CW, создавая 3 augmented примера для классов 90/180/270 (на лету в `__getitem__`, не хранятся на диске)
- **Результат:**
  - `train.jsonl`: 3768 записей (классы: 0=801, 90=1130, 180=914, 270=923)
  - `val.jsonl`: 599 записей (без аугментации)

### 1.3 Обучение LoRA

**Скрипт:** `train_lora.py`

**Архитектура LoRA:**
- Vision encoder (ViT) — ЗАМОРОЖЕН, `requires_grad=False`
- LoRA на LLM q/k/v/o + gate/up/down всех 28 слоёв (exact-match target module paths, не bare suffixes — visual encoder содержит `mlp.gate_proj`/`up_proj`/`down_proj`)
- r=16, alpha=32, dropout=0.05
- 40.4M trainable / 8.33B total = 0.48%

**Label masking:** Ручная маскировка по `<|im_start|>assistant ... <|im_end|>` (chat template Qwen2.5-VL не содержит `{% generation %}` директивы → `return_assistant_tokens_mask` возвращает нули)

**Критический баг — DataParallel + StopIteration:**
- Trainer при `torch.cuda.device_count() > 1` оборачивает модель в `torch.nn.DataParallel`
- DataParallel делает scatter подмодулей → в не-primary реплике `self.visual.parameters()` пуст → `self.visual.dtype` → `next(param.dtype for ...)` → `StopIteration`
- **Решение:** `os.environ["CUDA_VISIBLE_DEVICES"] = "0"` ДО импорта torch

**Настройки обучения (финальные):**
- 8 эпох, bs=2, grad_accum=8 (effective batch=16), lr=1e-4, cosine scheduler
- Early Stopping patience=8, `load_best_model_at_end=True` (metric=eval_loss)
- gradient_checkpointing + bf16 на cuda:0
- MAX_SIDE=768, MIN_SIDE=64 (pre-resize в `__getitem__`)

**Результат обучения Qwen2.5-VL+LoRA:**
- Завершилось на 6.357 эпохах (early stopping)
- train_loss=0.015, eval_loss=0.00095
- Время: ~9 часов
- Adapter: 157 МБ → `D:\MODELS\Transformers\Qwen2.5-VL-7B-Instruct-Orient-LoRA\`

### 1.4 Eval Qwen2.5-VL+LoRA

**Скрипт:** `eval_lora.py`

| Класс | Точность | Ошибки |
|-------|----------|--------|
| 0     | 346/346 = 100.0% | 0 |
| 90    | 114/114 = 100.0% | 0 |
| **180** | **58/71 = 81.7%** | **13** |
| 270   | 67/68 = 98.5%   | 1 |
| **Overall** | **97.66%** | **14** |
| **Macro** | **95.05%** | — |

**Baseline zero-shot Qwen2.5-VL (без LoRA):** 81.88% — прирост **+15.78 п.п.**

**Проблема:** Class 180 — 13 ошибок (18.3%). Причина: `MAX_SIDE=768` теряет мелкие детали текста, а различение 180° от 0° часто требует чтения мелкого текста (подписи, даты, нумерация страниц).

---

## Этап 2: Создание qwen_quick_test_v2_lora.py

**Файл:** `D:\Yandex.Disk\PYTHON\NLTK\Кластеризация\TRAIN\qwen_quick_test_v2_lora.py`

Локальный инференс (без LM Studio HTTP) — модель в VRAM, ~0.4 с/файл (7× быстрее HTTP). Подробные комментарии на русском (~340 строк). CLI: `--baseline`, `--full_prompt`, `--no_db`, `--test_limit`, `--adapter_path`.

---

## Этап 3: Реорганизация моделей

| Модель | Путь | Размер |
|--------|------|--------|
| Базовая Qwen2.5-VL-7B-Instruct | `D:\MODELS\Transformers\Qwen2.5-VL-7B-Instruct\` | 15.46 ГБ |
| LoRA адаптер Orient | `D:\MODELS\Transformers\Qwen2.5-VL-7B-Instruct-Orient-LoRA\` | 165 МБ |

Старый HF cache (`D:\MODELS\hf_cache\hub\models--Qwen--Qwen2.5-VL-7B-Instruct\`) удалён для освобождения места.

`qwen_quick_test_v2_lora.py` скорректирован: `MODEL_ID` → абсолютный путь, `HF_HUB_OFFLINE=1`.

Smoke-test после переноса: 10/10 корректно, 4.1 с.

---

## Этап 4: Переход на Qwen3-VL-8B-Instruct

### 4.1 Обоснование

Пользователь указал, что в **Qwen3-VL-8B-Instruct** используется **SigLIP-2** в vision encoder, а в проекте уже были успешные эксперименты с SigLIP-2 (`orientation_siglip2_best.pth`). Дополнительные преимущества:
- Interleaved-MRoPE (новая схема позиционирования)
- DeepStack (fuses multi-level ViT features)
- Upgraded OCR (32 языка, включая русский)
- +1B параметров над Qwen2.5-VL-7B

### 4.2 Скачивание Qwen3-VL-8B-Instruct

- 16 файлов, 16.34 ГБ → `D:\MODELS\Transformers\Qwen3-VL-8B-Instruct\`
- Через `download_qwen3vl.py` (прямый HTTPS с resume)
- Время: ~27 мин

### 4.3 Адаптация train_lora для Qwen3-VL

**Скрипт:** `train_lora_qwen3vl.py`

**Отличия от Qwen2.5-VL версии:**
1. Класс модели: `Qwen3VLForConditionalGeneration` (вместо `Qwen2_5_VLForConditionalGeneration`)
2. **LoRA target_modules = bare `["q_proj", ...]`** — SigLIP-2 visual encoder не содержит `q_proj`/`k_proj` (у него `qkv` — одиночный Linear), так что bare PEFT endswith mathing не затрагивает visual. Probe подтвердил: visual_count=0 для всех target suffixes.
3. **Новое поле `mm_token_type_ids`** — Qwen3-VL требует его (ValueError без него). Processor возвращает его в batch; добавлено в dataset `__getitem__` и collator.
4. Visual encoder: 36 LLM слоёв (против 28 в Qwen2.5-VL), imёнa target_modules те же
5. `freeze_vision` находит `model.visual` (а не `model.model.visual`)

### 4.4 Исправление ресайза

Диагностика показала:
- Qwen3-VL: `patch_size=16`, `merge_size=2` → effective patch 32×32 px (vs 28 в Qwen2.5-VL)
- `MAX_SIDE=768` был слишком агрессивным — терял текстовые детали
- Тест трёх размеров: 768 (быстро, но потеря деталей), 1280 (медленно — 50+ ч), **1024** (компромисс ~13 ч)
- `preprocessor_config.json`: `longest_edge=16777216` (до 4096×4096), processor сам не сильно ужимает

### 4.5 Обучение Qwen3-VL+LoRA (on-the-fly, MAX_SIDE=1024)

**Настройки:** 8 эпох, bs=2/accum=8, lr=1e-4, early_stopping=6, load_best_model_at_end

- Завершилось на 5.934 эпохах (early stopping сработал)
- train_loss=0.0160, eval_loss=0.0042
- Время: ~11.6 часов
- Adapter: 170 МБ → `D:\FileOrganizer\TRAIN\qwen_orient\lora_qwen3vl_output\final\`

### 4.6 Eval Qwen3-VL+LoRA — РЕКОРДНЫЙ РЕЗУЛЬТАТ

**Скрипт:** `eval_lora_qwen3vl.py` (val.jsonl, `do_resize=True` — консистентно с train)

| Класс | Qwen2.5-VL+LoRA (768px) | Qwen3-VL+LoRA (1024px) | Дельта |
|-------|--------------------------|-------------------------|--------|
| 0     | 100.0%                   | 99.7% (345/346)         | -0.3%  |
| 90    | 100.0%                   | **100.0%** (114/114)    | ±0     |
| **180** | **81.7%** (58/71)        | **100.0%** (71/71)      | **+18.3%** |
| 270   | 98.5%                    | **100.0%** (68/68)      | +1.5%  |
| **Overall** | **97.66%**               | **99.83%**              | **+2.17 п.п.** |
| **Macro** | 95.05%                   | 99.93%                  | +4.88 п.п. |
| **Ошибок** | **14**                   | **1**                   | -13    |

**Гипотеза пользователя подтвердилась:** SigLIP-2 в vision encoder Qwen3-VL полностью решил проблему класса 180 (было 13 ошибок → стало 0). Это было главное слабое место Qwen2.5-VL+LoRA.

---

## Этап 5: Pre-prep датасет (LANCZOS, подготовлен, не запущен)

### 5.1 Обоснование

Выявлена проблема двойного ресайза: мой `maybe_resize(BICUBIC, MAX_SIDE=1024)` → потом `smart_resize(BICUBIC)` внутри processor. Две BICUBIC операции подряд теряют sharp edges. LANCZOS сохраняет мелкие текстуры лучше BICUBIC.

### 5.2 prep_dataset.py

- LANCZOS resize (vs BICUBIC по умолчанию в image_processor)
- Целевой longest side = 1760 px (A4@150dpi проходит почти 1:1)
- Все размеры factor-32 aligned (`patch_size=16 × merge_size=2`)
- Поворот-аугментация выполнена на диске (2403 файлов `__aug_rot{90,180,270}.png`)
- 4367 файлов, 47 уникальных размеров, **0 файлов с нарушением factor-alignment**

### 5.3 train_lora_qwen3vl_prep.py

- `processor.image_processor.do_resize = False` — processor берёт PNG AS-IS, не делает BICUBIC поверх LANCZOS
- `pre_resized=True` в jsonl → пропускает `maybe_resize`
- `rotation_applied=0` для pre-resized (поворот уже на диске)
- Поле `mm_token_type_ids` прокинуто

### 5.4 Статус

Pre-prep датасет и скрипты **готовы к запуску**, но **не запускались** — Qwen3-VL+LoRA (on-the-fly 1024) уже дал 99.83%, проблема class 180 решена. Pre-prep запуск отложен до анализа единственной ошибки.

---

## Сводка результатов

| Модель | Accuracy | Class 0 | Class 90 | Class 180 | Class 270 | Ошибок | Время обучения |
|--------|----------|---------|----------|-----------|-----------|--------|----------------|
| Zero-shot Qwen2.5-VL (baseline) | 81.88% | — | — | — | — | ~543 | 0 |
| Qwen2.5-VL + LoRA (768px) | 97.66% | 100.0% | 100.0% | **81.7%** | 98.5% | 14 | ~9 ч |
| **Qwen3-VL + LoRA (1024px)** | **99.83%** | 99.7% | **100.0%** | **100.0%** | **100.0%** | **1** | ~11.6 ч |

---

## Файлы проекта

### Скрипты (D:\FileOrganizer\TRAIN\qwen_orient\)

| Файл | Назначение |
|------|------------|
| `build_dataset.py` | Подготовка train/val jsonl с on-the-fly rotation augmentation |
| `train_lora.py` | Обучение Qwen2.5-VL + LoRA (on-the-fly, MAX_SIDE=768) |
| `eval_lora.py` | Eval Qwen2.5-VL + LoRA |
| `train_lora_qwen3vl.py` | Обучение Qwen3-VL + LoRA (on-the-fly, MAX_SIDE=1024) |
| `eval_lora_qwen3vl.py` | Eval Qwen3-VL + LoRA |
| `prep_dataset.py` | Pre-prep датасета (LANCZOS, factor-32 aligned, на диск) |
| `train_lora_qwen3vl_prep.py` | Обучение Qwen3-VL + LoRA на pre-prep датасете (do_resize=False) |
| `download_https.py` | Скачивание Qwen2.5-VL-7B-Instruct через HTTPS |
| `download_qwen3vl.py` | Скачивание Qwen3-VL-8B-Instruct через HTTPS |
| `probe_qwen3vl.py` | Проверка API Qwen3-VL (assistant_masks, LoRA targets, visual modules) |

### Модели (D:\MODELS\Transformers\)

| Каталог | Содержимое | Размер |
|---------|-----------|--------|
| `Qwen2.5-VL-7B-Instruct\` | Базовая модель bf16 (5 safetensors shards) | 15.46 ГБ |
| `Qwen2.5-VL-7B-Instruct-Orient-LoRA\` | LoRA адаптер (adapter_model.safetensors) | 165 МБ |
| `Qwen3-VL-8B-Instruct\` | Базовая модель bf16 (4 safetensors shards) | 16.34 ГБ |

### LoRA адаптеры (D:\FileOrganizer\TRAIN\qwen_orient\)

| Каталог | Модель | Accuracy |
|---------|--------|----------|
| `lora_output\final\` | Qwen2.5-VL+LoRA (768px) | 97.66% |
| `lora_qwen3vl_output\final\` | Qwen3-VL+LoRA (1024px) | **99.83%** |

### Датасеты

| Файл | Записей | Описание |
|------|---------|----------|
| `train.jsonl` | 3768 | On-the-fly rotation augmentation |
| `val.jsonl` | 599 | Без аугментации |
| `train_prep.jsonl` | 3768 | Pre-resized (LANCZOS, factor-32 aligned) |
| `val_prep.jsonl` | 599 | Pre-resized |
| `dataset_prep\` | 4367 PNG файлов | Pre-resized изображения на диске |

### Прочее

| Файл | Назначение |
|------|------------|
| `D:\Yandex.Disk\PYTHON\NLTK\Кластеризация\TRAIN\qwen_quick_test_v2_lora.py` | Локальный инференс + запись в MySQL (Qwen2.5-VL+LoRA) |
| `D:\FileOrganizer\TRAIN\models\qwen_vs_vit_report.json` | Источник семантики меток (суффикс `_rotNNN` → `true_angle=NNN`) |

---

## Технические находки

### 1. DataParallel + StopIteration
Trainer при 2+ GPU оборачивает Qwen2_5_VL/Qwen3VL в `torch.nn.DataParallel`. В репликах `self.visual.parameters()` пуст → `self.visual.dtype` → `StopIteration`. Решение: `CUDA_VISIBLE_DEVICES=0` до импорта torch. Альтернативы: DDP через `accelerate launch --multi_gpu`.

### 2. Chat template без `{% generation %}`
Qwen2.5-VL и Qwen3-VL chat templates не содержат `{% generation %}` директивы → `return_assistant_tokens_mask` возвращает все нули. Решение: ручная маскировка labels по `<|im_start|>assistant ... <|im_end|>` (walk input_ids, mark assistant content tokens).

### 3. LoRA target_modules и visual encoder
В Qwen2.5-VL visual encoder имеет `mlp.gate_proj`/`up_proj`/`down_proj` — bare `"gate_proj"` в target_modules задевает и visual. Решение: enumerate по 28 слоям `model.language_model.layers.{i}.*`. В Qwen3-VL visual (SigLIP-2) использует `qkv` (один Linear), не `q_proj` — bare target_modules безопасны.

### 4. Двойной ресайз
Мой `maybe_resize(BICUBIC, MAX_SIDE)` → потом `smart_resize(BICUBIC)` внутри processor = две BICUBIC операции подряд. Потеря sharp edges на подписях/штампах. Решение (prep): LANCZOS один раз на диск + `processor.image_processor.do_resize = False`.

### 5. mm_token_type_ids (новое в Qwen3-VL)
Qwen3-VL требует `mm_token_type_ids` для Multimodal RoPE. Без него → `ValueError: Multimodal data was passed... but mm_token_type_ids is missing`. Processor возвращает его в batch; нужно прокинуть через dataset `__getitem__` и collator.

### 6. Ресайз image processor
- Qwen2.5-VL: `patch_size=14`, `merge_size=2`, factor=28, BICUBIC (resample=3)
- Qwen3-VL: `patch_size=16`, `merge_size=2`, factor=32, BICUBIC (resample=3)
- `smart_resize()` округляет dims кратно factor, aspect ratio preserved, bounds: `min_pixels`..`max_pixels`

---

## Незавершённые задачи

1. **Pre-prep обучение LANCZOS** — скрипты готовы (`train_lora_qwen3vl_prep.py`, `train_prep.jsonl`, `val_prep.jsonl`), но не запускались. Может дать marginal improvement (~+0.1-0.3%) или ускорить throughput.
2. **Анализ 1 ошибки** Qwen3-VL+LoRA — найти неверный пример в `eval_qwen3vl_v1_report.json`, визуально оценить, можно ли его исправить.
3. **Адаптация `qwen_quick_test_v2_lora.py` для Qwen3-VL** — обновить MODEL_ID и adapter_path для использования Qwen3-VL+LoRA в продакшене.
4. **Перенос обученного Qwen3-VL LoRA адаптера** в `D:\MODELS\Transformers\Qwen3-VL-8B-Instruct-Orient-LoRA\` (аналогично Qwen2.5-VL).
5. **Опционально: tiling** для крупных сканов >2500px — обоснован только если анализ ошибок покажет, что FNs приходятся на крупные сканы.