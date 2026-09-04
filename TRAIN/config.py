"""Общий конфигурационный файл для скриптов Vision Transformer (определение ориентации сканов).

Используется модулями: prepare_dataset.py, train.py, test_inference.py, review_app.py.
Оптимизировано для ViT Large с attention-механизмом для решения проблемы 0° vs 180°.

Версия: ViT Large (307M параметров) вместо Base (86M параметров).
"""
import os
from pathlib import Path

# ---- Исходный датасет (подготовленный пользователем) ----
DISTRIB_DIR: Path = Path(r"D:\FileOrganizer\TRAIN_CLASS")

# ---- Разделённые наборы (создаются prepare_dataset.py) ----
TRAIN_DIR: Path = Path(r"D:\FileOrganizer\TRAIN\dataset\train")
VAL_DIR: Path = Path(r"D:\FileOrganizer\TRAIN\dataset\val")

# ---- Доля валидационной выборки ----
VAL_SPLIT: float = 0.2
RANDOM_SEED: int = 42

# ---- Модель Vision Transformer (LARGE версия) ----
MODELS_DIR: Path = Path(r"D:\FileOrganizer\TRAIN\models")
MODEL_NAME: str = "vit_large_patch16_384"   # timm: vit_large_patch16_384 (для A4 сканов)
OUTPUT_CLASSES: int = 4                    # 0°, 90°, 180°, 270° (4 класса)
CLASS_NAMES = [0, 90, 180, 270]             # индекс -> угол ориентации
MODEL_FILENAME: str = "orientation_vit_large_384.pth"
BEST_MODEL_FILENAME: str = "orientation_vit_large_384_best.pth"

# ---- Параметры обучения (для ViT Large) ----
IMAGE_SIZE: int = 512                      # ↑ увеличено с 384 до 512 для лучших деталей текста
BATCH_SIZE: int = 16                       # оставлено 16 из-за увеличения IMAGE_SIZE (больше VRAM)
EPOCHS: int = 100                          # ↑ увеличено с 70 до 100
LEARNING_RATE: float = 3e-5                # ↓ уменьшено с 5e-5 до 3e-5
WEIGHT_DECAY: float = 0.05
NUM_WORKERS: int = 8
LR_SCHEDULER: str = "cosine"
LR_MIN: float = 1e-7                       # ↓ уменьшено с 1e-6 до 1e-7
BEST_METRIC: str = "balanced"

# ---- Дисбаланс классов (особенно важно для 0° vs 180°) ----
CLASS_WEIGHTS: str = "effective_num"       # эффективное число образцов (Cui-2019)
USE_AMP: bool = True                       # mixed precision для ускорения
EARLY_STOP_PATIENCE: int = 10              # ↑ увеличено с 7 до 10

# ---- Вход модели: кроп текстовой области в высоком разрешении ----
INPUT_MODE: str = "full"                   # "full" | "crop" (как в MobileNetV3)
CROP_ADAPTIVE: bool = True
CROP_TARGET_GLYPH: float = 14.0
CROP_WINDOW: int = 512                     # ↑ увеличено с 384 до 512 (соответствует IMAGE_SIZE)
CROP_SEARCH_MAXSIDE: int = 800
CROP_JITTER: int = 64
CROP_FALLBACK_FULL: bool = True
CROP_TEXT_MIN_COMPONENTS: int = 40

# ---- Resize / сохранение пропорций (для INPUT_MODE="full") ----
RESIZE_MODE: str = "letterbox"             # letterbox | squash
LETTERBOX_PAD_TRAIN: str = "random"        # random | white | black | gray

# ---- Аугментация ориентации (критично для 0° vs 180°) ----
ROTATION_AUG: bool = True                  # поворот на 0/90/180/270 с пересчётом метки
ROTATION_AUG_VAL_FULL: bool = True         # валидация во всех 4 ориентациях

# ---- Аугментация перекоса сканирования (skew) ----
SKEW_AUG: bool = True
SKEW_MAX_DEG: float = 4.0                  # ± максимальный угол перекоса
SKEW_PROB: float = 0.5                     # вероятность применения

# ---- Аугментация чёрных полей по краям листа ----
BORDER_AUG: bool = True
BORDER_PROB: float = 0.3
BORDER_MAX_FRAC: float = 0.06

# ---- Горизонтальный флип (для документов обычно выключен) ----
HFLIP_AUG: bool = False

# ---- Attention mechanism для улучшения 0° vs 180° ----
USE_ATTENTION: bool = True                 # добавить attention-механизм к модели
ATTENTION_DROPOUT: float = 0.3             # dropout в attention блоке
ATTENTION_HIDDEN_SIZE: int = 512           # размер скрытого слоя attention

# ---- Тестовый прогон (test_inference.py) ----
INFERENCE_INPUT_DIR: Path = Path(r"D:\FileOrganizer\Extracted\PDF_Images")
CORRECTED_DIR: Path = Path(r"D:\FileOrganizer\TRAIN\corrected")
NOT_CORRECTED_DIR: Path = Path(r"D:\FileOrganizer\TRAIN\not_corrected")
LOW_CONFIDENCE_DIR: Path = Path(r"D:\FileOrganizer\TRAIN\low_confidence")  # отдельная директория для неуверенных
INFERENCE_RESULTS_FILE: Path = MODELS_DIR / "inference_results.json"
USE_BEST_MODEL_FOR_INFERENCE: bool = True
INFERENCE_DEVICE: str = "auto"             # auto | cuda | cpu

# ---- Производительность инференса (3_test_inference.py) ----
INFERENCE_BATCH_SIZE: int = 32             # размер батча классификации на GPU (уменьшен для Large)
INFERENCE_WORKERS: int = max(1, (os.cpu_count() or 8) - 2)
INFERENCE_CONFIDENCE_THRESHOLD: float = 0.85
INFERENCE_ABSTAIN: bool = True             # abstain для low confidence

# ---- Постобработка после коррекции ориентации (deskew + crop) ----
DESKEW_ENABLED: bool = True
DESKEW_MAX_ANGLE: float = 15.0
DESKEW_ANGLE_STEP: float = 0.1
DESKEW_SEARCH_MAXSIDE: int = 1000

CROP_BORDERS_ENABLED: bool = True
CROP_BORDERS_THRESHOLD: int = 20
CROP_BORDERS_MARGIN: int = 4
CROP_BORDERS_MIN_AREA_RATIO: float = 0.25

# ---- Отчёт проверки / дообучение (review_app.py) ----
RETRAIN_DIR: Path = Path(r"D:\FileOrganizer\TRAIN\retrain")
RETRAIN_CLASSES = CLASS_NAMES
REVIEW_REPORT_FILE: Path = RETRAIN_DIR / "review_report.csv"

# ---- Повторное дообучение (5_retrain.py) ----
RETRAIN_BASE_MODEL: str = "best"
RETRAIN_VAL_SPLIT: float = 0.15
RETRAIN_EPOCHS: int = 30                   # ViT требует больше эпох при retrain
RETRAIN_LEARNING_RATE: float = 2e-5        # меньше LR для fine-tuning
RETRAIN_BATCH_SIZE: int = 16               # уменьшаем батч для retrain (Large)
RETRAIN_WEIGHT_DECAY: float = 0.05
RETRAIN_EARLY_STOP_PATIENCE: int = 7       # 0 = отключить
# On-the-fly ротация (0/90/180/270 с пересчётом метки) ОТДЕЛЬНО для дообучения.
RETRAIN_ROTATION_AUG: bool = True
# Имя итогового файла: создаётся N-я версия, чтобы не затирать старые модели
RETRAIN_FILE_PREFIX: str = "orientation_vit_large_retrain"
# Префикс для поиска последней retrain модели в 3_test_inference_retrain.py
RETRAIN_MODEL_PREFIX: str = "orientation_vit_large_retrain"  # Для тестирования дообученной модели
RETRAIN_REPLACE_BEST: bool = True

# ---- Поддерживаемые расширения изображений ----
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".gif")

# ---- Размер превью в приложении проверки (px) ----
PREVIEW_MAX_WIDTH: int = 520
PREVIEW_MAX_HEIGHT: int = 680

# ---- GPU конфигурация для обучения (2x RTX PRO 4000 48GB) ----
NUM_GPUS: int = 2                          # количество доступных GPU
DISTRIBUTED_TRAINING: bool = False         # использовать DataParallel/DistributedDataParallel (отключено для стабильности)
GRADIENT_ACCUMULATION_STEPS: int = 1       # накопление градиентов для увеличения effective batch size

# ---- ViT-specific параметры ----
USE_VIT_LAYERS: str = "large"              # **LARGE** версия (307M параметров вместо 86M)
PATCH_SIZE: int = 16                       # размер патча для ViT (стандартно 16)
NUM_PATCHES: int = (IMAGE_SIZE // PATCH_SIZE) ** 2

# ---- Экспериментальные параметры для улучшения 0° vs 180° ----
VERTICAL_FLIP_AUG: bool = True             # вертикальная инверсия (критично для 0° vs 180°)
VERTICAL_FLIP_PROB: float = 0.9            # ↑ увеличено с 0.7 до 0.9
SYMMETRY_WEIGHT: float = 3.0               # вес для путаницы 0°↔180° (не используется при FOCAL_GAMMA=0)
FOCAL_GAMMA: float = 0.0                   # gamma=0 отключает Focal Loss (используется CrossEntropyLoss)
BOTTOM_MARKER_AUG: bool = False            # отключено (вносит шум)
BOTTOM_MARKER_PROB: float = 0.3            # вероятность применения bottom marker

# ---- Аугментация: тонкая рамка сверху (помогает выучить "верх" документа) ----
TOP_BORDER_AUG: bool = False               # ✗ отключено (ухудшает результат, модель переобучается на рамку)
TOP_BORDER_PROB: float = 0.3               # вероятность применения
TOP_BORDER_THICKNESS: float = 0.005        # толщина рамки (доля от высоты изображения)
TOP_BORDER_COLOR: tuple = (100, 100, 100)  # цвет рамки (серый)
