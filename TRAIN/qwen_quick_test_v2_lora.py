"""qwen_quick_test_v2_lora.py
==========================

Быстрый тест модели Qwen2.5-VL-7B-Instruct с обученным LoRA-адаптером для
классификации ориентации отсканированных документов (0 / 90 / 180 / 270).

ОТЛИЧИЯ ОТ qwen_quick_test.py (v1)
----------------------------------
v1 обращалась к модели Qwen2.5-VL-32B-Instruct через LM Studio HTTP API
(localhost:1234) и использовала сложный промпт prompt.md, ожидая на выходе
развёрнутый JSON с голосованием по 14 критериям.

v2 ЗАГРУЖАЕТ ЛОКАЛЬНУЮ базовую модель Qwen2.5-VL-7B-Instruct в формате
HuggingFace (bf16) + LoRA-адаптер, дообученный в проекте qwen_orient на задачу
классификации угла. Запуск идёт без HTTP — модель держится в VRAM в одном
процессе Python, что даёт ~7× ускорение (4 мин на 599 файлов против ~30 минут
через LM Studio).

ПУТИ К МОДЕЛЯМ (после реорганизации 2026-07-04)
------------------------------------------------
Базовая модель и адаптер лежат в отдельных плоских каталогах на D:, рядом с
другими HF-моделями проекта. Никаких обращений к HF Hub при запуске нет.
  база:    D:\MODELS\Transformers\Qwen2.5-VL-7B-Instruct\        (~15.5 ГБ)
  адаптер: D:\MODELS\Transformers\Qwen2.5-VL-7B-Instruct-Orient-LoRA\  (~165 МБ)

ВАЖНО: LoRA-адаптер обучен отвечать ОДНИМ числом (0/90/180/270), а не развёрнутым
JSON. Поэтому промпт, который передаётся модели, — упрощённый (точно такой же,
как при обучении в train_lora.py: SYSTEM_PROMPT + PROMPT). Полный prompt.md
остаётся доступным по флагу --full-prompt, но в этом режиме модель не использует
обученные веса LoRA — результаты будут как у zero-shot baseline.

АРХИТЕКТУРА LoRA НА ВХОДЕ
-------------------------
- base_model: D:\MODELS\Transformers\Qwen2.5-VL-7B-Instruct\ (bf16, 15.5 ГБ)
- adapter:    D:\MODELS\Transformers\Qwen2.5-VL-7B-Instruct-Orient-LoRA\
              LoRA r=16, alpha=32, на q/k/v/o + gate/up/down всех 28 LLM слоёв
              (~40.4M trainable params / 8.33B total = 0.48%)
- visual encoder (ViT) — ЗАМОРОЖЕН, в градиенты не входит
- всего ~165 МБ адаптера поверх 15.5 ГБ базы

ВАЖНОСТЬ CUDA_VISIBLE_DEVICES=0
-------------------------------
Если в системе 2+ GPU, Trainer из transformers оборачивает модель в
torch.nn.DataParallel. DataParallel реплицирует модель, и в каждой реплике
`self.visual.parameters()` становится пустым — Qwen2_5_VLModel.forward внутри
делает `self.visual.dtype` -> `next(param.dtype for p in self.parameters() ...)` ->
StopIteration. Явное ограничение одним GPU отключает DataParallel.
Эту же переменную нужно выставить ДО импорта torch.
"""

# ============================================================================
# 0. НАСТРОЙКА ОКРУЖЕНИЯ — ДО импорта torch / transformers / peft
# ============================================================================
import os

# КРИТИЧНО: только один GPU виден процессу -> Trainer не оборачивает в DataParallel.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Раньше был нужен HF_HOME, потому что модель лежала в Hub cache. Теперь база
# и адаптер лежат в локальных папках (см. MODEL_ID / DEFAULT_ADAPTER_PATH ниже),
# и from_pretrained() грузится прямо из них. HF_HUB_OFFLINE=1 запрещает любые
# сетевые обращения (если config.json упоминает remote repo, всё равно локально).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# Windows без Developer Mode выдаёт warning про symlinks в кэше — глушим,
# на случай если какие-то утилиты ещё обращаются к cache.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
# Чтобы предупреждение о параллелизме tokenizer'а не спамило вывод.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ============================================================================
# 1. Импорты
# ============================================================================
import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import requests  # оставлен только на случай проверки удалённых endpoint'ов

# Импорт модуля работы с MySQL — тот же, что в v1.
sys.path.insert(0, r"D:\Yandex.Disk\PYTHON\DIADOC\optimized")
import db_mysql_optimized as mysql_db  # noqa: E402  (импорт после sys.path.insert)

# torch и transformers импортируем после выставления env-переменных.
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # noqa: E402
from peft import PeftModel  # noqa: E402

# ============================================================================
# 2. КОНСТАНТЫ И КОНФИГУРАЦИЯ
# ============================================================================

# Базовая модель лежит В ВИДЕ ЛОКАЛЬНОЙ ПАПКИ на D: (раньше использовался
# HF Hub cache в D:\MODELS\hf_cache\hub\models--Qwen--Qwen2.5-VL-7B-Instruct,
# но в итоге содержимое snapshot скопировано в плоский каталог, и обращение
# идёт напрямую по абсолютному пути — без взаимодействия с Hub).
# from_pretrained() принимает как имя репозитория на HF Hub, так и путь к
# локальной папке с config.json + model-*.safetensors.
MODEL_ID = r"D:\MODELS\Transformers\Qwen2.5-VL-7B-Instruct"

# Путь к обученному LoRA-адаптеру. Содержит adapter_config.json +
# adapter_model.safetensors (~157 МБ). Аналогично базе, лежит в локальной папке
# на D: рядом с другими моделями; никаких обращений к Hub не требуется.
DEFAULT_ADAPTER_PATH = Path(r"D:\MODELS\Transformers\Qwen2.5-VL-7B-Instruct-Orient-LoRA")

# Каталог testовой выборки (`distrib/{0,90,180,270}/*.png`) — как в v1.
DISTRIB_DIR = Path(r"D:\FileOrganizer\TRAIN\dataset\distrib")

# Полный промпт prompt.md — для опционального режима zero-shot (--full-prompt).
PROMPT_FILE = Path(r"D:\Yandex.Disk\PYTHON\NLTK\Кластеризация\TRAIN\prompt.md")

# Метки классов; индекс в массиве соответствует числовому значению угла (0/90/180/270).
ANGLE_ORDER = [0, 90, 180, 270]

# Каталог с картинками для обработки по умолчанию (test-выборка).
PDF_IMAGES_DIR = Path(r"D:\FileOrganizer\Extracted\PDF_Images")

# MySQL конфиг (та же база, что в v1 — таблица nltk.test_orient).
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "passwd": "mysql",
    "port": 3308,
    "charset": "utf8mb4",
    "collation": "utf8mb4_general_ci",
}

# Имя модели, под которым она записывается в БД (для отделения от других версий).
MODEL_DISPLAY_NAME = "qwen2.5-vl-7b-instruct-lora"

# Тест-лимит по умолчанию (как в v1). Можно переопределить через CLI.
TEST_LIMIT = 10

# ----- Промпты для LoRA: те же строки, что в train_lora.py -------------------
# Их НЕЛЬЗЯ менять без переобучения LoRA — модель научилась отвечать на этот
# промпт одним числом. Любое отклонение ведёт к деградации точности.

# Системный ролик: задаёт роль "только классификатор".
SYSTEM_PROMPT_LORA = (
    "Ты — модель классификации ориентации отсканированных документов. "
    "Выводи только одно число: 0, 90, 180 или 270."
)

# Пользовательский ролик: завершается `Ориентация:` — это односложный
# completion-trick. Без него Qwen3.x может начать переформулировать задачу.
PROMPT_LORA = (
    "Изображение — отсканированный документ. Определи его ориентацию: 0, 90, 180 или 270 "
    "градусов (по часовой стрелке) относительно вертикального (upright) положения текста. "
    "Ответь только одним числом.\nОриентация:"
)

# Семантика меток (важно для интерпретации результата):
#   0   = документ уже в upright-ориентации, ничего поворачивать не нужно
#   90  = скан повёрнут на 90° CW относительно upright -> для выпрямления крутить CCW на 90
#   180 = скан перевёрнут                          -> крутить на 180
#   270 = скан повёрнут на 270° CW (= 90° CCW)     -> для выпрямления крутить CW на 90
# Эта конвенция была восстановлена из имён файлов qwen_vs_vit_report.json
# (`XXX_rot090.png` -> true_angle=90).

# Максимальная сторона изображения после ресайза. Аналог MAX_SIDE в train_lora.py.
# 768 px -> ~576 vision tokens, влезает в 24 ГБ VRAM вместе с LoRA-весами.
MAX_IMAGE_SIDE = 768
MIN_IMAGE_SIDE = 64


# ============================================================================
# 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================================

def maybe_resize(img: Image.Image) -> Image.Image:
    """Ограничивает размер изображения, как в train_lora.py.

    Qwen2.5-VL разбивает картинку на патчи 14×14; число vision-токенов пропор-
    ционально H*W / 14 / 14 / (spatial_merge_size**2). Огромные сканы (>1500 px)
    выдают 5000+ токенов и переполняют VRAM. Мы ограничиваем сторону 768 px.
    """
    w, h = img.size
    scale = 1.0
    if max(w, h) > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / max(w, h)
    # Не даём короткой стороне упасть ниже MIN_SIDE — иначе image_grid_thw станет
    # нулевым и processor упадёт.
    if min(w, h) * scale < MIN_IMAGE_SIDE:
        scale = MIN_IMAGE_SIDE / min(w, h) if min(w, h) > 0 else 1.0
    if abs(scale - 1.0) > 1e-3:
        nw = max(1, int(round(w * scale)))
        nh = max(1, int(round(h * scale)))
        return img.resize((nw, nh))
    return img


def load_prompt_full() -> str:
    """Загружает полный промпт из prompt.md (используется только с --full-prompt).

    Внимание: полный промпт ожидает развёрнутый JSON-ответ. LoRA-адаптер НЕ
    обучался под этот формат — он обучен отвечать одним числом. Поэтому в паре
    с --full-prompt нужно включать ещё и --baseline (без LoRA).
    """
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        return f.read()


def build_ground_truth() -> dict[str, int]:
    """Сканирует distrib/{0,90,180,270}/ и строит {file_name: true_angle}.

    Истина берётся ИЗ ИМЕНИ ПОДКАТАЛОГА. То есть предполагаем, что разметка в
    каталоге dataset корректна — это полностью автоматический source-of-truth,
    без ручной разметки. Можно доверять, потому что файлы туда попадали точно
    по углу перевёрнутости (или по суффиксу _rotNNN у аугментированных версий).
    """
    gt = {}
    for angle in ANGLE_ORDER:
        d = DISTRIB_DIR / str(angle)
        if d.exists():
            for f in d.iterdir():
                if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                    gt[f.name] = angle
    return gt


# ============================================================================
# 4. КЛАСС МОДЕЛИ — инкапсулирует загрузку и инференс
# ============================================================================

class OrientationModel:
    """Загружает один раз и переиспользует:
        - процессор (токенизатор + image preprocessor)
        - базовую VLM Qwen2.5-VL-7B-Instruct в bf16 на указанном GPU
        - (опционально) LoRA-адаптер поверх неё

    Жизненный цикл: load() -> [predict() ... N раз] -> (free при выходе).
    """

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.processor = None
        self.model = None
        self.loaded = False

    def load(self, adapter_path: Path | None, use_lora: bool = True):
        """Загрузка процессора и модели. Вызывается ОДИН раз в main().

        Параметры
        ---------
        adapter_path : Path or None
            Каталог с adapter_config.json + adapter_model.safetensors.
            None — использовать дефолтный путь DEFAULT_ADAPTER_PATH.
        use_lora : bool
            True — наложить обученный LoRA-адаптер поверх базы (для задачи
                   классификации ориентации).
            False — оставить только базовую модель (zero-shot; используется
                   либо с --baseline, либо вместе с --full-prompt).
        """
        print(f"[model] загрузка процессора из: {MODEL_ID}")
        # from_pretrained() умеет принимать абсолютный путь к каталогу с
        # tokenizer.json + preprocessor_config.json. Никакой сетевой обращений
        # к Hub нет — значит можно выставить HF_HUB_OFFLINE=1 для надёжности.
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)

        print(f"[model] загрузка базовой модели из {MODEL_ID} на {self.device} (bf16, sdpa)")
        # device_map=self.device — простой способ разместить всё на одной GPU
        # (модель помещается в 24 ГБ; device_map="auto" бы потребовал map в
        # разные устройства при >1 GPU, что привело бы к DataParallel).
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID,                          # абсолютный путь к локальному каталогу базы
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",   # sdpa работает везде (flash_attn не обязателен)
            device_map=self.device,        # грузим на указанный GPU напрямую
        )

        # use_cache=True на инференсе — с KV-кэшем модель быстрее генерирует
        # последовательность токенов. На обучении стояло False, т.к. KV-кэш
        # несовместим с gradient checkpointing.
        self.model.config.use_cache = True
        self.model.eval()

        if use_lora:
            if adapter_path is None:
                adapter_path = DEFAULT_ADAPTER_PATH
            if not adapter_path.exists():
                raise FileNotFoundError(
                    f"LoRA-адаптер не найден: {adapter_path}. "
                    "Запустите train_lora.py или передайте --adapter_path."
                )
            print(f"[model] наложение LoRA-адаптера: {adapter_path}")
            # PeftModel.from_pretrained загружает adapter_model.safetensors и
            # подставляет LoRA-слои в нужные модули (q/k/v/o + gate/up/down
            # в каждом из 28 LLM слоёв). Базовые веса остаются неизменными.
            self.model = PeftModel.from_pretrained(self.model, str(adapter_path))
            print("[model] LoRA активна ( trainable%: ~0.48 от всех параметров )")
        else:
            print("[model] LoRA ОТКЛЮЧЕНА (zero-shot базовая модель)")

        self.loaded = True

    # ----- Низкоуровневая генерация ----------------------------------------

    def _build_messages(self, prompt_text: str, system_text: str, img: Image.Image) -> list[dict]:
        """Сборка chat-сообщений в формате Qwen2.5-VL.

        Из процессора в apply_chat_template они превратятся в строку вида:
            <|im_start|>system ...
            <|im_start|>user
            <vision_start><|image_pad|> x N <vision_end>
            текст-промпта
            <|im_start|>assistant
        Где N =число vision-токенов = (H/14 * W/14) / (spatial_merge_size**2).
        """
        return [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt_text},
            ]},
        ]

    def _generate_text(self, rendered_text: str, img: Image.Image, temperature: float = 0.1) -> str:
        """Один прямой проход model.generate(...).

        Возвращает сгенерированный текст (только NEW токены, без input_ids).
        Используется `do_sample=(temperature>0)`, поэтому temperature=0 (или очень
        малое значение) делает вывод практически детерминированным.
        У LoRA-модели есть preferred answer — короткое число — поэтому max_new_tokens=8
        достаточно. Для развёрнутого JSON (full-prompt) нужен запас — 2048.
        """
        # processor(...): токенизация текста + подготовка pixel_values/image_grid_thw.
        # padding=False — мы гоняем по одной картинке за раз.
        inputs = self.processor(
            text=[rendered_text],
            images=[img],
            return_tensors="pt",
            padding=False,
        ).to(self.device)

        max_new_tokens = 8 if temperature < 0.05 else 2048

        do_sample = temperature > 0.0
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9   # консервативный nucleus sampling

        # autocast — на всякий случай, чтобы случайно не остаться в fp32.
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            out = self.model.generate(**inputs, **gen_kwargs)

        # out[0] имеет форму [1, T_total]; отрезаем input_ids (T_input) и декодируем
        # только "новую" часть.
        input_len = inputs["input_ids"].shape[-1]
        new_tokens = out[0][input_len:]
        return self.processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # ----- Высокоуровневый predict ------------------------------------------

    def predict(self, image_path: Path, use_full_prompt: bool, full_prompt_text: str) -> dict:
        """Классифицирует одну картинку.

        Возвращает словарь:
          raw_response :str   — что буквально сгенерировала модель
          parsed_json   :dict — если распарсился JSON (full-prompt mode), иначе None
          pred_angle    :int  — финальная оценка угла 0/90/180/270 или -1 если не вышло
          confidence    :float — NULL- Placeholder для совместимости БД. LoRA-режим
                                 confidence не возвращает, поэтому ставим 1.0 только
                                 если удалось распарсить число.
        """
        img = Image.open(image_path).convert("RGB")
        img = maybe_resize(img)

        if use_full_prompt:
            # Режим полного промпта (zero-shot, без LoRA) — модель ожидается
            # выдать развёрнутый JSON.
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": full_prompt_text},
                ]}
            ]
            rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            raw_text = self._generate_text(rendered, img, temperature=0.1)
            parsed_json = parse_json_response(raw_text)
            if parsed_json is not None:
                angle = parsed.get("orientation", -1)
                # Нормализация углов: -90 -> 270, 360+ -> mod 360
                if angle not in (-1, 0, 90, 180, 270) and angle is not None:
                    norm = (angle % 360 + 360) % 360
                    angle = norm if norm in (0, 90, 180, 270) else angle
                confidence = float(parsed.get("confidence", 0.0))
                return {"raw_response": raw_text, "parsed_json": parsed_json,
                        "pred_angle": angle, "confidence": confidence}
            # JSON не распарсился — попытаемся извлечь число как fallback.
            angle, conf = parse_simple_angle(raw_text)
            return {"raw_response": raw_text, "parsed_json": None,
                    "pred_angle": angle, "confidence": conf}

        # Режим LoRA — простой промпт,.model отвечает одним числом.
        messages = self._build_messages(PROMPT_LORA, SYSTEM_PROMPT_LORA, img)
        rendered = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # temperature=0.0 — детерминированный argmax; LoRA обучена конкретному
        # ответу, sampling нафиг не нужен для классификации.
        raw_text = self._generate_text(rendered, img, temperature=0.0)
        angle, conf = parse_simple_angle(raw_text)
        return {"raw_response": raw_text, "parsed_json": None,
                "pred_angle": angle, "confidence": conf}


# ============================================================================
# 5. ПАРСИНГ ОТВЕТА
# ============================================================================

# Регулярка для извлечения первого числа из текста. Используется в LoRA-режиме,
# где модель обычно выдаёт либо "90", либо "90<|im_end|>" (im_end удаляется при
# skip_special_tokens=True в decode, но иногда попадает в raw_response).
NUM_RE = re.compile(r"\b(0|90|180|270)\b")
VALID_ANGLES = {0, 90, 180, 270}


def parse_simple_angle(text: str) -> tuple[int, float]:
    """Извлекает одно число 0/90/180/270 из сгенерированного текста.

    Возвращает (angle, confidence):
        angle       — 0/90/180/270 или -1 если не распознано
        confidence — 1.0 если это одно из валидных чисел, 0.0 иначе.
                     Заметим: настоящая уверенность модели не вычисляется на
                     инференсе. Здесь 1.0/0.0 — placeholder для схемы БД.
    """
    if not text:
        return -1, 0.0
    text = text.strip()
    # Прямое соответствие — часто модель выдаёт ровно "90".
    if text in {"0", "90", "180", "270"}:
        return int(text), 1.0
    # Ищем целое число 0/90/180/270 как отдельное слово.
    m = NUM_RE.search(text)
    if m:
        return int(m.group(1)), 1.0
    return -1, 0.0


def parse_json_response(content: str) -> dict | None:
    """Парсит JSON-ответ (используется только в --full-prompt режиме).

    Пытается извлечь JSON из:
      1) ```json ... ... ``` блока
      2) произвольного {...} в тексте
      3) всего содержимого как JSON
    Возвращает dict или None.
    """
    if not content:
        return None
    json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    json_match = re.search(r"\{.*?\}", content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return None


# ============================================================================
# 6. РАБОТА С БД
# ============================================================================

def init_db(model_name: str):
    """Подключается к MySQL и удаляет старые записи данной модели из test_orient.

    Так гарантируется идемпотентность запусков: каждый прогон записывает свежий
    результат без дублей.
    """
    mysql_db.init_db(DB_CONFIG)
    try:
        mysql_db.exec_val("DELETE FROM nltk.test_orient WHERE model_name = %s", (model_name,))
        mysql_db.mydb.commit()
        print(f"[DB] удалены старые записи для модели {model_name}\n")
    except Exception as e:
        print(f"[WARN] ошибка удаления старых записей: {e}")


def save_to_db(model_name: str, fname: str, true_angle: int, pred_angle: int,
               raw_response: str, parsed: dict | None):
    """Вставляет одну строку в nltk.test_orient.

    Схема таблицы (см. SHOW CREATE TABLE):
        - все поля кроме id — NULLable
        - votes_*    DEFAULT 0
        - applied_criteria DEFAULT 0
        - key_evidence, contradictions — longtext с CHECK json_valid

    Поэтому:
        - Для LoRA-режима: пишем только pred_angle + raw_response, остальные поля
          NULL/дефолт. JSON-колонки НЕ трогаем (NULL платит CHECK constraint).
        - Для full-prompt: заполняем полностью как в v1.
    """
    try:
        if parsed is not None:
            votes = parsed.get("votes", {})
            evidence = json.dumps(parsed.get("key_evidence", []), ensure_ascii=False)
            contradictions = json.dumps(parsed.get("contradictions", []), ensure_ascii=False)
            mysql_db.exec_val(
                """INSERT INTO nltk.test_orient
                   (model_name, file_name, true_angle, pred_angle,
                    orientation, correction_angle, correction_direction, confidence,
                    votes_0, votes_90, votes_180, votes_270,
                    applied_criteria, key_evidence, contradictions, raw_response)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (
                    model_name, fname, true_angle, pred_angle,
                    parsed.get("orientation", -1),
                    parsed.get("correction_angle", 0),
                    parsed.get("correction_direction", "none"),
                    float(parsed.get("confidence", 0.0)),
                    int(votes.get("0", 0)),
                    int(votes.get("90", 0)),
                    int(votes.get("180", 0)),
                    int(votes.get("270", 0)),
                    int(parsed.get("applied_criteria", 0)),
                    evidence if evidence != "[]" else None,
                    contradictions if contradictions != "[]" else None,
                    raw_response[:32000] if raw_response else None,
                ),
            )
        else:
            # LoRA-режим — только базовые поля. Остальные останутся NULL/0 по DEFAULT.
            mysql_db.exec_val(
                """INSERT INTO nltk.test_orient
                   (model_name, file_name, true_angle, pred_angle, raw_response)
                   VALUES (%s,%s,%s,%s,%s)""",
                (model_name, fname, true_angle, pred_angle,
                 raw_response[:32000] if raw_response else None),
            )
    except Exception as e:
        print(f"[DB] ошибка сохранения {fname}: {e}")


# ============================================================================
# 7. MAIN
# ============================================================================

def main():
    # ----- Разбор аргументов CLI --------------------------------------------
    ap = argparse.ArgumentParser(
        description="Тест LoRA-модели Qwen2.5-VL-7B-Instruct на задаче ориентации.")
    ap.add_argument("--test_limit", type=int, default=TEST_LIMIT,
                    help=f"Лимит файлов (по умолчанию {TEST_LIMIT}). 0 = все файлы.")
    ap.add_argument("--device", type=str, default="cuda:0",
                    help="Устройство (по умолчанию cuda:0). Должно совпадать с CUDA_VISIBLE_DEVICES=0.")
    ap.add_argument("--adapter_path", type=str, default=str(DEFAULT_ADAPTER_PATH),
                    help="Путь к каталогу LoRA-адаптера.")
    ap.add_argument("--baseline", action="store_true",
                    help="Не использовать LoRA (zero-shot Qwen2.5-VL-7B).")
    ap.add_argument("--full_prompt", action="store_true",
                    help="Использовать полный prompt.md (требует --baseline).")
    ap.add_argument("--no_db", action="store_true",
                    help="Не подключаться к MySQL (только печать в stdout).")
    ap.add_argument("--images_dir", type=str, default=str(PDF_IMAGES_DIR),
                    help="Каталог с тестовыми PNG для прохождения.")
    ap.add_argument("--report", type=str,
                    default=r"D:\FileOrganizer\TRAIN\models\qwen_lora_test.json",
                    help="Куда писать JSON-отчёт.")
    args = ap.parse_args()

    # --full_prompt имеет смысл только вместе с --baseline: LoRA-модель обучена
    # под упрощённый промпт и не будет выдавать валидный JSON по полному промпту.
    if args.full_prompt and not args.baseline:
        print("[WARN] --full_prompt обычно идёт вместе с --baseline. LoRA обучена "
              "отвечать одним числом, не развёрнутым JSON. Продолжаем, но "
              "качество предсказания JSON-ответа НЕ гарантируется.")

    # ----- Загрузка полного промпта (если нужно) ---------------------------
    full_prompt_text = ""
    if args.full_prompt:
        full_prompt_text = load_prompt_full()
        print(f"[prompt] загружен {PROMPT_FILE.name}: {len(full_prompt_text)} символов")

    # ----- Каталог тестовых файлов -----------------------------------------
    images_dir = Path(args.images_dir)
    all_files = sorted([f for f in images_dir.iterdir()
                        if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg")])
    if args.test_limit and args.test_limit > 0:
        files = all_files[:args.test_limit]
    else:
        files = all_files
    print(f"[scan] {images_dir}: всего {len(all_files)} файлов, обрабатываем {len(files)}")

    # ----- Ground truth по distrib/{0,90,180,270} ---------------------------
    ground_truth = build_ground_truth()
    print(f"[gt] ground truth: {len(ground_truth)} файлов из {DISTRIB_DIR}")

    # ----- Загрузка модели ------------------------------------------------
    m = OrientationModel(device=args.device)
    m.load(
        adapter_path=Path(args.adapter_path) if args.adapter_path else None,
        use_lora=not args.baseline,
    )

    # ----- БД (опционально) ------------------------------------------------
    if not args.no_db:
        init_db(MODEL_DISPLAY_NAME)

    # ----- Главный цикл ---------------------------------------------------
    results = []
    n_json = 0
    t0 = time.time()

    print(f"\n=== Тест {MODEL_DISPLAY_NAME} | LoRA={'on' if not args.baseline else 'off'} "
          f"| full_prompt={'on' if args.full_prompt else 'off'} | {len(files)} files ===\n")

    for i, fpath in enumerate(files, 1):
        fname = fpath.name

        # Берём только файлы, для которых есть gt (иначе нечем оценить accuracy).
        if fname not in ground_truth:
            continue
        true_angle = ground_truth[fname]

        # Инференс одной картинки.
        try:
            pred = m.predict(fpath, use_full_prompt=args.full_prompt, full_prompt_text=full_prompt_text)
        except Exception as e:
            print(f"[err] {fname}: {repr(e)[:200]}")
            pred = {"raw_response": f"<error: {repr(e)[:100]}>", "parsed_json": None,
                    "pred_angle": -1, "confidence": 0.0}

        pred_angle = pred["pred_angle"]
        correct = (pred_angle == true_angle)
        results.append({
            "file": fname,
            "true_angle": true_angle,
            "qwen_pred": pred_angle,
            "qwen_conf": pred["confidence"],
            "qwen_correct": correct,
            "is_json": pred["parsed_json"] is not None,
        })
        if pred["parsed_json"] is not None:
            n_json += 1

        # Печать одной строки прогресса.
        status = "OK " if correct else "FAIL"
        print(f"{i:3d}/{len(files)} | true={true_angle:3d} pred={pred_angle:3d} | {status} "
              f"raw={pred['raw_response'][:40]!r}")

        # Сохранение в БД построчно (сразу, чтобы не терять результат при крэше).
        if not args.no_db:
            save_to_db(MODEL_DISPLAY_NAME, fname, true_angle, pred_angle,
                       pred["raw_response"], pred["parsed_json"])

    elapsed = time.time() - t0
    total = len(results)
    if total == 0:
        print("\n[WARN] нет совпадений с ground truth. Завершаем.")
        if not args.no_db:
            mysql_db.close()
        return

    # ----- Подсчёт метрик ------------------------------------------------
    # no_pred — случаи, когда модель не выдала валидный угол (pred=-1).
    no_pred = [r for r in results if r["qwen_pred"] == -1]
    classified = [r for r in results if r["qwen_pred"] != -1]
    correct = sum(1 for r in classified if r["qwen_correct"])
    accuracy = correct / len(classified) if classified else 0.0

    print(f"\n=== ИТОГИ ({total} файлов, {elapsed:.1f}s) ===\n")
    print(f"всего файлов:             {total}")
    print(f"JSON-ответов (full mode): {n_json}/{total} ({n_json/total:.2%})" if total else "—")
    print(f"нет предсказания (=-1):   {len(no_pred)}")
    print(f"классифицировано:         {len(classified)}")
    print(f"правильных / классифиц.:  {correct}/{len(classified)} = {accuracy:.2%}")

    print("\n=== Per-class метрики ===\n")
    for angle in ANGLE_ORDER:
        angle_results = [r for r in classified if r["true_angle"] == angle]
        angle_no_pred = [r for r in no_pred if r["true_angle"] == angle]
        if angle_results or angle_no_pred:
            ok = sum(1 for r in angle_results if r["qwen_correct"])
            n_total = len(angle_results) + len(angle_no_pred)
            print(f"  {angle:3d}deg: {ok}/{n_total} = {ok / n_total:.2%}  (no_pred: {len(angle_no_pred)})")

    print("\n=== Confusion matrix ===")
    cm = defaultdict(lambda: defaultdict(int))
    for r in results:
        cm[r["true_angle"]][r["qwen_pred"]] += 1
    all_preds = [-1] + ANGLE_ORDER
    header = "         pred: " + "  ".join(f"{a:4d}" for a in all_preds)
    print(header)
    for true_a in ANGLE_ORDER:
        row = [cm[true_a][p] for p in all_preds]
        print(f"  true {true_a:3d}  [{row[0]:4d} {row[1]:4d} {row[2]:4d} {row[3]:4d} {row[4]:4d}]")

    # ----- Сохранение отчёта ----------------------------------------------
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_name": MODEL_DISPLAY_NAME,
            "lora_used": not args.baseline,
            "full_prompt_used": args.full_prompt,
            "adapter_path": args.adapter_path,
            "total_files": total,
            "json_responses": n_json,
            "qwen_accuracy": round(accuracy, 4),
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[report] {report_path}")

    if not args.no_db:
        mysql_db.close()
        print("[DB] соединение закрыто")


if __name__ == "__main__":
    main()