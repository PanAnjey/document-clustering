# engines/oc_qwen3vl.py
# Движок Qwen3-VL-8B-Instruct (Qwen/Qwen3-VL-8B-Instruct).
# GPU (bf16, ~16 GB VRAM). Вывод — markdown/html (срезается в
# oc_textnorm.normalize при подсчёте метрик).
# Требуется transformers >= 4.57.

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import re

import torch
from PIL import Image

_FENCE_RE = re.compile(r'```(?:html|markdown|text)?\n?')  # обёртка ```html ... ```

MODEL_PATH = r"D:\MODELS\Transformers\Qwen3-VL-8B-Instruct"
DEVICE = 'cuda:0'
MAX_NEW_TOKENS = 4096
# Как у nanonets: рендеры 300 dpi (~2480x3508) дают слишком много
# визуальных токенов → OOM. Даунскейл до 1600px по длинной стороне.
MAX_SIDE = 1600

# ВАЖНО: короткий промпт в стиле nanonets («extract text, tables in html»)
# заставляет Qwen3-VL извлекать ТОЛЬКО таблицу, пропуская шапку и текст
# документа (проверено на docs 161468/49811, F1 0.36/0.77 → prompt v2).
PROMPT = (
    "Transcribe the entire document from the image into text, preserving "
    "the natural reading order. Include ALL elements: letterheads, "
    "organization names, dates, reference numbers, addresses, body text, "
    "lists, footers, signatures. Do not skip any text. Format tables in "
    "HTML. Output only the document text, without comments."
)

_instance = None


class Qwen3VLEngine:
    name = 'qwen3vl'

    def __init__(self, device: str = DEVICE, model_path: str = MODEL_PATH,
                 name: str = None, gen_overrides: dict = None):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.device = device
        self.model_path = model_path
        self.gen_overrides = gen_overrides or {}
        if name:
            self.name = name
        if device.startswith('cuda'):
            _ = torch.tensor([0], device=device)
            torch.cuda.synchronize(int(device.split(':')[1]))
        self.proc = AutoProcessor.from_pretrained(self.model_path)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path, torch_dtype=torch.bfloat16).to(device).eval()

    def ocr(self, png_path: str) -> str:
        img = Image.open(png_path).convert('RGB')
        if max(img.size) > MAX_SIDE:
            img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        messages = [{'role': 'user', 'content': [
            {'type': 'image'},
            {'type': 'text', 'text': PROMPT},
        ]}]
        text = self.proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self.proc(text=[text], images=[img],
                           return_tensors='pt', padding=True).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS,
                                      do_sample=False, **self.gen_overrides)
        trimmed = out[:, inputs['input_ids'].shape[1]:]
        text = self.proc.batch_decode(
            trimmed, skip_special_tokens=True)[0].strip()
        return _FENCE_RE.sub('', text).strip()


def get_engine():
    global _instance
    if _instance is None:
        _instance = Qwen3VLEngine()
    return _instance
