# engines/oc_gemma4.py
# Движок google/gemma-4-E4B-it (мультимодальная, ~7.6B dense).
# GPU (bf16, ~16 GB VRAM). Вывод — markdown/plain (срезается в
# oc_textnorm.normalize при подсчёте метрик).
# Требуется transformers >= 5.x (venv_gemma), gemma4 появилась в 5.5.

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import re

import torch
from PIL import Image

_FENCE_RE = re.compile(r'```(?:html|markdown|text)?\n?')

MODEL_PATH = r"D:\MODELS\Transformers\google_gemma-4-E4B-it"
DEVICE = 'cuda:0'
MAX_NEW_TOKENS = 4096
# как у остальных VLM-движков пилота
MAX_SIDE = 1600

# тот же промпт v2, что у qwen3vl (короткий → извлекается только таблица)
PROMPT = (
    "Transcribe the entire document from the image into text, preserving "
    "the natural reading order. Include ALL elements: letterheads, "
    "organization names, dates, reference numbers, addresses, body text, "
    "lists, footers, signatures. Do not skip any text. Format tables in "
    "HTML. Output only the document text, without comments."
)

_instance = None


class Gemma4Engine:
    name = 'gemma4'

    def __init__(self, device: str = DEVICE):
        from transformers import AutoProcessor, Gemma4ForConditionalGeneration
        self.device = device
        if device.startswith('cuda'):
            _ = torch.tensor([0], device=device)
            torch.cuda.synchronize(int(device.split(':')[1]))
        self.proc = AutoProcessor.from_pretrained(MODEL_PATH)
        self.model = Gemma4ForConditionalGeneration.from_pretrained(
            MODEL_PATH, dtype=torch.bfloat16).to(device).eval()

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
                                      do_sample=False)
        trimmed = out[:, inputs['input_ids'].shape[1]:]
        text = self.proc.batch_decode(
            trimmed, skip_special_tokens=True)[0].strip()
        return _FENCE_RE.sub('', text).strip()


def get_engine(device: str = DEVICE):
    global _instance
    if _instance is None:
        _instance = Gemma4Engine(device=device)
    return _instance
