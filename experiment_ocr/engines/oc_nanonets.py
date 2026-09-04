# engines/oc_nanonets.py
# Движок Nanonets-OCR2-3B (nanonets/Nanonets-OCR2-3B, Qwen2.5-VL-3B).
# GPU (bf16, ~6.5 GB VRAM). Вывод — markdown/html (срезается в
# oc_textnorm.normalize при подсчёте метрик).

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from PIL import Image

MODEL_PATH = r"D:\MODELS\Transformers\nanonets_Nanonets-OCR2-3B"
DEVICE = 'cuda:0'
MAX_NEW_TOKENS = 4096
# Рендеры 300 dpi (~2480x3508) дают слишком много визуальных токенов → OOM.
# Даунскейл до 1600px по длинной стороне (типичный размер обучения VLM-OCR).
MAX_SIDE = 1600

PROMPT = (
    "Extract the text from the above document as if you were reading it "
    "naturally. Return the tables in html format. Return the equations in "
    "LaTeX representation. If there is an image in the document and the "
    "image caption is not present, add a small description of the image "
    "inside the <img></img> tag; otherwise, add the image caption inside "
    "<img></img>."
)

_instance = None


class NanonetsEngine:
    name = 'nanonets'

    def __init__(self, device: str = DEVICE):
        from transformers import AutoModelForVision2Seq, AutoProcessor
        self.device = device
        if device.startswith('cuda'):
            _ = torch.tensor([0], device=device)
            torch.cuda.synchronize(int(device.split(':')[1]))
        self.proc = AutoProcessor.from_pretrained(MODEL_PATH)
        self.model = AutoModelForVision2Seq.from_pretrained(
            MODEL_PATH, torch_dtype=torch.bfloat16).to(device).eval()

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
        return self.proc.batch_decode(
            trimmed, skip_special_tokens=True)[0].strip()


def get_engine():
    global _instance
    if _instance is None:
        _instance = NanonetsEngine()
    return _instance
