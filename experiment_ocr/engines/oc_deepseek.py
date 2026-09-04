# engines/oc_deepseek.py
# Движок DeepSeek-OCR (deepseek-ai/DeepSeek-OCR, 3B MoE VLM).
# GPU (bf16, ~7 GB VRAM). Вывод — markdown (теги/разметка срезаются
# в oc_textnorm.normalize при подсчёте метрик).
#
# prompt "Free OCR." — режим свободного OCR (текст документа).

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from pathlib import Path

MODEL_PATH = r"D:\MODELS\Transformers\deepseek-ai_DeepSeek-OCR"
PROMPT = "<image>\nFree OCR."
DEVICE = 'cuda:0'
# infer() безусловно делает os.makedirs(output_path) — нужен реальный каталог
# (save_results=False, файлы не пишутся, но каталог требуется).
OUT_DIR = Path(r"C:\Windows\Temp\opencode\deepseek_out")

_instance = None


class DeepSeekEngine:
    name = 'deepseek'

    def __init__(self, device: str = DEVICE):
        from transformers import AutoModel, AutoTokenizer
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        self.device = device
        if device.startswith('cuda'):
            _ = torch.tensor([0], device=device)
            torch.cuda.synchronize(int(device.split(':')[1]))
        self.tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        try:
            self.model = AutoModel.from_pretrained(
                MODEL_PATH, trust_remote_code=True, use_safetensors=True,
                _attn_implementation='eager')  # flash_attn на Windows нет
        except TypeError:
            self.model = AutoModel.from_pretrained(
                MODEL_PATH, trust_remote_code=True, use_safetensors=True)
        self.model = self.model.eval().to(device).to(torch.bfloat16)

    def ocr(self, png_path: str) -> str:
        with torch.no_grad():
            res = self.model.infer(
                self.tok, prompt=PROMPT, image_file=png_path,
                output_path=str(OUT_DIR), base_size=1024, image_size=640,
                crop_mode=True, save_results=False,
                eval_mode=True)  # только в eval_mode infer() возвращает текст
        return (res or '').strip()


def get_engine():
    global _instance
    if _instance is None:
        _instance = DeepSeekEngine()
    return _instance
