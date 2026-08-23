# engines/oc_qwen3vl4b.py
# Движок Qwen3-VL-4B-Instruct (Qwen/Qwen3-VL-4B-Instruct) — «золотая середина»
# между DeepSeek-OCR (3B MoE) и Qwen3-VL-8B. Тот же класс Qwen3VLEngine,
# другой model_path. GPU (bf16, ~9 GB VRAM).

from engines.oc_qwen3vl import Qwen3VLEngine

MODEL_PATH = r"D:\MODELS\Transformers\Qwen3-VL-4B-Instruct"

_instance = None


def get_engine(device: str = 'cuda:0'):
    global _instance
    if _instance is None:
        # repetition_penalty=1.15: без него 4B на бланках впадает в
        # HTML/imgur-галлюцинацию (<img src="https://i.imgur.com/..."> до
        # max tokens, F1=0.00). Проверено на doc 161468: 0.00 -> 0.966,
        # на нормальных документах качество не меняется.
        _instance = Qwen3VLEngine(device=device, model_path=MODEL_PATH,
                                  name='qwen3vl4b',
                                  gen_overrides={'repetition_penalty': 1.15})
    return _instance
