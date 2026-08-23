# xml_embed_helper.py
# Текстовые эмбеддинги nomic-embed-text-v1.5 для XML-эксперимента.
#
# Отличия от experiment_excel/embed_helper.py:
#   - bf16 autocast (4.4x быстрее fp32 на RTX PRO 4000: 33→144 док/с);
#   - сортировка по длине внутри чанка (меньше паддинга);
#   - та же модель, тот же TEXT_PREFIX, та же L2-нормализация —
#     векторное пространство идентично предыдущему эксперименту
#     (расхождение bf16/fp32 ~1e-3 по cosine, на порог 0.8 не влияет).

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
EXPERIMENT_EXCEL = PROJECT_ROOT / 'experiment_excel'
if str(EXPERIMENT_EXCEL) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_EXCEL))

from config import cfg  # noqa: E402
from embeddings_engine import _mean_pool, TEXT_PREFIX  # noqa: E402
from embed_helper import get_text_engine  # noqa: E402  (синглтон модели)


def encode_texts(texts, batch_size: int = 64, max_length: int = 2048) -> np.ndarray:
    """Батчевый эмбеддинг: (N, 768) float32, L2-нормализованные.

    Внутри: сортировка по длине текста (паддинг ≈ минимальный),
    bf16 autocast. Порядок результата соответствует входу.
    """
    if not texts:
        return np.empty((0, cfg.EMB_DIMENSION), dtype=np.float32)
    engine = get_text_engine()
    out = np.zeros((len(texts), cfg.EMB_DIMENSION), dtype=np.float32)

    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    for start in range(0, len(order), batch_size):
        idxs = order[start:start + batch_size]
        batch = [TEXT_PREFIX + (texts[i] or '') for i in idxs]
        try:
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                inputs = engine.text_tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=max_length, return_tensors='pt'
                ).to(engine.device)
                outputs = engine.text_model(**inputs)
            emb = _mean_pool(outputs.last_hidden_state.float(), inputs['attention_mask'])
            emb = engine._safe_normalize(emb)
            out[idxs] = emb.cpu().numpy().astype(np.float32)
        except Exception as e:  # noqa: BLE001
            print(f"\n  batch {start} failed ({type(e).__name__}: {e}), per-item fallback")
            for i in idxs:
                single = engine.compute_text_embedding(texts[i] or '')
                if single is not None:
                    out[i] = single.astype(np.float32)
        done = min(start + batch_size, len(order))
        print(f"\r  embeddings: {done}/{len(order)}", end='', flush=True)
    print()
    return out
