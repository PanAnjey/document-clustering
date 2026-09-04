# embed_helper.py
# Текстовые эмбеддинги nomic-embed-text-v1.5 для эксперимента.
#
# Отличия от прямого использования EmbeddingEngine:
#   - грузится ТОЛЬКО текстовая модель (image-модель ~2 ГБ VRAM на cuda:1
#     эксперименту не нужна; get_engine() грузил бы обе);
#   - эмбеддинги НЕ пишутся в .npy (get_text_embeddings сохраняет файлы
#     на диск — для временных эмбеддингов эксперимента это лишний I/O);
#   - сигнатура encode_texts(texts, batch_size) -> np.ndarray (N, 768) float32,
#     L2-нормализованные (как и в продакшен-движке).

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import cfg  # noqa: E402
from embeddings_engine import EmbeddingEngine, _mean_pool, TEXT_PREFIX  # noqa: E402

_engine = None


def get_text_engine() -> EmbeddingEngine:
    """Singleton-движок с загруженной только текстовой моделью."""
    global _engine
    if _engine is None:
        _engine = EmbeddingEngine()
        _engine._load_text_model()  # намеренно НЕ вызываем _load_image_model()
    return _engine


def encode_texts(texts, batch_size: int = 0) -> np.ndarray:
    """Батчевый эмбеддинг списка текстов.

    Returns:
        np.ndarray (N, cfg.EMB_DIMENSION) float32, L2-нормализованный.
        Для упавших батчей — per-item fallback (нулевой вектор в худшем случае),
        как в EmbeddingEngine.get_text_embeddings.
    """
    if not texts:
        return np.empty((0, cfg.EMB_DIMENSION), dtype=np.float32)
    if batch_size <= 0:
        batch_size = cfg.EMB_BATCH_SIZE

    engine = get_text_engine()
    out = []
    n = len(texts)
    for start in range(0, n, batch_size):
        batch_texts = texts[start:start + batch_size]
        prefixed = [TEXT_PREFIX + (t or "") for t in batch_texts]
        try:
            with torch.no_grad():
                inputs = engine.text_tokenizer(
                    prefixed, padding=True, truncation=True,
                    max_length=2048, return_tensors="pt"
                ).to(engine.device)
                outputs = engine.text_model(**inputs)
                emb = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
                emb = engine._safe_normalize(emb)
                out.append(emb.cpu().numpy().astype(np.float32))
        except Exception as e:
            print(f"\n  ⚠️ batch {start}-{start + len(batch_texts)} failed ({e}), per-item fallback")
            fallback = np.zeros((len(batch_texts), cfg.EMB_DIMENSION), dtype=np.float32)
            for i, t in enumerate(batch_texts):
                single = engine.compute_text_embedding(t or "")
                if single is not None:
                    fallback[i] = single.astype(np.float32)
            out.append(fallback)
        done = min(start + batch_size, n)
        print(f"\r  embeddings: {done}/{n}", end="", flush=True)
    print()
    return np.vstack(out)
