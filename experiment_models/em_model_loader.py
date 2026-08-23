# em_model_loader.py
# Загрузка моделей и батчевый эмбеддинг для сравнительного эксперимента.
#
# Единая методология для всех моделей:
#   prefix_doc + текст → tokenizer (truncation max_len) → model →
#   mean-pooling по attention_mask → L2-нормализация → float32 (N, dim).
#
# Скорость: сортировка по длине текста (минимум паддинга) + bf16 autocast
# (fallback на fp32 при любой ошибке батча). При CUDA OOM батч делится
# пополам автоматически (важно для дисплейной cuda:1 с занятой VRAM).
#
# ВНИМАНИЕ: для nomic (trust_remote_code) обязательны патчи
# PreTrainedModel (баг all_tied_weights_keys в кастомном коде nomic,
# см. embeddings_engine.py строки 25-41).

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedModel

from em_config import MODELS, model_local_path

# ── Патчи для кастомного кода nomic (v1.5 и v2-moe) ──────────────
_orig_mark_tied = PreTrainedModel.mark_tied_weights_as_initialized


def _patched_mark_tied(self, loading_info):
    if not hasattr(self, 'all_tied_weights_keys'):
        self.all_tied_weights_keys = {}
    return _orig_mark_tied(self, loading_info)


PreTrainedModel.mark_tied_weights_as_initialized = _patched_mark_tied

_orig_move_missing = PreTrainedModel._move_missing_keys_from_meta_to_device


def _patched_move_missing(self, *args, **kwargs):
    if not hasattr(self, 'all_tied_weights_keys') or self.all_tied_weights_keys is None:
        self.all_tied_weights_keys = {}
    return _orig_move_missing(self, *args, **kwargs)


PreTrainedModel._move_missing_keys_from_meta_to_device = _patched_move_missing


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    return summed / torch.clamp(mask.sum(dim=1), min=1e-9)


def last_token_pool(last_hidden_states, attention_mask):
    """Qwen3-Embedding pooling (model card). Токенизатор паддит справа —
    позиция последнего непаддингового токена = sum(mask)-1."""
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


class EmbeddingModel:
    """Одна модель на одном GPU. Контекст: with EmbeddingModel(...) as em."""

    def __init__(self, key: str, device: str = 'cuda:0', logger=print):
        self.key = key
        self.meta = MODELS[key]
        self.device = device
        self.log = logger
        self.model = None
        self.tokenizer = None

    def load(self):
        path = str(model_local_path(self.key))
        # Явная инициализация CUDA-контекста в этом потоке (gotcha 9b)
        if self.device.startswith('cuda'):
            _ = torch.tensor([0], device=self.device)
            torch.cuda.synchronize(int(self.device.split(':')[1]))
            free_gb, total_gb = [v / 1024**3 for v in torch.cuda.mem_get_info(self.device)]
            self.log(f"[{self.key}] {self.device}: free {free_gb:.1f}/{total_gb:.1f} GB")
        self.log(f"[{self.key}] loading {self.meta['hf_name']} from {path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=self.meta['trust_remote_code'])
        self.model = AutoModel.from_pretrained(
            path, trust_remote_code=self.meta['trust_remote_code']).to(self.device)
        self.model.eval()
        self.log(f"[{self.key}] loaded, dim={self.meta['dim']}")
        return self

    def unload(self):
        if self.model is not None:
            del self.model
            self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self):
        return self.load()

    def __exit__(self, *exc):
        self.unload()

    def _forward_batch(self, batch):
        """Один батч: bf16 autocast, при ошибке — fp32."""
        prefix = self.meta['prefix_doc']
        inputs = self.tokenizer(
            [prefix + t for t in batch], padding=True, truncation=True,
            max_length=self.meta['max_len'], return_tensors='pt',
        ).to(self.device)
        try:
            with torch.no_grad(), torch.autocast(self.device.split(':')[0], dtype=torch.bfloat16):
                outputs = self.model(**inputs)
        except Exception:
            with torch.no_grad():
                outputs = self.model(**inputs)
        hidden = outputs.last_hidden_state.float()
        if self.meta.get('pooling') == 'last_token':
            emb = last_token_pool(hidden, inputs['attention_mask'])
        else:
            emb = mean_pool(hidden, inputs['attention_mask'])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy().astype(np.float32)

    def encode(self, texts, batch_size=None, log_every=10):
        """(N, dim) float32. Порядок соответствует входу."""
        if batch_size is None:
            batch_size = self.meta['batch']
        n = len(texts)
        dim = self.meta['dim']
        out = np.zeros((n, dim), dtype=np.float32)
        if n == 0:
            return out

        order = sorted(range(n), key=lambda i: len(texts[i] or ''))
        cur_bs = batch_size
        pos = 0
        n_batches = (n + cur_bs - 1) // cur_bs
        while pos < len(order):
            idxs = order[pos:pos + cur_bs]
            batch = [(texts[i] or '') for i in idxs]
            try:
                out[idxs] = self._forward_batch(batch)
                pos += cur_bs
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if cur_bs > 8:
                    cur_bs //= 2
                    self.log(f"[{self.key}] CUDA OOM → batch_size {cur_bs}")
                else:  # совсем мелкий батч — по одному элементу
                    for i in idxs:
                        try:
                            out[i] = self._forward_batch([texts[i] or ''])
                        except Exception as e:  # noqa: BLE001
                            self.log(f"[{self.key}] doc {i} failed: {e}")
                    pos += cur_bs
            except Exception as e:  # noqa: BLE001
                self.log(f"[{self.key}] batch failed ({type(e).__name__}: {e}), per-item")
                for i in idxs:
                    try:
                        out[i] = self._forward_batch([texts[i] or ''])
                    except Exception as e2:  # noqa: BLE001
                        self.log(f"[{self.key}] doc {i} failed: {e2}")
                pos += cur_bs
            done = min(pos, len(order))
            if (done // max(cur_bs, 1)) % log_every == 0 or done == len(order):
                self.log(f"[{self.key}] {done}/{n}")
        return out
