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

import sys

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

# ── Патч для giga_3b: transformers 5.x при миграции на rope_parameters
# убрал ключ 'default' из ROPE_INIT_FUNCTIONS (см. FutureWarning про
# rope_config_validation → RotaryEmbeddingConfigMixin.validate_rope), а
# modeling_gigarembed.py (giga_3b) написан под старый API и падает с
# KeyError: 'default' в GigarRotaryEmbedding.__init__. 'proportional' с
# параметрами по умолчанию (partial_rotary_factor=1.0, factor=1.0) даёт
# математически идентичную несмасштабированную RoPE — безопасная замена.
from transformers.modeling_rope_utils import (  # noqa: E402
    ROPE_INIT_FUNCTIONS, _compute_proportional_rope_parameters)

if 'default' not in ROPE_INIT_FUNCTIONS:
    ROPE_INIT_FUNCTIONS['default'] = _compute_proportional_rope_parameters


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

    def __init__(self, key: str, device: str = 'cuda:0', logger=print,
                 load_dtype=None):
        # load_dtype: в каком типе ХРАНИТЬ веса.
        #
        # None (умолчание) — тип не указывается при загрузке: transformers берёт
        # torch_dtype из config.json модели, а если его там нет — грузит в fp32.
        # Именно так считались ВСЕ сохранённые векторы в embeddings/ и все
        # метрики отчётов M3-M9, поэтому умолчание менять нельзя: иначе новые
        # прогоны станут несопоставимы с уже посчитанным.
        #
        # Фактический тип весов по моделям (2026-08-30, проверено загрузкой,
        # не разбором config.json — у giga_480m тип лежит в ключе dtype, а не
        # torch_dtype, и по torch_dtype читается как «не указан»):
        #   bfloat16 — qwen3e_4b (7.49 ГБ), qwen3e_06b (1.11), giga_480m (0.90);
        #   float32  — nomic_v15 (0.51), nomic_v2 (1.77), e5_large (2.09),
        #              bge_m3 (2.12), rubert_tiny2 (0.11), rubert_base_dp (0.66),
        #              sbert_ru (1.59), giga_3b (12.85).
        #
        # Форвард в _forward_batch идёт под autocast(bfloat16), который приводит
        # к bf16 только активации. У fp32-весов получается худшее из двух: память
        # по fp32, точность по bf16. Для крупных моделей это дорого — у giga_3b
        # веса занимают 12 ГБ из 24, остатка не хватает на активации, encode()
        # ловит OOM и вдвое режет батч. Замер на 46,501 XML (2026-08-30):
        # load_dtype=bfloat16 даёт у giga_3b 144 док/с против 118 и пик VRAM
        # 10.9 ГБ против 22.2. Из восьми fp32-моделей это касается только её —
        # у остальных веса 0.1-2.1 ГБ, давления на память нет, батч не режется.
        #
        # ВНИМАНИЕ, если будете передавать bfloat16 явно: эквивалентность
        # векторов bf16-весам НЕ подтверждена. У giga_3b они и без смены типа
        # не воспроизводятся между прогонами (cos 0.93 на шарде 0 при повторе
        # того же прогона), у остальных моделей не проверялось. Считать метрики
        # на bf16-векторах и сравнивать их с записанными в отчётах нельзя.
        self.key = key
        self.meta = MODELS[key]
        self.device = device
        self.log = logger
        self.load_dtype = load_dtype
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
        # dtype не передаём вовсе, если не задан явно — так вёл себя вызов до
        # правки от 2026-08-29, и так посчитаны все сохранённые векторы.
        dtype_kw = {} if self.load_dtype is None else {'dtype': self.load_dtype}
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, trust_remote_code=self.meta['trust_remote_code'])
        if self.meta.get('pooling') == 'latent_attention':
            # giga_3b: AutoModel.from_pretrained() резолвит configuration_gigarembed.py
            # ДВАЖДЫ независимо — один раз для AutoConfig, второй раз как relative
            # import внутри modeling_gigarembed.py — в ДВЕ разные хэш-папки кэша
            # transformers_modules → два разных объекта класса LatentAttentionConfig
            # → AutoModel.register() не совпадает по identity ("Unrecognized
            # configuration class ... LatentAttentionConfig"). Обход: получаем класс
            # модели через get_class_from_dynamic_module и берём GigarEmbedConfig из
            # ТОГО ЖЕ модуля (той же хэш-папки) — гарантированно единый объект класса.
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            model_cls = get_class_from_dynamic_module(
                'modeling_gigarembed.GigarEmbedModel', path)
            config_module_name = model_cls.__module__.rsplit('.', 1)[0] + '.configuration_gigarembed'
            config_cls = sys.modules[config_module_name].GigarEmbedConfig
            config = config_cls.from_pretrained(path)
            self.model = model_cls.from_pretrained(
                path, config=config, **dtype_kw).to(self.device)
        else:
            self.model = AutoModel.from_pretrained(
                path, trust_remote_code=self.meta['trust_remote_code'],
                **dtype_kw).to(self.device)
        self.model.eval()
        weights_dtype = str(next(self.model.parameters()).dtype).replace('torch.', '')
        self.log(f"[{self.key}] loaded, dim={self.meta['dim']}, веса {weights_dtype}")
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
        pooling = self.meta.get('pooling')
        # giga_3b (Latent-Attention): модель САМА пулит через кастомный
        # forward(..., return_embeddings=True) — на выходе уже вектор
        # (batch, dim), а не last_hidden_state. Пулинг ниже НЕ применяется.
        forward_kwargs = {'return_embeddings': True} if pooling == 'latent_attention' else {}
        try:
            with torch.no_grad(), torch.autocast(self.device.split(':')[0], dtype=torch.bfloat16):
                outputs = self.model(**inputs, **forward_kwargs)
        except Exception:
            with torch.no_grad():
                outputs = self.model(**inputs, **forward_kwargs)
        if pooling == 'latent_attention':
            # Ожидается тензор (batch, dim) или объект с .last_hidden_state
            # того же вида — оба варианта видел в model card giga_3b.
            emb = outputs if torch.is_tensor(outputs) else outputs.last_hidden_state
            emb = emb.float()
        else:
            hidden = outputs.last_hidden_state.float()
            if pooling == 'last_token':
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
