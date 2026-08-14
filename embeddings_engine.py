# embeddings_engine.py
# Версия: 4.0
# Дата: 2026-06-06
# Описание: Эмбеддинги на nomic-embed-text-v1.5 + nomic-embed-vision-v1.5.
#            Единое векторное пространство 768-dim для текста и изображений.
#            Батчевая обработка, сохранение .npy, fallback на single.

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import threading
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor, PreTrainedModel
from PIL import Image
from typing import List, Optional
import numpy as np
from pathlib import Path

from config import cfg
from logger_utils import logger

Image.MAX_IMAGE_PIXELS = cfg.MAX_IMAGE_PIXELS

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


TEXT_PREFIX = "search_document: "


def _clear_gpu_memory(device: str = None):
    """Очистка GPU памяти перед загрузкой модели."""
    try:
        if device is None:
            device = cfg.EMB_GPU_DEVICE
        
        if torch.cuda.is_available():
            device_idx = 0
            if device.startswith('cuda:'):
                device_idx = int(device.split(':')[1])
            
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device_idx)
            
            allocated = torch.cuda.memory_allocated(device_idx) / 1024**3
            reserved = torch.cuda.memory_reserved(device_idx) / 1024**3
            logger.info(f"GPU {device} cleared: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    except Exception as e:
        logger.warning(f"Failed to clear GPU memory: {e}")


def _ensure_cuda_context():
    """Инициализация CUDA-контекста в текущем потоке."""
    if torch.cuda.is_available():
        device_id = 0
        dev = cfg.EMB_GPU_DEVICE
        if dev.startswith('cuda:'):
            device_id = int(dev.split(':')[1])
        
        # Проверка текущего устройства для избежания [Errno 22] при повторном вызове
        try:
            current_device = torch.cuda.current_device()
            if current_device != device_id:
                logger.debug(f"Switching CUDA context from {current_device} to {device_id}")
        except Exception:
            pass
        
        # Принудительная инициализация CUDA-контекста в текущем потоке.
        _ = torch.tensor([0], device=dev)
        torch.cuda.synchronize(device_id)


def unload_all_models():
    """Выгрузка всех моделей из GPU для освобождения памяти."""
    global _engine_instance
    if _engine_instance is not None:
        if _engine_instance.text_model is not None:
            del _engine_instance.text_model
            _engine_instance.text_model = None
            logger.info("Text embedding model unloaded from GPU")
        if _engine_instance.image_model is not None:
            del _engine_instance.image_model
            _engine_instance.image_model = None
            logger.info("Image embedding model unloaded from GPU")
        _clear_gpu_memory(cfg.EMB_GPU_DEVICE)


def _mean_pool(last_hidden_state, attention_mask):
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    sum_embeddings = torch.sum(last_hidden_state * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


class EmbeddingEngine:
    def __init__(self):
        self.text_model = None
        self.text_tokenizer = None
        self.image_model = None
        self.image_processor = None
        self.device = cfg.EMB_GPU_DEVICE

    def load_models(self):
        self._load_text_model()
        self._load_image_model()

    def _load_text_model(self):
        if self.text_model is not None:
            return
        _ensure_cuda_context()
        logger.info(f"Loading TEXT model: {cfg.TEXT_EMB_MODEL_NAME} on {self.device}")
        _clear_gpu_memory(self.device)
        try:
            self.text_model = AutoModel.from_pretrained(
                cfg.TEXT_EMB_MODEL_PATH, trust_remote_code=True
            ).to(self.device)
            self.text_tokenizer = AutoTokenizer.from_pretrained(
                cfg.TEXT_EMB_MODEL_PATH, trust_remote_code=True
            )
            self.text_model.eval()
            logger.info("TEXT model loaded successfully.")
        except Exception as e:
            logger.critical(f"TEXT model load failed: {e}")
            raise

    def _load_image_model(self):
        if self.image_model is not None:
            return
        _ensure_cuda_context()
        logger.info(f"Loading IMAGE model: {cfg.IMAGE_EMB_MODEL_NAME} on {self.device}")
        _clear_gpu_memory(self.device)
        try:
            self.image_model = AutoModel.from_pretrained(
                cfg.IMAGE_EMB_MODEL_PATH, trust_remote_code=True
            ).to(self.device)
            self.image_processor = CLIPImageProcessor.from_pretrained(
                cfg.IMAGE_EMB_MODEL_PATH, trust_remote_code=True
            )
            self.image_model.eval()
            logger.info("IMAGE model loaded successfully.")
        except Exception as e:
            logger.critical(f"IMAGE model load failed: {e}")
            raise

    def compute_text_embedding(self, text: str) -> Optional[np.ndarray]:
        if not text:
            return None
        self._load_text_model()
        try:
            prefixed = TEXT_PREFIX + text
            with torch.no_grad():
                inputs = self.text_tokenizer(
                    [prefixed], padding=True, truncation=True,
                    max_length=8192, return_tensors="pt"
                ).to(self.device)
                outputs = self.text_model(**inputs)
                embedding = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
                embedding = self._safe_normalize(embedding)
                return embedding.cpu().numpy()[0]
        except Exception as e:
            logger.error(f"Text embedding error: {e}")
            return None

    def _safe_normalize(self, embeddings: torch.Tensor) -> torch.Tensor:
        norms = torch.norm(embeddings, p=2, dim=1, keepdim=True)
        norms = torch.clamp(norms, min=1e-8)
        normalized = embeddings / norms
        if torch.isnan(normalized).any():
            nan_mask = torch.isnan(normalized).any(dim=1)
            logger.warning(f"NaN detected after normalization in {nan_mask.sum().item()} vectors, replacing with zeros")
            normalized[nan_mask] = 0.0
        return normalized

    def compute_image_embedding(self, image_path: str) -> Optional[np.ndarray]:
        self._load_image_model()
        try:
            img = Image.open(image_path).convert("RGB")
            with torch.no_grad():
                inputs = self.image_processor(
                    images=img, return_tensors="pt"
                ).to(self.device)
                outputs = self.image_model(**inputs)
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    embedding = outputs.pooler_output
                else:
                    embedding = outputs.last_hidden_state[:, 0]
                embedding = self._safe_normalize(embedding)
                return embedding.cpu().numpy()[0]
        except Exception as e:
            logger.error(f"Image embedding error for {image_path}: {e}")
            return None

    def get_text_embeddings(
        self,
        texts: List[str],
        save_dir: Path,
        names: List[str],
        batch_size: int = 0,
    ) -> np.ndarray:
        if not texts:
            return np.array([])
        self._load_text_model()
        if batch_size <= 0:
            batch_size = cfg.EMB_BATCH_SIZE

        save_dir.mkdir(parents=True, exist_ok=True)
        all_embeddings = []

        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch_texts = [TEXT_PREFIX + t for t in texts[start:end]]
            batch_names = names[start:end]

            try:
                with torch.no_grad():
                    inputs = self.text_tokenizer(
                        batch_texts, padding=True, truncation=True,
                        max_length=2048, return_tensors="pt"
                    ).to(self.device)
                    outputs = self.text_model(**inputs)
                    embeddings = _mean_pool(outputs.last_hidden_state, inputs["attention_mask"])
                    embeddings = self._safe_normalize(embeddings)
                    batch_emb = embeddings.cpu().numpy()

                for i, name in enumerate(batch_names):
                    safe_name = Path(name).stem
                    np.save(save_dir / f"{safe_name}_{start + i}_text.npy", batch_emb[i])

                all_embeddings.append(batch_emb)
            except Exception as e:
                logger.error(f"Text batch {start}-{end} error: {e}")
                batch_fallback = np.zeros((len(batch_names), cfg.EMB_DIMENSION), dtype=np.float32)
                for i in range(len(batch_names)):
                    try:
                        single = self.compute_text_embedding(texts[start + i])
                        if single is not None:
                            safe_name = Path(batch_names[i]).stem
                            np.save(save_dir / f"{safe_name}_{start + i}_text.npy", single)
                            batch_fallback[i] = single
                    except Exception as e2:
                        logger.error(f"Fallback text embedding error for {batch_names[i]}: {e2}")
                all_embeddings.append(batch_fallback)

            if (start // batch_size) % 3 == 0:
                logger.info(f"Text embeddings: {end}/{len(texts)}")

        if all_embeddings:
            return np.vstack(all_embeddings)
        return np.array([])

    def get_image_embeddings(
        self,
        image_paths: List[str],
        save_dir: Path,
        names: List[str],
        batch_size: int = 0,
    ) -> np.ndarray:
        if not image_paths:
            return np.array([])
        self._load_image_model()
        if batch_size <= 0:
            batch_size = cfg.EMB_BATCH_SIZE

        save_dir.mkdir(parents=True, exist_ok=True)
        dim = cfg.EMB_DIMENSION
        all_embeddings = []

        for start in range(0, len(image_paths), batch_size):
            end = min(start + batch_size, len(image_paths))
            batch_paths = image_paths[start:end]
            batch_names = names[start:end]

            try:
                images = []
                valid_indices = []
                for idx, p in enumerate(batch_paths):
                    try:
                        img = Image.open(p).convert("RGB")
                        images.append(img)
                        valid_indices.append(idx)
                    except Exception as e:
                        logger.warning(f"Cannot open image {p}: {e}")

                if not images:
                    batch_emb = np.zeros((len(batch_paths), dim))
                    all_embeddings.append(batch_emb)
                    continue

                with torch.no_grad():
                    inputs = self.image_processor(
                        images=images, return_tensors="pt"
                    ).to(self.device)
                    outputs = self.image_model(**inputs)
                    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                        embeddings = outputs.pooler_output
                    else:
                        embeddings = outputs.last_hidden_state[:, 0]
                    embeddings = self._safe_normalize(embeddings)
                    batch_emb = embeddings.cpu().numpy()

                for local_i, global_i in enumerate(valid_indices):
                    safe_name = Path(batch_names[global_i]).stem
                    np.save(save_dir / f"{safe_name}_{start + global_i}_image.npy", batch_emb[local_i])

                # Восстанавливаем полный массив: zero-вектор для упавших изображений
                full_batch = np.zeros((len(batch_paths), dim))
                for local_i, global_i in enumerate(valid_indices):
                    full_batch[global_i] = batch_emb[local_i]
                all_embeddings.append(full_batch)
            except Exception as e:
                logger.error(f"Image batch {start}-{end} error: {e}")
                all_embeddings.append(np.zeros((len(batch_paths), dim)))
                for i in range(len(batch_paths)):
                    try:
                        single = self.compute_image_embedding(batch_paths[i])
                        if single is not None:
                            safe_name = Path(batch_names[i]).stem
                            np.save(save_dir / f"{safe_name}_{start + i}_image.npy", single)
                            all_embeddings[-1][i] = single
                    except Exception as e2:
                        logger.error(f"Fallback image embedding error for {batch_names[i]}: {e2}")

            if (start // batch_size) % 3 == 0:
                logger.info(f"Image embeddings: {end}/{len(image_paths)}")

        if all_embeddings:
            return np.vstack(all_embeddings)
        return np.array([])


_engine_instance = None
_engine_lock = threading.Lock()


def get_engine() -> EmbeddingEngine:
    global _engine_instance
    with _engine_lock:
        if _engine_instance is None:
            _engine_instance = EmbeddingEngine()
            _engine_instance.load_models()
    return _engine_instance


def clear_all_gpu_models():
    """Выгрузка всех моделей из всех GPU для полного освобождения памяти."""
    unload_all_models()
    try:
        from llm_summarizer import unload_model
        unload_model()
    except Exception as e:
        logger.debug(f"LLM unload skipped: {e}")