# hf_summarizer.py
# Версия: 1.2
# Дата: 2026-06-12
# Описание: Саммаризация через Transformers (Qwen3.5-4B) с batched inference.
#            v1.2: Единый промпт с llama-бэкендом (ТЕМА/ТИП/НАЗНАЧЕНИЕ),
#                  stripping reasoning-блока Qwen, простой парсинг через
#                  llm_summarizer._parse_summary.

import re
import time
import threading
import torch
from typing import List, Optional
from pathlib import Path
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

from config import cfg
from logger_utils import logger


_model_instance = None
_tokenizer_instance = None
_model_lock = threading.Lock()

SYSTEM_PROMPT = """\
Классифицируй документ по трём полям: ТЕМА, ТИП, НАЗНАЧЕНИЕ.
Каждое поле на отдельной строке. Без пояснений."""


def _strip_reasoning(text: str) -> str:
    """Удаляет <think>...</think> блоки Qwen (если остались)."""
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    return text.strip()


def _ensure_cuda_context():
    """Инициализация CUDA-контекста в текущем потоке."""
    if torch.cuda.is_available():
        device_id = 0
        dev = cfg.LLM_GPU_DEVICE
        if dev.startswith('cuda:'):
            device_id = int(dev.split(':')[1])
        
        # Проверка текущего устройства для избежания [Errno 22] при повторном вызове
        try:
            current_device = torch.cuda.current_device()
            if current_device != device_id:
                logger.debug(f"Switching CUDA context from {current_device} to {device_id}")
        except Exception:
            pass
        
        try:
            _ = torch.tensor([0], device=dev)
            torch.cuda.synchronize(device_id)
        except Exception as e:
            logger.warning(f"CUDA init failed on {dev}: {e}")
            raise


def load_model():
    global _model_instance, _tokenizer_instance
    with _model_lock:
        if _model_instance is not None:
            return

        _ensure_cuda_context()
        logger.info(f"Loading HF model from {cfg.HF_MODEL_PATH}...")
        t0 = time.time()

        _tokenizer_instance = AutoTokenizer.from_pretrained(
            str(cfg.HF_MODEL_PATH),
            trust_remote_code=True,
        )
        _tokenizer_instance.padding_side = "left"
        _tokenizer_instance.pad_token = _tokenizer_instance.eos_token

        _model_instance = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(cfg.HF_MODEL_PATH),
            torch_dtype=torch.bfloat16,
            device_map=cfg.LLM_GPU_DEVICE,
            trust_remote_code=True,
        )
        _model_instance.eval()

        elapsed = time.time() - t0
        alloc = torch.cuda.memory_allocated() / 1e9
        logger.info(
            f"HF model loaded in {elapsed:.1f}s, "
            f"VRAM: {alloc:.2f} GB"
        )

        warmup_inputs = _tokenizer_instance(
            ["тест"],
            return_tensors="pt",
            padding=True,
        ).to(cfg.LLM_GPU_DEVICE)
        with torch.no_grad():
            _model_instance.generate(
                **warmup_inputs,
                max_new_tokens=1,
                do_sample=False,
            )
        logger.debug("HF warmup done")


def unload_model():
    global _model_instance, _tokenizer_instance
    if _model_instance is not None:
        del _model_instance
        _model_instance = None
    _tokenizer_instance = None
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    logger.info("HF model unloaded, GPU cache cleared")


def check_available() -> bool:
    try:
        load_model()
        return True
    except Exception as e:
        logger.error(f"HF model check failed: {e}")
        return False


def _make_prompt(text: str) -> str:
    """Формирует промпт с one-shot примером в user-сообщении."""
    user_content = (
        "Пример:\n"
        "Текст документа: Договор аренды помещения №5 от 01.01.2026 между ООО Арендатор и ИП Ивановым на 11 месяцев.\n"
        "Ответ:\n"
        "ТЕМА: Аренда\n"
        "ТИП: Договор аренды\n"
        "НАЗНАЧЕНИЕ: Договор аренды нежилого помещения\n\n"
        "Теперь классифицируй:\n"
        f"Текст документа: {text}\n"
        "Ответ:\n"
        "ТЕМА:"
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    try:
        return _tokenizer_instance.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Старые версии transformers не поддерживают enable_thinking
        return _tokenizer_instance.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


def summarize(texts: List[str]) -> List[Optional[str]]:
    if not texts:
        return []

    load_model()

    prompts = [_make_prompt(t) for t in texts]

    inputs = _tokenizer_instance(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=cfg.LLM_MAX_CONTEXT_TOKENS,
    ).to(cfg.LLM_GPU_DEVICE)

    with torch.no_grad():
        out = _model_instance.generate(
            **inputs,
            max_new_tokens=cfg.HF_MAX_TOKENS,
            do_sample=False,
            pad_token_id=_tokenizer_instance.pad_token_id,
            eos_token_id=_tokenizer_instance.eos_token_id,
        )

    results = []
    for i in range(len(texts)):
        gen_ids = out[i][inputs.input_ids.shape[1]:]
        decoded = _tokenizer_instance.decode(gen_ids, skip_special_tokens=True)
        stripped = _strip_reasoning(decoded.strip()) if decoded else None
        results.append(stripped or None)

    return results


def _parse_summary(llm_output: str, original_text: str) -> tuple:
    """Парсит вывод Qwen через стандартный парсер (сначала strip reasoning)."""
    from llm_summarizer import _parse_summary as _llm_parse
    cleaned = _strip_reasoning(llm_output) if llm_output else llm_output
    return _llm_parse(cleaned or "", original_text)


def _save_summaries_to_db(records: list):
    from llm_summarizer import _save_summaries_to_db as _llm_save
    _llm_save(records)
