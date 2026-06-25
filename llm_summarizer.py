# llm_summarizer.py
# Бывший v5.0 (llama-server), переписан: только transformers (Qwen3.5-4B).
# Оставлены: парсинг вывода, БД, общие утилиты GPU.

import asyncio
import time
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path

from config import cfg
from logger_utils import logger
from pipeline_progress import progress


def _report_gpu_memory(label: str = ""):
    try:
        import torch
        if not torch.cuda.is_available():
            return
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            free = torch.cuda.get_device_properties(i).total_memory / 1024**3 - alloc
            tag = f" [{label}]" if label else ""
            logger.info(f"GPU{i} ({torch.cuda.get_device_name(i)}){tag}: {alloc:.2f}G used / {reserved:.2f}G reserved / {free:.2f}G free")
    except Exception:
        pass


def _clear_gpu_memory(device: str = None):
    try:
        import torch
        if not torch.cuda.is_available():
            return
        torch.cuda.empty_cache()
        idx = int(device.split(":")[1]) if device and ":" in device else 0
        torch.cuda.synchronize(idx)
    except Exception:
        pass


def _clear_all_gpu_memory():
    try:
        import torch
        if torch.cuda.is_available():
            try:
                from embeddings_engine import unload_all_models
                unload_all_models()
            except ImportError:
                pass
            for i in range(torch.cuda.device_count()):
                torch.cuda.empty_cache()
                torch.cuda.synchronize(i)
    except Exception:
        pass


def _trim_text(text: str, max_chars: int = 0) -> str:
    if not text:
        return ""
    if max_chars <= 0:
        max_chars = cfg.LLM_MAX_TEXT_LENGTH
    if len(text) <= max_chars:
        return text
    head = text[: max_chars // 2]
    tail = text[len(text) - (max_chars // 2):]
    return head + "\n...[TRUNCATED]...\n" + tail


def _parse_summary(llm_output: str, original_text: str) -> Tuple[str, Optional[str], Optional[str], Optional[str]]:
    if not llm_output:
        return original_text, None, None, None

    import re

    cleaned = llm_output.strip()
    cleaned = re.sub(r'<think[\s>].*?</think\s*>', '', cleaned, flags=re.DOTALL)
    cleaned = re.sub(r'<thinking[\s>].*?</thinking\s*>', '', cleaned, flags=re.DOTALL)

    def _strip_md(s: str) -> str:
        s = s.strip()
        s = re.sub(r'\*{1,2}', '', s)
        s = re.sub(r'_{1,2}', '', s)
        s = re.sub(r'`{1,3}', '', s)
        return s.strip()

    keys = ["ТЕМА", "TOPIC", "ТИП", "TYPE", "НАЗНАЧЕНИЕ", "SUMMARY"]
    parts = {"ТЕМА": "", "ТИП": "", "НАЗНАЧЕНИЕ": ""}

    lines = cleaned.split("\n")
    current_key = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        bullets = re.sub(r'^[\d\.\)\-\*]+\s*', '', line)
        stripped = _strip_md(bullets)
        matched = False
        for key in keys:
            if re.match(rf'^{re.escape(key)}\s*[:\-–—]\s*', stripped, re.IGNORECASE):
                normalized = {"TOPIC": "ТЕМА", "TYPE": "ТИП", "SUMMARY": "НАЗНАЧЕНИЕ"}.get(key, key)
                value = re.sub(rf'^{re.escape(key)}\s*[:\-–—]\s*', '', stripped, flags=re.IGNORECASE).strip()
                value = _strip_md(value)
                if value:
                    parts[normalized] = value
                current_key = normalized
                matched = True
                break
        if not matched and current_key and current_key in parts:
            continuation = _strip_md(stripped)
            if continuation and not continuation.startswith(("ТЕМА", "TOPIC", "ТИП", "TYPE", "НАЗНАЧЕНИЕ", "SUMMARY")):
                parts[current_key] += " " + continuation

    topic = parts["ТЕМА"] or None
    doc_type = parts["ТИП"] or None
    purpose = parts["НАЗНАЧЕНИЕ"] or None

    _echo_patterns_topic = [
        r'^широкая предметная область\s*\(?1[–\-]3 слова\)?\s*[:\-–—]\s*',
        r'^одно[- ]два слова\s*[:\-–—]\s*',
        r'^предметная область\s*[:\-–—]\s*',
    ]
    _echo_patterns_type = [
        r'^конкретный вид документа\s*[:\-–—]\s*',
        r'^название вида документа\s*[:\-–—]\s*',
        r'^вид документа\s*[:\-–—]\s*',
    ]
    _echo_list_pattern = r'^(?:Аренда|Поставка|Строительство|Бухгалтерия|Кадры|Налоги|Связь|Транспорт|Электроэнергия|[Сс]чёт на оплату|[Сс]чёт-фактура|Договор аренды|Договор поставки|Договор оказания услуг|Акт выполненных работ|Акт сверки|Коммерческое предложение|Заявка|Спецификация|Доверенность|Приказ|Уведомление|Техническое задание|Паспорт|Сертификат|Отчёт|Письмо|Накладная|Универсальный передаточный документ)\s*[,\s]'

    if topic:
        for pat in _echo_patterns_topic:
            topic = re.sub(pat, '', topic, flags=re.IGNORECASE).strip()
        topic = re.sub(r'\s*и\s*т\.д\.?\s*$', '', topic, flags=re.IGNORECASE).strip()
        while re.match(_echo_list_pattern, topic):
            topic = re.sub(_echo_list_pattern, '', topic, flags=re.IGNORECASE).strip()
        topic = topic.rstrip(',').rstrip('.').strip()
    if doc_type:
        for pat in _echo_patterns_type:
            doc_type = re.sub(pat, '', doc_type, flags=re.IGNORECASE).strip()
        doc_type = re.sub(r'\s*и\s*т\.д\.?\s*$', '', doc_type, flags=re.IGNORECASE).strip()
        while re.match(_echo_list_pattern, doc_type):
            doc_type = re.sub(_echo_list_pattern, '', doc_type, flags=re.IGNORECASE).strip()
        doc_type = doc_type.rstrip(',').rstrip('.').strip()

    enriched = original_text
    if topic:
        enriched = f"[Тема: {topic}] "
    if doc_type:
        enriched += f"[Тип: {doc_type}] "
    if purpose:
        enriched += f"[Назначение: {purpose}]"

    if not topic and not doc_type:
        logger.warning(f"No tags parsed from LLM response. Raw (first 300 chars): {llm_output[:300]}")
        return llm_output + "\n" + original_text, None, None, purpose

    return enriched, topic, doc_type, purpose


def _get_db():
    from database import get_db
    try:
        return get_db()
    except Exception as e:
        logger.warning(f"DB not available for summaries: {e}")
        return None


def _load_summarized_paths_from_db() -> Set[str]:
    db = _get_db()
    if db is None:
        return set()
    try:
        return db.get_summarized_paths()
    except Exception as e:
        logger.warning(f"Failed to load summarized paths from DB: {e}")
        return set()


def _apply_db_summaries(file_data: List[Dict], summaries: Dict[str, Dict]) -> int:
    applied = 0
    for d in file_data:
        src = d.get("source", "")
        if src in summaries:
            s = summaries[src]
            if s.get("enriched_text"):
                d["text"] = s["enriched_text"]
            if s.get("topic"):
                d["topic"] = s["topic"]
            if s.get("doc_type"):
                d["doc_type"] = s["doc_type"]
            applied += 1
    return applied


def _save_summaries_to_db(batch_records: List[tuple]):
    if not batch_records:
        return
    db = _get_db()
    if db is None:
        return
    try:
        db.update_summary_batch(batch_records)
        logger.info(f"Saved {len(batch_records)} summaries to DB")
    except Exception as e:
        logger.error(f"Failed to save summaries to DB: {e}")


def delete_all_summaries():
    db = _get_db()
    if db is None:
        return
    try:
        db.clear_stage3()
    except Exception as e:
        logger.warning(f"Failed to clear stage 3 data: {e}")


def unload_model():
    try:
        from hf_summarizer import unload_model as _hf_unload
        _hf_unload()
    except Exception as e:
        logger.warning(f"Failed to unload HF model: {e}")
    _clear_all_gpu_memory()
    logger.info("LLM model unloaded, all GPU memory cleared.")


def check_llm_available() -> bool:
    from hf_summarizer import check_available
    return check_available()


# ===== HF (transformers) backend =====

def _summarize_texts_sync(texts: List[str]) -> List[Optional[str]]:
    from hf_summarizer import summarize
    trimmed = [_trim_text(t) for t in texts]
    return summarize(trimmed)


async def _summarize_batch_hf(file_data: List[Dict], texts: List[str],
                               indices: List[int], already: int, total: int) -> List[Dict]:
    import hf_summarizer
    from main import stop_event

    hf_summarizer.load_model()
    batch_size = cfg.HF_BATCH_SIZE
    batch_records = []
    completed = 0
    success_count = 0
    topic_count = 0
    type_count = 0

    loop = asyncio.get_running_loop()

    for batch_start in range(0, total, batch_size):
        if stop_event.is_set():
            logger.warning("Остановка HF саммаризации по запросу.")
            break

        batch_end = min(batch_start + batch_size, total)
        batch_texts = texts[batch_start:batch_end]
        batch_indices = indices[batch_start:batch_end]

        llm_results = await loop.run_in_executor(
            None, _summarize_texts_sync, batch_texts
        )

        for idx_in_batch, (doc_idx, llm_result) in enumerate(zip(batch_indices, llm_results)):
            completed += 1
            file_name = Path(file_data[doc_idx]["source"]).name
            progress.update(completed, f"Саммаризация: {file_name}")

            if llm_result is None:
                logger.info(f"[{already + completed}/{already + total}] {file_name} — skipped (no response)")
                continue

            original_text = file_data[doc_idx].get("text", "")
            enriched_text, topic, doc_type, purpose = _parse_summary(llm_result, original_text)
            if not topic and not doc_type:
                logger.warning(f"[{file_name}] No tags parsed. Raw: {llm_result[:200]}")

            file_data[doc_idx]["text"] = enriched_text
            if topic:
                file_data[doc_idx]["topic"] = topic
                topic_count += 1
            if doc_type:
                file_data[doc_idx]["doc_type"] = doc_type
                type_count += 1
            success_count += 1

            batch_records.append((
                str(file_data[doc_idx]["source"]),
                (enriched_text or "")[:8000],
                topic or "",
                doc_type or "",
                purpose or "",
            ))

            label_parts = []
            if topic:
                label_parts.append(f"Т={topic}")
            if doc_type:
                label_parts.append(f"Тип={doc_type}")
            label = ", ".join(label_parts) if label_parts else "no tags"
            logger.info(f"[{already + completed}/{already + total}] {file_name} — {label}")

            if len(batch_records) >= 10:
                _save_summaries_to_db(batch_records)
                batch_records = []

        if batch_end < total:
            try:
                import torch
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except ImportError:
                pass

    if batch_records:
        _save_summaries_to_db(batch_records)

    logger.info(
        f"HF summarization complete: {success_count}/{total} new processed. "
        f"Topics: {topic_count}, Types: {type_count}. "
        f"Total in DB: {already + success_count}."
    )
    return file_data


async def summarize_batch(file_data: List[Dict]) -> List[Dict]:
    if not cfg.LLM_ENABLED:
        logger.info("LLM summarization is DISABLED. Using original text.")
        return file_data

    logger.info(f"Phase 3: LLM Summarization of {len(file_data)} documents...")

    summarized_paths = _load_summarized_paths_from_db()
    if summarized_paths:
        summaries = {}
        db = _get_db()
        if db:
            try:
                file_paths = [d.get("source", "") for d in file_data]
                summaries = db.load_summaries(file_paths)
            except Exception as e:
                logger.warning(f"Failed to load summaries content from DB: {e}")

        applied = _apply_db_summaries(file_data, summaries)
        logger.info(f"Loaded {len(summarized_paths)} summaries from DB, applied {applied} to current data.")

    texts_to_summarize = []
    indices = []

    for i, d in enumerate(file_data):
        src = d.get("source", "")
        if src in summarized_paths:
            continue
        text = d.get("text")
        if text and len(text.strip()) >= cfg.LLM_MIN_TEXT_LENGTH:
            texts_to_summarize.append(text)
            indices.append(i)

    if not texts_to_summarize:
        logger.info("All documents already summarized or no texts meet the threshold.")
        return file_data

    total = len(texts_to_summarize)
    already = len(summarized_paths)

    logger.info(f"Using HF transformers backend ({cfg.HF_MODEL_PATH})")
    logger.info(f"Need to summarize: {total} documents ({already} already in DB).")
    return await _summarize_batch_hf(file_data, texts_to_summarize, indices, already, total)
