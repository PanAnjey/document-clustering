"""LLM-классификация всех PDF из PDF_Tables\\fin.

Без фиксированного набора типов: модель возвращает произвольное короткое
название типа. Результаты пишутся в CSV инкрементно (для возобновления).
"""

from __future__ import annotations

import csv
import gc
import os
import sys
import time
import io
from pathlib import Path
from typing import List

import fitz
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

MODEL_PATH = r"D:\MODELS\Transformers\Qwen_Qwen3.5-4B"
DEVICE = "cuda:0"
SRC_DIR = Path(r"D:\FileOrganizer\Sorted\PDF_Tables\fin")
OUT_CSV = Path(r"D:\Yandex.Disk\PYTHON\NLTK\Кластеризация\experiments\fin_tables\work\llm_classification_all.csv")
FREQ_TXT = Path(r"D:\Yandex.Disk\PYTHON\NLTK\Кластеризация\experiments\fin_tables\work\llm_classification_freq.txt")

BATCH_SIZE = 16
MAX_NEW_TOKENS = 12
TEXT_CHARS = 3500
MAX_FILES = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # 0 = все

SYS = (
    "Определи тип финансового документа по тексту его первой страницы. "
    "Ответь коротким названием типа (1–3 слова), без пояснений, без пунктуации. "
    "Например: счет-фактура, упд, счет, акт, акт сверки, отчет агента, "
    "приложение, детализация, смета, расшифровка, акт выполненных работ."
)

tok = None
model = None


def ensure_ascii_safe(s: str) -> str:
    return s.encode("utf-8", "replace").decode("utf-8")


def load_model():
    global tok, model
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map=DEVICE, trust_remote_code=True
    )
    model.eval()
    alloc = torch.cuda.memory_allocated() / 1e9
    print(f"model loaded, VRAM {alloc:.2f}GB", flush=True)


def first_page_text(p: Path) -> str:
    try:
        d = fitz.open(str(p))
        try:
            return d[0].get_text("text") if d.page_count else ""
        finally:
            d.close()
    except Exception:
        return ""


def make_prompt(text: str) -> str:
    t = (text or "").replace("\n", " ")[:TEXT_CHARS]
    msgs = [
        {"role": "system", "content": SYS},
        {"role": "user", "content": f"Текст первой страницы:\n{t}\n\nТИП:"},
    ]
    try:
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def classify_batch(texts: List[str]) -> List[str]:
    prompts = [make_prompt(t) for t in texts]
    inp = tok(
        prompts, return_tensors="pt", padding=True, truncation=True, max_length=8000
    ).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inp,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    res = []
    for i in range(len(texts)):
        gen = out[i][inp.input_ids.shape[1]:]
        ans = tok.decode(gen, skip_special_tokens=True).strip()
        ans = ans.splitlines()[0].strip().lower() if ans else ""
        # нормализуем
        ans = ans.strip(".-— ").strip()
        res.append(ans[:60])
    return res


# def normalize_type(raw: str) -> str:
#     r = raw.lower().strip()
#     match r:
#         case _ if ("счет-фактура" in r or "счет фактура" in r or "счёт-фактура" in r
#                    or r.startswith("фактура") or r == "фактура"):
#             return "счет-фактура"
#         case _ if "упд" in r or "универсальный передаточный" in r:
#             return "упд"
#         case _ if "акт сверки" in r:
#             return "акт сверки"
#         case _ if r.startswith("акт"):
#             return "акт"
#         case _ if "счет на оплату" in r or r.startswith("счет") or r.startswith("счёт"):
#             return "счет"
#         case _ if "платежное поручение" in r or "платёжное поручение" in r:
#             return "платежное поручение"
#         case _ if "отчет" in r or "отчёт" in r:
#             return "отчет"
#         case _ if "приложение" in r:
#             return "приложение"
#         case _ if "детализация" in r:
#             return "детализация"
#         case _ if "смета" in r:
#             return "смета"
#         case _ if "расшифровка" in r:
#             return "расшифровка"
#         case _ if "ведомость" in r:
#             return "ведомость"
#         case _ if "выписка" in r:
#             return "выписка из лицевого счета"
#         case _ if "справка" in r:
#             return "справка"
#         case _ if "соглашение" in r:
#             return "соглашение"
#         case _ if "расчет" in r or "расчёт" in r:
#             return "расчет"
#         case _ if "договор" in r:
#             return "договор"
#         case _ if "накладн" in r:
#             return "накладная"
#         case _ if "заказ" in r:
#             return "заказ"
#         case _ if "квитанция" in r:
#             return "квитанция"
#         case _ if "доверенность" in r:
#             return "доверенность"
#         case _ if "уведомление" in r:
#             return "уведомление"
#         case _ if "реестр" in r:
#             return "реестр"
#         case _ if "анкета" in r:
#             return "анкета"
#         case _ if "оферта" in r:
#             return "оферта"
#         case _ if not r:
#             return "прочее"
#         case _:
#             return r


def existing_files() -> set:
    if not OUT_CSV.exists():
        return set()
    s = set()
    with open(OUT_CSV, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            s.add(row["file"])
    return s


def main():
    files = sorted([SRC_DIR / f for f in os.listdir(SRC_DIR) if f.lower().endswith(".pdf")])
    if MAX_FILES > 0:
        files = files[:MAX_FILES]
    print(f"total files: {len(files)}", flush=True)

    done = existing_files()
    print(f"already done: {len(done)}", flush=True)

    load_model()

    out_exists = OUT_CSV.exists()
    fout = open(OUT_CSV, "a", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(fout, fieldnames=["file", "raw", "doc_type"])
    if not out_exists:
        writer.writeheader()

    freq: dict = {}
    t0 = time.time()
    processed = 0
    buf: List[Path] = []
    texts: List[str] = []

    def flush_buf():
        nonlocal processed
        if not buf:
            return
        results = classify_batch(texts)
        for p, raw in zip(buf, results):
            # dt = normalize_type(raw)
            # writer.writerow({"file": p.name, "raw": raw, "doc_type": dt})
            writer.writerow({"file": p.name, "raw": raw})
            freq[raw] = freq.get(raw, 0) + 1
        fout.flush()
        processed += len(buf)
        buf.clear()
        texts.clear()

    pending = [p for p in files if p.name not in done]
    print(f"pending: {len(pending)}", flush=True)

    for i, p in enumerate(pending, 1):
        txt = first_page_text(p)
        buf.append(p)
        texts.append(txt)
        if len(buf) >= BATCH_SIZE:
            flush_buf()
            if processed % 100 < BATCH_SIZE:
                el = time.time() - t0
                rate = processed / el if el else 0
                eta = (len(pending) - processed) / rate if rate else 0
                print(
                    f"{processed}/{len(pending)}  elapsed={el:.0f}s  rate={rate:.1f}/s  eta={eta/60:.1f}min",
                    flush=True,
                )
        if processed > 0 and processed % 1000 == 0 and len(buf) == 0:
            # periodically dump frequency
            with open(FREQ_TXT, "w", encoding="utf-8") as f:
                f.write(f"processed: {processed}\n\n")
                for k, v in sorted(freq.items(), key=lambda x: -x[1]):
                    f.write(f"{k}\t{v}\n")

    flush_buf()
    fout.close()

    with open(FREQ_TXT, "w", encoding="utf-8") as f:
        f.write(f"total processed: {processed}\n\n")
        for k, v in sorted(freq.items(), key=lambda x: -x[1]):
            f.write(f"{k}\t{v}\n")
    print("done", flush=True)


if __name__ == "__main__":
    main()