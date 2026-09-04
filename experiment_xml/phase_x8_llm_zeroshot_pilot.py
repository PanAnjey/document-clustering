# phase_x8_llm_zeroshot_pilot.py
# Пилот: LLM zero-shot классификация остаточного шума дрейф-буфера
# (13,008 XML-документов, не покрытых ни 26 базовыми, ни 30 дрейф-темами
# при cos-пороге 0.8) — Qwen3.5-4B получает КАТАЛОГ из 56 тем (топ-слова)
# и относит документ к одной из них или отвечает NONE (действительно новая тема).
#
# Цель пилота: проверить, годится ли LLM-классификация по каталогу тем как
# альтернатива/дополнение centroid-присвоению для документов, которые
# эмбеддинги не смогли уверенно разместить.
#
# Запуск:
#   python phase_x8_llm_zeroshot_pilot.py [--n 500] [--seed 42]

import argparse
import json
import re
import sys
import time
from pathlib import Path

import psycopg2
import torch
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

from experiment_xml_config import DB_URL

MODEL_PATH = r"D:\MODELS\Transformers\Qwen_Qwen3.5-4B"
OUT_DIR = Path(__file__).resolve().parent / "reports"
OUT_CSV = OUT_DIR / "x8_llm_zeroshot_pilot.csv"
OUT_MD = OUT_DIR / "x8_llm_zeroshot_pilot.md"

BATCH_SIZE = 16
MAX_NEW_TOKENS = 12
TEXT_CHARS = 800  # subject_text короткий (медиана ~165 симв.)
MAX_LENGTH = 4000

# Человекочитаемые названия 26 базовых тем (make_word_report.py::THEME_NAMES)
BASE_NAMES = {
    0: 'размещение линий связи на опорах ЛЭП',
    1: 'уборка помещений и территории',
    2: 'выдача материальных ценностей (ТМЦ)',
    3: 'аварийно-восстановительные работы',
    4: 'дополнительные работы на объектах',
    5: 'логистика: приёмка и хранение грузов',
    7: 'аренда помещений',
    8: 'приёмка-передача оборудования',
    9: 'электроснабжение',
    10: 'услуги связи (в т.ч. сотовой)',
    11: 'поставка и монтаж оборудования, материалы',
    12: 'акты сверки',
    13: 'аренда порта Metro Ethernet',
    15: 'регистрация и изменения юрлица',
    16: 'замена SIM-карт',
    17: 'доступ к инфраструктуре, интернет',
    19: 'поставка шин, неустойки',
    20: 'цифровые каналы связи (Ethernet/EPL)',
    21: 'реклама на телевидении и в СМИ',
    22: 'выписки из ЕГРН',
    23: 'PR и рекламное размещение',
    24: 'оплата услуг HeadHunter',
    25: 'закупка ТМЦ, мебели, аккумуляторов',
    26: 'вознаграждение по договору (агентское)',
}

# Человекочитаемые названия крупных дрейф-тем (AGENTS.md, сессия x5); для
# остальных используем топ-слова из БД как есть.
DRIFT_NAMES = {
    5: 'модернизация/строительство БС',
    6: 'выполненные работы на сети (ПАО Вымпелком)',
    9: 'выручка от аренды',
    11: 'авиабилеты',
    12: 'возмещение электроэнергии',
    14: 'комплект исполнительной документации',
    23: 'SFP-модули',
    25: 'оборудование Huawei (складские акты)',
    27: 'оборудование Huawei/Ericsson LTE/UMTS',
    28: 'антенны',
}


def build_catalog(conn):
    """Возвращает [(code, description)] — 26 базовых ('B{n}') + 30 дрейф ('D{n}')."""
    catalog = []
    with conn.cursor() as cur:
        cur.execute("SELECT theme_label, top_words FROM temp_theme2_centroids_v2 ORDER BY theme_label")
        for label, top_words in cur.fetchall():
            words = ', '.join((top_words or '').split(', ')[:5])
            name = BASE_NAMES.get(label)
            desc = f"{name} ({words})" if name else words
            catalog.append((f"B{label}", desc))

        cur.execute("SELECT new_theme, top_words FROM temp_xml_drift_centroids ORDER BY new_theme")
        for label, top_words in cur.fetchall():
            words = ', '.join((top_words or '').split(', ')[:6])
            name = DRIFT_NAMES.get(label)
            desc = f"{name} ({words})" if name else words
            catalog.append((f"D{label}", desc))
    return catalog


def fetch_sample(conn, n, seed):
    with conn.cursor() as cur:
        cur.execute("SELECT setseed(%s)", (seed,))
        cur.execute(f"""
            SELECT a.xml_id, c.subject_text, c.type_named_id
            FROM temp_xml_subject_clean_assign a
            JOIN temp_xml_canonical c ON c.id = a.xml_id
            WHERE a.is_unknown
              AND a.xml_id NOT IN (SELECT xml_id FROM temp_xml_drift_assign)
              AND c.subject_text IS NOT NULL AND c.subject_text != ''
            ORDER BY random()
            LIMIT %s
        """, (n,))
        return cur.fetchall()


def make_system_prompt(catalog):
    lines = [f"- {code}: {desc}" for code, desc in catalog]
    return (
        "Ты — классификатор тем корпоративных документов оператора связи "
        "(счета-фактуры, УПД, акты). Ниже список из 56 известных тем (код и "
        "ключевые слова). Определи, к какой теме относится документ по его "
        "тексту.\n\n"
        "Известные темы:\n" + "\n".join(lines) + "\n\n"
        "Правила:\n"
        "1. Если документ явно соответствует одной из тем — ответь ЕЁ кодом "
        "(например B10 или D5).\n"
        "2. Если документ не подходит уверенно ни под одну тему — ответь NONE.\n"
        "3. Ответь ТОЛЬКО кодом темы или NONE, без пояснений."
    )


def make_prompt(tok, sys_prompt, subject_text, doc_type):
    t = (subject_text or "").replace("\n", " ")[:TEXT_CHARS]
    user = f"Тип документа: {doc_type}\nТекст (предмет): {t}\n\nТЕМА:"
    msgs = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user},
    ]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def parse_code(ans, valid_codes):
    if not ans:
        return "PARSE_FAIL"
    ans = ans.strip().upper()
    if ans.startswith("NONE"):
        return "NONE"
    m = re.search(r'\b([BD]\d{1,2})\b', ans)
    if m and m.group(1) in valid_codes:
        return m.group(1)
    return "PARSE_FAIL"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--seed', type=float, default=0.42, help='postgres setseed() value, [-1, 1]')
    args = ap.parse_args()

    conn = psycopg2.connect(DB_URL)
    catalog = build_catalog(conn)
    valid_codes = {c for c, _ in catalog}
    print(f"каталог тем: {len(catalog)} ({sum(1 for c,_ in catalog if c.startswith('B'))} базовых + "
          f"{sum(1 for c,_ in catalog if c.startswith('D'))} дрейф)")

    rows = fetch_sample(conn, args.n, args.seed)
    conn.close()
    print(f"выборка остаточного шума: {len(rows)} документов")
    if not rows:
        sys.exit("пустая выборка")

    sys_prompt = make_system_prompt(catalog)
    print(f"system prompt: {len(sys_prompt)} символов")

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    )
    model.eval()
    print(f"модель загружена, VRAM={torch.cuda.memory_allocated(0)/1e9:.2f}GB")

    OUT_DIR.mkdir(exist_ok=True)
    fout = open(OUT_CSV, "w", encoding="utf-8-sig", newline="")
    import csv
    writer = csv.DictWriter(fout, fieldnames=["xml_id", "type_named_id", "subject_text", "raw", "code"])
    writer.writeheader()

    freq = {}
    t0 = time.time()
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        prompts = [make_prompt(tok, sys_prompt, r[1], r[2]) for r in batch]
        inp = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=MAX_LENGTH).to(next(model.parameters()).device)
        with torch.no_grad():
            out = model.generate(
                **inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
            )
        for j, (xml_id, subj, dtype) in enumerate(batch):
            gen = out[j][inp.input_ids.shape[1]:]
            ans = tok.decode(gen, skip_special_tokens=True).strip()
            code = parse_code(ans, valid_codes)
            freq[code] = freq.get(code, 0) + 1
            writer.writerow({
                "xml_id": xml_id, "type_named_id": dtype,
                "subject_text": (subj or "")[:200], "raw": ans[:80], "code": code,
            })
        fout.flush()
        done = min(i + BATCH_SIZE, len(rows))
        if done % (BATCH_SIZE * 5) < BATCH_SIZE:
            el = time.time() - t0
            rate = done / el if el else 0
            print(f"{done}/{len(rows)}  elapsed={el:.0f}s  rate={rate:.1f}/s", flush=True)
    fout.close()

    n_total = len(rows)
    n_none = freq.get("NONE", 0)
    n_fail = freq.get("PARSE_FAIL", 0)
    n_assigned = n_total - n_none - n_fail
    print(f"\nитого: {n_total}  назначено={n_assigned} ({n_assigned/n_total:.1%})  "
          f"NONE={n_none} ({n_none/n_total:.1%})  parse_fail={n_fail} ({n_fail/n_total:.1%})")

    top = sorted(freq.items(), key=lambda x: -x[1])[:15]
    name_map = dict(catalog)
    lines = [
        "# X8: LLM zero-shot пилот на остаточном шуме дрейф-буфера",
        "",
        f"Выборка: {n_total} документов (случайная, seed={args.seed}) из 13,008 "
        "остаточного шума (не покрыты ни 26 базовыми, ни 30 дрейф-темами при cos>=0.8).",
        f"Каталог: {len(catalog)} тем (26 базовых + 30 дрейф), модель Qwen3.5-4B, "
        f"zero-shot (без примеров), greedy decoding.",
        "",
        "## Итог",
        "",
        f"- Назначено теме из каталога: {n_assigned} ({n_assigned/n_total:.1%})",
        f"- NONE (LLM считает — реально новая тема): {n_none} ({n_none/n_total:.1%})",
        f"- Не удалось распарсить ответ: {n_fail} ({n_fail/n_total:.1%})",
        "",
        "## Топ-15 предсказанных кодов",
        "",
        "| Код | Название | N | % |",
        "|---|---|---|---|",
    ]
    for code, n in top:
        label = name_map.get(code, code)
        lines.append(f"| {code} | {label} | {n} | {n/n_total:.1%} |")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")

    print(f"written {OUT_CSV}")
    print(f"written {OUT_MD}")


if __name__ == "__main__":
    main()
