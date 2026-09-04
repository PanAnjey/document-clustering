# phase_5v_validate_contract.py
# Валидация 4-полевого LLM-промпта (ПРЕДМЕТ/ТИП/ОСНОВАНИЕ/КОНТРАГЕНТ)
# на 1000 случайных документах PDF_Tables — ТОЛЬКО фин.первичка
# (счёт/УПД/сч-ф/накладная/акты/КС-2/КС-3).
#
# Метрики качества: полнота ОСНОВАНИЕ/ИНН/КПП/региона, валидность ИНН-10/12,
# ИНН-есть-в-тексте (анти-галлюцинация), self-фильтр (ВымпелКом).
#
# Таблица: temp_contract_validate (pdftext_id PK, predmet, tip, osnovanie,
#   kp_name, kp_inn, kp_kpp, kp_region, raw)

import re
import sys
import time
import traceback

import psycopg2
from psycopg2.extras import execute_values

from experiment_config import DB_URL  # noqa: sys.path+utf8

BATCH = 8
LIMIT = 1000

SYSTEM_PROMPT = """\
Извлеки из текста документа четыре поля: ПРЕДМЕТ, ТИП, ОСНОВАНИЕ, КОНТРАГЕНТ.
ПРЕДМЕТ — направление деятельности: ЧТО поставляется, какие работы или услуги выполняются. Не дублирует ТИП.
ТИП — вид документа (счёт, акт, УПД, накладная, реестр и т.п.).
ОСНОВАНИЕ — реквизиты документа-основания (договора): номер и дата, приложение/доп.соглашение если указано. Пусто, если не указано.
КОНТРАГЕНТ — внешняя сторона договора (НЕ ПАО «ВымпелКом»): название | ИНН | КПП | регион (город). Пустые части оставляй пустыми.
Каждое поле на отдельной строке. Без пояснений."""

_USER_TMPL = """\
Пример:
Текст документа: Счёт № 45 от 12.03.2026 на оплату электроэнергии за февраль, поставщик ООО Энергосбыт, ИНН 7707083893, КПП 770701001, г. Москва, по договору № 12-Э от 01.02.2020
Ответ:
ПРЕДМЕТ: Электроснабжение
ТИП: Счёт на оплату
ОСНОВАНИЕ: Договор № 12-Э от 01.02.2020
КОНТРАГЕНТ: ООО Энергосбыт | ИНН 7707083893 | КПП 770701001 | Москва

Пример:
Текст документа: Акт выполненных работ по монтажу оборудования связи на объекте г. Казань, ул. Ленина 5, подрядчик ООО МонтажСервис
Ответ:
ПРЕДМЕТ: Монтаж оборудования связи
ТИП: Акт выполненных работ
ОСНОВАНИЕ:
КОНТРАГЕНТ: ООО МонтажСервис |  |  | Казань

Теперь извлеки:
Текст документа: {text}
Ответ:
ПРЕДМЕТ:"""

_FIELDS = ('ПРЕДМЕТ', 'ТИП', 'ОСНОВАНИЕ', 'КОНТРАГЕНТ')
# [ \t]*, а НЕ \s* — иначе пустое поле захватывает следующее (перевод строки)
_FIELD_RX = {f: re.compile(rf'{f}[ \t]*[:\-–—][ \t]*(.*)', re.IGNORECASE) for f in _FIELDS}
INN_WEIGHTS_10 = [2, 4, 10, 3, 5, 9, 4, 6, 8]
INN_W12_D11 = [7, 2, 4, 10, 3, 5, 9, 4, 6, 8]   # ИНН-12, контрольная 11-я цифра
INN_W12_D12 = [3, 7, 2, 4, 10, 3, 5, 9, 4, 6, 8]  # ИНН-12, контрольная 12-я цифра
# «свои» компании — не контрагент (LLM иногда захватывает двусторонние документы)
_SELF_RX = re.compile(r'(вымпел|vimpel|билайн|beeline)', re.IGNORECASE)
# Финансовая первичка + акты/КС (фильтр документов)
_DOC_FILTER_SQL = """
    lower(s.tip) LIKE '%счет%' OR lower(s.tip) LIKE '%счёт%'
    OR lower(s.tip) LIKE '%фактура%' OR lower(s.tip) LIKE '%упд%'
    OR lower(s.tip) LIKE '%накладн%' OR lower(s.tip) LIKE '%акт%'
    OR lower(s.tip) LIKE '%кс-2%' OR lower(s.tip) LIKE '%кс-3%'
    OR lower(s.tip) LIKE '%кс 2%' OR lower(s.tip) LIKE '%кс 3%'
"""


def inn_valid_10(inn: str) -> bool:
    """Контрольная цифра ИНН-10 (юрлицо)."""
    if not re.fullmatch(r'\d{10}', inn or ''):
        return False
    s = sum(int(d) * w for d, w in zip(inn[:9], INN_WEIGHTS_10))
    return s % 11 % 10 == int(inn[9])


def inn_valid_12(inn: str) -> bool:
    """Контрольные цифры ИНН-12 (физлицо/ИП)."""
    if not re.fullmatch(r'\d{12}', inn or ''):
        return False
    s11 = sum(int(d) * w for d, w in zip(inn[:10], INN_W12_D11))
    if s11 % 11 % 10 != int(inn[10]):
        return False
    s12 = sum(int(d) * w for d, w in zip(inn[:11], INN_W12_D12))
    return s12 % 11 % 10 == int(inn[11])


def inn_valid(inn: str) -> bool:
    return inn_valid_10(inn) or inn_valid_12(inn)


def is_self(name) -> bool:
    return bool(name) and bool(_SELF_RX.search(name))


def parse_kp(kp_raw: str):
    """'ООО Ромашка | ИНН 7707... | КПП 7707... | Москва' → компоненты.
    ИНН 12 цифр (ИП) предпочтительнее 10 (юрлицо) — сначала длинный паттерн."""
    parts = [p.strip() for p in (kp_raw or '').split('|')]
    parts += [''] * (4 - len(parts))
    name = parts[0] or None
    inn = None
    kpp = None
    region = parts[3] or None
    if len(parts) > 1:
        m = re.search(r'\d{12}|\d{10}', parts[1] or '')
        if m:
            inn = m.group(0)
    if len(parts) > 2:
        m = re.search(r'\d{9}', parts[2] or '')
        if m:
            kpp = m.group(0)
    return name, inn, kpp, region


def _make_prompts(texts, tokenizer):
    prompts = []
    for t in texts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _USER_TMPL.format(text=t)},
        ]
        try:
            p = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            p = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
        prompts.append(p)
    return prompts


def parse_raw(raw: str):
    out = {}
    for f in _FIELDS:
        m = _FIELD_RX[f].search(raw or '')
        val = m.group(1).strip().split('\n')[0].strip() if m else ''
        out[f] = val or None
    return out


def main():
    conn = psycopg2.connect(DB_URL)
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS temp_contract_validate")
        cur.execute("""
            CREATE TABLE temp_contract_validate (
                pdftext_id INTEGER PRIMARY KEY,
                predmet TEXT, tip TEXT, osnovanie TEXT,
                kp_name TEXT, kp_inn TEXT, kp_kpp TEXT, kp_region TEXT,
                raw TEXT
            )""")
        cur.execute(f"""
            SELECT f.id, f.txt FROM temp_pdftables_files f
            JOIN temp_llm_subject_pdftables s ON s.train_id = f.id
            WHERE s.tip IS NOT NULL AND ({_DOC_FILTER_SQL})
            ORDER BY random() LIMIT {LIMIT}
        """)
        rows = cur.fetchall()
    conn.commit()
    print(f'Валидация на {len(rows)} случайных документах PDF_Tables '
          f'(фин.первичка: счёт/УПД/сч-ф/накладная/акты/КС)')

    import torch
    import hf_summarizer
    from config import cfg
    hf_summarizer.load_model()
    tokenizer = hf_summarizer._tokenizer_instance
    model = hf_summarizer._model_instance

    t0 = time.time()
    done = 0
    stats = dict(osn=0, inn=0, inn_valid=0, inn_in_text=0, kpp=0, region=0,
                 kp=0, self=0)
    try:
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]
            ids = [r[0] for r in chunk]
            texts = [r[1] or '' for r in chunk]
            prompts = _make_prompts(texts, tokenizer)
            inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                               truncation=True, max_length=cfg.LLM_MAX_CONTEXT_TOKENS
                               ).to(cfg.LLM_GPU_DEVICE)
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs, max_new_tokens=140, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id)
            out = []
            for i, tid in enumerate(ids):
                gen = out_ids[i][inputs.input_ids.shape[1]:]
                raw = tokenizer.decode(gen, skip_special_tokens=True).strip()
                f = parse_raw(raw)
                name, inn, kpp, region = parse_kp(f['КОНТРАГЕНТ'] or '')
                # self-фильтр: собственная компания — не контрагент
                if is_self(name):
                    stats['self'] += 1
                    name = None
                # верификация ИНН по исходному тексту (анти-галлюцинация)
                txt_i = texts[i].replace(' ', '')
                inn_in_text = bool(inn) and inn in txt_i
                out.append((int(tid), f['ПРЕДМЕТ'], f['ТИП'], f['ОСНОВАНИЕ'],
                            name, inn, kpp, region, raw[:600]))
                stats['osn'] += bool(f['ОСНОВАНИЕ'])
                stats['kp'] += bool(name)
                stats['inn'] += bool(inn)
                stats['inn_valid'] += bool(inn and inn_valid(inn))
                stats['inn_in_text'] += inn_in_text
                stats['kpp'] += bool(kpp)
                stats['region'] += bool(region)
            with conn.cursor() as cur:
                execute_values(cur,
                    """INSERT INTO temp_contract_validate
                       (pdftext_id, predmet, tip, osnovanie,
                        kp_name, kp_inn, kp_kpp, kp_region, raw) VALUES %s""",
                    out, page_size=100)
            conn.commit()
            done += len(chunk)
            dt = time.time() - t0
            rate = done / dt if dt else 0
            print(f'\r  LLM: {done}/{len(rows)} ({rate:.2f} д/с)', end='', flush=True)
    finally:
        print()
        hf_summarizer.unload_model()

    dt = time.time() - t0
    n = done or 1
    print(f'\n✅ {done} док. за {dt / 60:.1f} мин ({done / dt:.2f} д/с)')
    print(f'   ОСНОВАНИЕ извлечено: {stats["osn"]} ({stats["osn"] / n:.1%})')
    print(f'   КОНТРАГЕНТ (название, без self): {stats["kp"]} ({stats["kp"] / n:.1%}); '
          f'self отсеяно: {stats["self"]}')
    print(f'   ИНН: {stats["inn"]} ({stats["inn"] / n:.1%}), '
          f'валидный чексумма-10/12: {stats["inn_valid"]} '
          f'({stats["inn_valid"] / max(stats["inn"], 1):.1%}), '
          f'есть в тексте: {stats["inn_in_text"]} '
          f'({stats["inn_in_text"] / max(stats["inn"], 1):.1%})')
    print(f'   КПП: {stats["kpp"]} ({stats["kpp"] / n:.1%})')
    print(f'   Регион: {stats["region"]} ({stats["region"] / n:.1%})')
    conn.close()


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
