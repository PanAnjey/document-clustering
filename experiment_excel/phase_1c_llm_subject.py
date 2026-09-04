# phase_1c_llm_subject.py
# LLM-извлечение ПРЕДМЕТ/ТИП для train-набора (Qwen3.5-4B, cuda:0).
#
# Замена regex-пути: документ читает LLM (модель из hf_summarizer), но с
# СОБСТВЕННЫМ промптом: ПРЕДМЕТ (направление деятельности) НЕ должен
# дублировать ТИП документа — иначе модель отвечает «ТЕМА: Счёт на оплату»
# и кластеризация снова вырождается в типы документов.
# Результат — в temp_llm_subject.
#
# Возобновляемый: уже обработанные train_id пропускаются.
#
# Запуск:
#   python phase_1c_llm_subject.py --limit 2000   # валидационная подвыборка
#   python phase_1c_llm_subject.py                # полный train (~несколько часов)

import argparse
import re
import sys
import time
import traceback

import psycopg2
from psycopg2.extras import execute_values

from experiment_config import DB_URL  # noqa: sys.path+utf8

BATCH = 8  # 16 → CUDA OOM на батчах с длинными текстами (3000 симв.)

SYSTEM_PROMPT = """\
Извлеки из текста документа два поля: ПРЕДМЕТ и ТИП.
ПРЕДМЕТ — направление деятельности / предметная область: ЧТО поставляется, какие работы или услуги выполняются (например: монтаж оборудования, аренда помещений, электроснабжение, закупка ТМЦ, услуги связи, логистика, строительство).
ТИП — вид документа (счёт, акт, УПД, накладная, реестр и т.п.).
ПРЕДМЕТ не должен дублировать ТИП: для счёта на электроэнергию предмет — «Электроснабжение», а не «Счёт».
Каждое поле на отдельной строке. Без пояснений."""

_USER_TMPL = """\
Пример:
Текст документа: Счёт № 45 от 12.03.2026 на оплату электроэнергии за февраль, поставщик ООО Энергосбыт, сумма 154 320 руб.
Ответ:
ПРЕДМЕТ: Электроснабжение
ТИП: Счёт на оплату

Пример:
Текст документа: Акт выполненных работ по монтажу оборудования связи на объекте г. Казань, ул. Ленина 5, подрядчик ООО МонтажСервис.
Ответ:
ПРЕДМЕТ: Монтаж оборудования связи
ТИП: Акт выполненных работ

Теперь извлеки:
Текст документа: {text}
Ответ:
ПРЕДМЕТ:"""

_PREDMET_RX = re.compile(r'ПРЕДМЕТ\s*[:\-–—]\s*(.+)', re.IGNORECASE)
_TIP_RX = re.compile(r'ТИП\s*[:\-–—]\s*(.+)', re.IGNORECASE)


def _make_prompts(texts, tokenizer):
    """Промпт ПРЕДМЕТ/ТИП через chat-шаблон (enable_thinking=False)."""
    prompts = []
    for t in texts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _USER_TMPL.format(text=t)},
        ]
        try:
            p = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            p = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        prompts.append(p)
    return prompts


def _parse_predmet(raw):
    """(predmet, tip) из ответа LLM. Парсинг структурированного вывода —
    это не классификация regex'ом, а разбор формата ПОЛЕ: значение."""
    if not raw:
        return None, None
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL)
    m = _PREDMET_RX.search(raw)
    predmet = m.group(1).strip().split('\n')[0].strip() if m else None
    m = _TIP_RX.search(raw)
    tip = m.group(1).strip().split('\n')[0].strip() if m else None
    return predmet or None, tip or None


def ensure_table(conn, table):
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table} (
                train_id INTEGER PRIMARY KEY,
                predmet TEXT,
                tip TEXT,
                raw TEXT
            )
        """)
    conn.commit()


def fetch_pending(conn, source_table, target_table, limit=0, shard=0, shards=1):
    """Документы из source_table, которых ещё нет в target_table.
    Детерминированный шард по id (для параллельных LLM-процессов:
    mod(id, shards) = shard — пересечений нет, гонки нет)."""
    with conn.cursor() as cur:
        q = f"""
            SELECT t.id, t.txt
            FROM {source_table} t
            LEFT JOIN {target_table} s ON s.train_id = t.id
            WHERE s.train_id IS NULL
              AND mod(t.id, {int(shards)}) = {int(shard)}
            ORDER BY t.id
        """
        if limit and limit > 0:
            q += f" LIMIT {int(limit)}"
        cur.execute(q)
        return cur.fetchall()


def store_batch(conn, table, rows):
    with conn.cursor() as cur:
        execute_values(cur,
            f"""INSERT INTO {table} (train_id, predmet, tip, raw)
               VALUES %s
               ON CONFLICT (train_id) DO NOTHING""",
            rows, page_size=100)
    conn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0,
                    help='обработать только N документов (валидация)')
    ap.add_argument('--test', action='store_true',
                    help='обрабатывать TEST-набор (temp_excel_test → temp_llm_subject_test)')
    ap.add_argument('--source-table', default='',
                    help='переопределить исходную таблицу (id, txt)')
    ap.add_argument('--target-table', default='',
                    help='переопределить целевую таблицу результатов')
    ap.add_argument('--gpu', default='',
                    help='GPU устройство (cuda:0/cuda:1), переопределяет cfg.LLM_GPU_DEVICE')
    ap.add_argument('--shard', type=int, default=0,
                    help='номер шарда для параллельных процессов (mod(id, shards))')
    ap.add_argument('--shards', type=int, default=1,
                    help='всего шардов')
    args = ap.parse_args()

    if args.source_table:
        source_table = args.source_table
        target_table = args.target_table or 'temp_llm_subject'
    else:
        source_table = 'temp_excel_test' if args.test else 'temp_excel_train'
        target_table = 'temp_llm_subject_test' if args.test else 'temp_llm_subject'

    conn = psycopg2.connect(DB_URL)
    ensure_table(conn, target_table)
    rows = fetch_pending(conn, source_table, target_table, args.limit,
                         shard=args.shard, shards=args.shards)
    if not rows:
        print(f'✅ Нечего обрабатывать — {target_table} уже полон (шард {args.shard}/{args.shards}).')
        conn.close()
        return
    print(f'К обработке: {len(rows)} документов из {source_table} '
          f'(batch={BATCH}, шард {args.shard}/{args.shards}, gpu={args.gpu or "cfg"})')

    import torch
    import hf_summarizer
    from config import cfg

    if args.gpu:
        cfg.LLM_GPU_DEVICE = args.gpu  # шард-процесс на своей GPU

    hf_summarizer.load_model()
    tokenizer = hf_summarizer._tokenizer_instance
    model = hf_summarizer._model_instance

    t0 = time.time()
    done = 0
    try:
        for start in range(0, len(rows), BATCH):
            chunk = rows[start:start + BATCH]
            ids = [r[0] for r in chunk]
            texts = [r[1] or '' for r in chunk]
            prompts = _make_prompts(texts, tokenizer)
            inputs = tokenizer(
                prompts, return_tensors="pt", padding=True,
                truncation=True, max_length=cfg.LLM_MAX_CONTEXT_TOKENS,
            ).to(cfg.LLM_GPU_DEVICE)
            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=80,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            out = []
            for i, tid in enumerate(ids):
                gen = out_ids[i][inputs.input_ids.shape[1]:]
                raw = tokenizer.decode(gen, skip_special_tokens=True).strip()
                predmet, tip = _parse_predmet(raw)
                out.append((int(tid), predmet, tip, raw[:500] if raw else None))
            store_batch(conn, target_table, out)
            done += len(chunk)
            dt = time.time() - t0
            rate = done / dt if dt > 0 else 0
            eta = (len(rows) - done) / rate if rate > 0 else 0
            print(f'\r  LLM: {done}/{len(rows)} ({rate:.2f} д/с, ETA {eta / 60:.0f} мин)',
                  end='', flush=True)
    finally:
        print()
        try:
            hf_summarizer.unload_model()
        except Exception:
            pass
        conn.close()
    dt = time.time() - t0
    print(f'✅ Phase 1c: {done} документов за {dt / 60:.1f} мин '
          f'({done / dt:.2f} д/с)')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
