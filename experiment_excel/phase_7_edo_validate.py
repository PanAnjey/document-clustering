# phase_7_edo_validate.py
# Валидация извлечённых LLM данных контрагента (temp_contract_fields:
# kp_name/kp_inn/kp_kpp) по метаданным оператора ЭДО (MySQL events_26.doc_in
# + diadoc_api.contragens_info).
#
# Шаги:
#   1. Из имени файла (temp_contract_files.file_path) выделяется EntityId —
#      подстрока слева от первой точки в имени файла.
#   2. MySQL: SELECT c.Inn, c.Kpp, c.FullName, c.ShortName
#      FROM events_26.doc_in d JOIN diadoc_api.contragens_info c
#        ON d.CounteragentBoxId = c.BoxId WHERE d.EntityId IN (...).
#   3. Снимок ЭДО-данных сохраняется в PG temp_contract_edo.
#   4. Сравнение ИНН/КПП (точное, digits-only) и наименования
#      (нормализация + rapidfuzz), сводная статистика.
#
# Выход:
#   PG:  temp_contract_edo (doc_id PK, entity_id, edo_inn/kpp/fullname/shortname, found)
#   Файлы: work/contract_edo_validate_report.txt, work/contract_edo_divergences.csv
#
# Запуск: python phase_7_edo_validate.py

import csv
import re
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import psycopg2
import pymysql
from psycopg2.extras import execute_values
from rapidfuzz import fuzz

from experiment_config import DB_URL  # noqa: sys.path+utf8

MYSQL = dict(host='localhost', port=3308, user='root', password='mysql',
             charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)

WORK = Path(__file__).parent / 'work'
CHUNK = 1000

# --- Нормализация наименований -------------------------------------------

_ORG_PHRASES = [
    'ОБЩЕСТВО С ОГРАНИЧЕННОЙ ОТВЕТСТВЕННОСТЬЮ',
    'ПУБЛИЧНОЕ АКЦИОНЕРНОЕ ОБЩЕСТВО',
    'НЕПУБЛИЧНОЕ АКЦИОНЕРНОЕ ОБЩЕСТВО',
    'ЗАКРЫТОЕ АКЦИОНЕРНОЕ ОБЩЕСТВО',
    'ОТКРЫТОЕ АКЦИОНЕРНОЕ ОБЩЕСТВО',
    'АКЦИОНЕРНОЕ ОБЩЕСТВО',
    'ИНДИВИДУАЛЬНЫЙ ПРЕДПРИНИМАТЕЛЬ',
    'ФЕДЕРАЛЬНОЕ ГОСУДАРСТВЕННОЕ УНИТАРНОЕ ПРЕДПРИЯТИЕ',
    'ГОСУДАРСТВЕННОЕ УНИТАРНОЕ ПРЕДПРИЯТИЕ',
    'МУНИЦИПАЛЬНОЕ УНИТАРНОЕ ПРЕДПРИЯТИЕ',
]
_ORG_TOKENS = {'ООО', 'АО', 'ПАО', 'ЗАО', 'ОАО', 'НАО', 'ИП', 'ГУП', 'МУП',
               'ФГУП', 'ФГБУ', 'ГБУ', 'АНО', 'ФОНД', 'ТСЖ', 'СНТ', 'ПК'}


def norm_name(s: str) -> str:
    """Верхний регистр, Ё->Е, без оргформ/кавычек, только значимые токены."""
    if not s:
        return ''
    s = s.upper().replace('Ё', 'Е')
    for ph in _ORG_PHRASES:
        s = s.replace(ph, ' ')
    tokens = re.findall(r'[А-ЯA-Z0-9]+', s)
    tokens = [t for t in tokens if t not in _ORG_TOKENS]
    return ' '.join(tokens)


def cmp_digits(a: str, b: str) -> str:
    """Сравнение ИНН/КПП (только цифры)."""
    a = re.sub(r'\D', '', a or '')
    b = re.sub(r'\D', '', b or '')
    if not a and not b:
        return 'both_missing'
    if not a:
        return 'llm_missing'
    if not b:
        return 'edo_missing'
    return 'match' if a == b else 'mismatch'


def cmp_name(llm_name: str, edo_short: str, edo_full: str) -> str:
    """Сравнение наименования: нормализованное равенство/вхождение/fuzzy>=85."""
    if not (llm_name or '').strip():
        return 'llm_missing'
    if not (edo_short or '').strip() and not (edo_full or '').strip():
        return 'edo_missing'
    a = norm_name(llm_name)
    if not a:
        return 'llm_missing'
    best = 0.0
    for cand in (edo_short, edo_full):
        b = norm_name(cand or '')
        if not b:
            continue
        if a == b:
            return 'match'
        if min(len(a), len(b)) >= 4 and (a in b or b in a):
            return 'match'
        best = max(best, fuzz.token_sort_ratio(a, b))
    return 'match' if best >= 85 else 'mismatch'


# --- Сбор данных ----------------------------------------------------------

def fetch_llm_rows(pg):
    with pg.cursor() as cur:
        cur.execute("""
            SELECT f.id, f.file_path, f.source,
                   r.kp_name, r.kp_inn, r.kp_kpp, r.kp_self
            FROM temp_contract_files f
            JOIN temp_contract_fields r ON r.doc_id = f.id
            ORDER BY f.id
        """)
        return cur.fetchall()


def fetch_edo_rows(entity_ids):
    """MySQL: entity_id -> dict(inn, kpp, fullname, shortname)."""
    out = {}
    conn = pymysql.connect(**MYSQL)
    try:
        with conn.cursor() as cur:
            for i in range(0, len(entity_ids), CHUNK):
                chunk = entity_ids[i:i + CHUNK]
                ph = ','.join(['%s'] * len(chunk))
                cur.execute(f"""
                    SELECT d.EntityId AS eid, c.Inn AS inn, c.Kpp AS kpp,
                           c.FullName AS fullname, c.ShortName AS shortname
                    FROM events_26.doc_in d
                    INNER JOIN diadoc_api.contragens_info c
                        ON d.CounteragentBoxId = c.BoxId
                    WHERE d.EntityId IN ({ph})
                """, chunk)
                for row in cur.fetchall():
                    out[row['eid']] = row
                print(f'\r  MySQL: {min(i + CHUNK, len(entity_ids))}/'
                      f'{len(entity_ids)}', end='', flush=True)
        print()
    finally:
        conn.close()
    return out


def save_edo_snapshot(pg, rows):
    """rows: list of (doc_id, entity_id, found, inn, kpp, fullname, shortname)."""
    with pg.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS temp_contract_edo")
        cur.execute("""
            CREATE TABLE temp_contract_edo (
                doc_id INTEGER PRIMARY KEY,
                entity_id VARCHAR(64) NOT NULL,
                found BOOLEAN NOT NULL,
                edo_inn TEXT, edo_kpp TEXT,
                edo_fullname TEXT, edo_shortname TEXT
            )""")
        execute_values(cur, """
            INSERT INTO temp_contract_edo
            (doc_id, entity_id, found, edo_inn, edo_kpp, edo_fullname, edo_shortname)
            VALUES %s
        """, rows, page_size=1000)
    pg.commit()


# --- Анализ ---------------------------------------------------------------

def pct(n, d):
    return f'{100.0 * n / d:.1f}%' if d else '—'


def stat_block(recs, key):
    c = Counter(r[key] for r in recs)
    total = len(recs)
    return c, total


def main():
    t0 = time.time()
    pg = psycopg2.connect(DB_URL)

    print('1. Чтение LLM-данных из PG...')
    llm_rows = fetch_llm_rows(pg)
    print(f'   Документов: {len(llm_rows)}')

    print('2. Извлечение EntityId из имён файлов...')
    docs = []
    for doc_id, fpath, source, kp_name, kp_inn, kp_kpp, kp_self in llm_rows:
        fname = Path(fpath).name
        eid = fname.split('.')[0]
        docs.append(dict(doc_id=doc_id, file=fname, source=source, eid=eid,
                         llm_name=kp_name, llm_inn=kp_inn, llm_kpp=kp_kpp,
                         self=bool(kp_self)))
    uniq_eids = list({d['eid'] for d in docs})
    print(f'   Уникальных EntityId: {len(uniq_eids)} из {len(docs)}')

    print('3. Запрос метаданных ЭДО из MySQL...')
    edo = fetch_edo_rows(uniq_eids)
    print(f'   Найдено в doc_in: {len(edo)}')

    print('4. Сохранение снимка в PG temp_contract_edo...')
    snap = [(d['doc_id'], d['eid'], d['eid'] in edo,
             (edo.get(d['eid']) or {}).get('inn'),
             (edo.get(d['eid']) or {}).get('kpp'),
             (edo.get(d['eid']) or {}).get('fullname'),
             (edo.get(d['eid']) or {}).get('shortname'))
            for d in docs]
    save_edo_snapshot(pg, snap)

    print('5. Сравнение...')
    for d in docs:
        e = edo.get(d['eid'])
        d['found'] = e is not None
        if not e:
            d['edo_inn'] = d['edo_kpp'] = d['edo_short'] = d['edo_full'] = None
            d['inn_st'] = d['kpp_st'] = d['name_st'] = 'no_edo'
            continue
        d['edo_inn'] = e['inn']
        d['edo_kpp'] = e['kpp']
        d['edo_short'] = e['shortname']
        d['edo_full'] = e['fullname']
        d['inn_st'] = cmp_digits(d['llm_inn'], e['inn'])
        d['kpp_st'] = cmp_digits(d['llm_kpp'], e['kpp'])
        d['name_st'] = cmp_name(d['llm_name'], e['shortname'], e['fullname'])

    # --- Отчёт ---
    rep = []
    found = [d for d in docs if d['found']]
    not_found = [d for d in docs if not d['found']]
    rep.append('=' * 72)
    rep.append('ВАЛИДАЦИЯ КОНТРАГЕНТА: LLM-экстракция vs метаданные ЭДО (Диадок)')
    rep.append('=' * 72)
    rep.append(f'Всего документов прогона : {len(docs)}')
    rep.append(f'Найдено в events_26.doc_in: {len(found)} ({pct(len(found), len(docs))})')
    rep.append(f'Не найдено                : {len(not_found)}')

    def section(title, recs):
        rep.append('')
        rep.append(f'--- {title} (n={len(recs)}) ---')
        for key, label in (('inn_st', 'ИНН '), ('kpp_st', 'КПП '), ('name_st', 'ИМЯ ')):
            c, total = stat_block(recs, key)
            parts = []
            for st in ('match', 'mismatch', 'llm_missing', 'edo_missing',
                       'both_missing'):
                if c.get(st):
                    parts.append(f'{st}={c[st]} ({pct(c[st], total)})')
            rep.append(f'  {label}: ' + '; '.join(parts))

    nonself = [d for d in found if not d['self']]
    selfd = [d for d in found if d['self']]
    section('Все найденные документы', found)
    section('Без self (основной контур)', nonself)
    section('Self (LLM взяла нашу сторону)', selfd)
    for src in ('tables', 'text'):
        section(f'Без self, source={src}',
                [d for d in nonself if d['source'] == src])

    # Тройное совпадение (без self)
    tri = [d for d in nonself
           if d['inn_st'] == 'match' and d['kpp_st'] == 'match'
           and d['name_st'] == 'match']
    rep.append('')
    rep.append(f'Полное совпадение ИНН+КПП+ИМЯ (без self): {len(tri)} '
               f'({pct(len(tri), len(nonself))})')
    any_mm = [d for d in nonself
              if 'mismatch' in (d['inn_st'], d['kpp_st'], d['name_st'])]
    rep.append(f'Есть хотя бы одно расхождение (mismatch): {len(any_mm)} '
               f'({pct(len(any_mm), len(nonself))})')

    # Топ контрагентов ЭДО по расхождениям ИНН (без self)
    inn_mm = Counter((d['edo_short'] or d['edo_full'] or '?')
                     for d in nonself if d['inn_st'] == 'mismatch')
    if inn_mm:
        rep.append('')
        rep.append('Топ-15 контрагентов ЭДО по расхождению ИНН (без self):')
        for name, cnt in inn_mm.most_common(15):
            rep.append(f'  {cnt:4d}  {name[:70]}')

    # Примеры расхождений
    rep.append('')
    rep.append('Примеры расхождений ИНН (первые 15, без self):')
    for d in [x for x in nonself if x['inn_st'] == 'mismatch'][:15]:
        rep.append(f"  {d['file'][:60]}")
        rep.append(f"    LLM: {d['llm_name']!r} ИНН {d['llm_inn']} КПП {d['llm_kpp']}")
        rep.append(f"    ЭДО: {d['edo_short']!r} ИНН {d['edo_inn']} КПП {d['edo_kpp']}")

    report = '\n'.join(rep)
    print()
    print(report)

    WORK.mkdir(exist_ok=True)
    (WORK / 'contract_edo_validate_report.txt').write_text(
        report, encoding='utf-8')

    # CSV расхождений (любой mismatch среди найденных, без self)
    csv_path = WORK / 'contract_edo_divergences.csv'
    with csv_path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.writer(f)
        w.writerow(['doc_id', 'entity_id', 'source', 'file',
                    'llm_name', 'llm_inn', 'llm_kpp',
                    'edo_shortname', 'edo_inn', 'edo_kpp',
                    'inn_status', 'kpp_status', 'name_status'])
        for d in any_mm:
            w.writerow([d['doc_id'], d['eid'], d['source'], d['file'],
                        d['llm_name'], d['llm_inn'], d['llm_kpp'],
                        d['edo_short'], d['edo_inn'], d['edo_kpp'],
                        d['inn_st'], d['kpp_st'], d['name_st']])
    pg.close()
    print()
    print(f'✅ Отчёт: {WORK / "contract_edo_validate_report.txt"}')
    print(f'✅ CSV расхождений ({len(any_mm)}): {csv_path}')
    print(f'   Время: {time.time() - t0:.0f}s')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
