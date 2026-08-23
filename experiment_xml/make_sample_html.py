# make_sample_html.py
# HTML-витрина результатов канонизатора: исходный XML (MySQL) рядом с
# каноническим V1, тематическим V2 и очищенным V3 (PostgreSQL).
#
# Выборка: стратифицированная — UTD/Invoice, обычные/многострочные,
# с ИнфПолФХЖ, разные регионы.
#
# Запуск: python make_sample_html.py [--n 30]
# Результат: experiment_xml/reports/canonical_samples.html

import argparse
import html
import sys
from pathlib import Path

import psycopg2
import pymysql

from experiment_xml_config import (  # noqa: E402
    DB_URL, MYSQL_HOST, MYSQL_PORT, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DB, MYSQL_TABLE,
)
from xml_canonicalizer import strip_contract_refs  # noqa: E402

REPORTS_DIR = Path(__file__).resolve().parent / 'reports'
MAX_XML_SHOW = 6000  # символов исходного XML в карточке (с кнопкой-усечением)

CSS = """
body { font-family: Segoe UI, Arial, sans-serif; margin: 20px; background: #f5f6f8; }
h1 { font-size: 22px; } h2 { font-size: 16px; margin: 4px 0; }
.card { background: #fff; border: 1px solid #d0d4da; border-radius: 8px;
        margin: 18px 0; padding: 14px; box-shadow: 0 1px 3px rgba(0,0,0,.06); }
.meta { color: #445; font-size: 13px; margin-bottom: 8px; }
.meta b { color: #000; }
.cols { display: flex; gap: 14px; }
.col { flex: 1; min-width: 0; }
pre { background: #f8f9fb; border: 1px solid #e3e6ea; border-radius: 6px;
      padding: 10px; font-size: 11.5px; white-space: pre-wrap;
      word-wrap: break-word; max-height: 420px; overflow-y: auto; }
.v1 pre { background: #eef5ff; } .v2 pre { background: #eefaf0; } .v3 pre { background: #fff7e8; }
.tag { display: inline-block; background: #e8ecf2; border-radius: 4px;
       padding: 1px 7px; margin: 1px 3px 1px 0; font-size: 11.5px; }
.len { color: #889; font-weight: normal; font-size: 12px; }
.nav { position: sticky; top: 0; background: #f5f6f8; padding: 8px 0; z-index: 5; }
a { color: #2a6fdb; }
"""


def fetch_sample(cur, n):
    """Стратифицированная выборка id из temp_xml_canonical."""
    buckets = [
        ("UTD обычные", f"""SELECT id FROM temp_xml_canonical
            WHERE parse_ok AND type_named_id='UniversalTransferDocument'
              AND n_rows_total <= 5 ORDER BY random() LIMIT {n // 3}"""),
        ("Invoice обычные", f"""SELECT id FROM temp_xml_canonical
            WHERE parse_ok AND type_named_id='Invoice'
              AND n_rows_total <= 5 ORDER BY random() LIMIT {n // 3}"""),
        ("Многострочные (>50 строк)", f"""SELECT id FROM temp_xml_canonical
            WHERE parse_ok AND n_rows_total > 50 ORDER BY random() LIMIT {max(2, n // 8)}"""),
        ("С ИнфПолФХЖ (Доп)", f"""SELECT id FROM temp_xml_canonical
            WHERE parse_ok AND canonical_text LIKE '%Доп%' ORDER BY random() LIMIT {max(2, n // 8)}"""),
        ("Разные регионы", f"""SELECT id FROM temp_xml_canonical
            WHERE parse_ok AND seller_region NOT IN ('Москва','Санкт-Петербург','')
              AND seller_region IS NOT NULL
            ORDER BY random() LIMIT {max(2, n // 8)}"""),
    ]
    out = []
    seen = set()
    for label, sql in buckets:
        cur.execute(sql)
        ids = [r[0] for r in cur.fetchall() if r[0] not in seen]
        seen.update(ids)
        out.append((label, ids))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=30)
    args = ap.parse_args()

    pg = psycopg2.connect(DB_URL)
    my = pymysql.connect(host=MYSQL_HOST, port=MYSQL_PORT, user=MYSQL_USER,
                         password=MYSQL_PASSWORD, database=MYSQL_DB, charset='utf8mb4')

    pgc = pg.cursor()
    buckets = fetch_sample(pgc, args.n)

    parts = [f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>"]
    parts.append("<h1>Канонизатор XML (ON_NSCHFDOPPR): исходник → V1 → V2 → V3</h1>")
    parts.append("<div class='meta'>V1 — полный канон; V2 — тематический; "
                 "V3 — тематический очищенный от договорных ссылок/периодов. "
                 "Таблица: temp_xml_canonical (file_organizer_db).</div>")

    # навигация
    nav = ["<div class='nav'><b>Карточки:</b> "]
    idx = 0
    for label, ids in buckets:
        nav.append(f"&nbsp;| <a href='#b{idx}'>{label} ({len(ids)})</a>")
        idx += 1
    nav.append("</div>")
    parts.append("".join(nav))

    idx = 0
    for label, ids in buckets:
        parts.append(f"<h2 id='b{idx}'>{label}</h2>")
        idx += 1
        if not ids:
            continue
        fmt = ','.join(['%s'] * len(ids))
        pgc.execute(f"""SELECT id, message_id, entity_id, type_named_id, doc_function,
                               doc_number, doc_date, seller_name, seller_inn, seller_region,
                               buyer_name, sum_total, currency, gov_contract, osn,
                               n_rows_total, n_rows_taken, src_len, canon_len, subject_len,
                               canonical_text, subject_text
                        FROM temp_xml_canonical WHERE id IN ({fmt})""", ids)
        rows = pgc.fetchall()
        for (cid, mid, eid, tnamed, func, dnum, ddate, sname, sinn, sreg,
             bname, total, curr, gov, osn, nrt, nrtk, srcl, canl, subl,
             canon, subj) in rows:
            with my.cursor() as mc:
                mc.execute(f"SELECT xml_val FROM {MYSQL_TABLE} "
                           f"WHERE MessageId=%s AND EntityId=%s", (mid, eid))
                r = mc.fetchone()
            xml_src = r[0] if r and r[0] else '(не найден в MySQL)'
            v3 = strip_contract_refs(subj or '')
            xml_show = xml_src[:MAX_XML_SHOW]
            cut = f"\n… [усечено, всего {len(xml_src)} симв.]" if len(xml_src) > MAX_XML_SHOW else ''

            parts.append("<div class='card'>")
            parts.append(
                f"<div class='meta'><b>#{cid} {tnamed}</b> · функция <b>{func}</b> · "
                f"№{html.escape(str(dnum))} от {ddate} · строк таблицы: <b>{nrt}</b> (взято {nrtk}) · "
                f"продавец: {html.escape(str(sname))} (ИНН {sinn}) · регион: <b>{html.escape(str(sreg or '—'))}</b> · "
                f"сумма: {total} {curr or ''} · ГК: {gov or '—'}<br>"
                f"<span class='tag'>src {srcl} симв.</span><span class='tag'>V1 {canl}</span>"
                f"<span class='tag'>V2 {subl}</span><span class='tag'>V3 {len(v3)}</span>"
                + (f"<span class='tag'>Осн: {html.escape((osn or '')[:80])}</span>" if osn else "")
                + "</div>")
            parts.append("<div class='cols'>")
            parts.append(f"<div class='col'><h2>Исходный XML</h2><pre>{html.escape(xml_show + cut)}</pre></div>")
            parts.append(f"<div class='col v1'><h2>V1 канон <span class='len'>{canl} симв.</span></h2><pre>{html.escape(canon or '')}</pre></div>")
            parts.append(f"<div class='col v2'><h2>V2 тематический <span class='len'>{subl} симв.</span></h2><pre>{html.escape(subj or '')}</pre></div>")
            parts.append(f"<div class='col v3'><h2>V3 очищенный <span class='len'>{len(v3)} симв.</span></h2><pre>{html.escape(v3)}</pre></div>")
            parts.append("</div></div>")

    parts.append("</body></html>")
    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / 'canonical_samples.html'
    out.write_text("\n".join(parts), encoding='utf-8')
    print(f"OK: {out} ({len(parts)} блоков)")
    pg.close()
    my.close()


if __name__ == '__main__':
    main()
