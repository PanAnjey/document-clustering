# phase_x3_report.py
# Этап X3 эксперимента XML: сравнительный анализ распределений тем
# формализованных (XML УПД/СФ) и неформализованных (pdf_text, pdf_tables)
# документов по 26 темам предыдущего эксперимента.
#
# Выход: консоль + markdown-отчёт в experiment_xml/reports/.
#
# Запуск: python phase_x3_report.py

import datetime
import sys
from pathlib import Path

import numpy as np
import psycopg2

from experiment_xml_config import (  # noqa: E402
    DB_URL, T_ASSIGN, T_SUBJECT_ASSIGN, CENTROIDS_TABLE, SIMILARITY_THRESHOLD,
    MARKED_THEME,
)

REPORTS_DIR = Path(__file__).resolve().parent / 'reports'

SOURCES = [
    ('XML V1 (полный канон)', T_ASSIGN),
    ('XML V2 (тематический)', T_SUBJECT_ASSIGN),
    ('XML V3 (темат. очищенный)', 'temp_xml_subject_clean_assign'),
    # _v2: переназначены на полный набор 26 центроидов (phase_x3b)
    ('PDF_Text', 'temp_pdftext_assign_v2'),
    ('PDF_Tables', 'temp_pdftables_assign_v2'),
]


def fetch_assignments(cur, table):
    cur.execute(f"""SELECT assigned_theme, COUNT(*), AVG(similarity),
                           SUM(is_unknown::int)
                    FROM {table} GROUP BY assigned_theme""")
    return {int(t): (int(c), float(s), int(u)) for t, c, s, u in cur.fetchall()}


def fetch_totals(cur, table):
    cur.execute(f"SELECT COUNT(*), AVG(similarity), SUM(is_unknown::int) FROM {table}")
    n, avg_s, unk = cur.fetchone()
    return int(n), float(avg_s), int(unk)


def js_divergence(p, q):
    """Jensen-Shannon divergence (nat) между двумя распределениями."""
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def main():
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()

    cur.execute(f"SELECT theme_label, top_words FROM {CENTROIDS_TABLE}")
    words = {int(t): (w or '') for t, w in cur.fetchall()}
    words[MARKED_THEME] = 'ГИС МТ (маркированные товары, N3=1)'
    themes = sorted(words)

    data = {}
    totals = {}
    for name, table in SOURCES:
        cur.execute(f"SELECT to_regclass('public.{table}')")
        if cur.fetchone()[0] is None:
            print(f"⚠️ таблица {table} отсутствует — источник пропущен")
            continue
        data[name] = fetch_assignments(cur, table)
        totals[name] = fetch_totals(cur, table)

    # Распределения по темам (доли от назначенных, без unknown)
    shares = {}
    for name in data:
        vec = np.array([data[name].get(t, (0, 0, 0))[0] for t in themes], dtype=float)
        shares[name] = vec / vec.sum() if vec.sum() else vec

    L = []
    L.append(f"# Сравнение формализованных (XML) и неформализованных (PDF) документов")
    L.append(f"по темам предыдущего эксперимента ({CENTROIDS_TABLE}, порог {SIMILARITY_THRESHOLD})")
    L.append(f"\n_Сформировано: {datetime.datetime.now():%Y-%m-%d %H:%M}_\n")

    # ── 1. Покрытие ──
    L.append("## 1. Покрытие назначением (similarity к ближайшему центроиду)\n")
    L.append("| Источник | Документов | avg sim | unknown | coverage |")
    L.append("|---|---|---|---|---|")
    for name in data:
        n, avg_s, unk = totals[name]
        L.append(f"| {name} | {n} | {avg_s:.3f} | {unk} ({unk/n:.1%}) | {1-unk/n:.1%} |")

    # ── 2. Распределение по темам ──
    L.append("\n## 2. Доля документов по темам (все назначения, включая unknown)\n")
    hdr = "| Тема | Топ-слова |" + "".join(f" {n} |" for n in data)
    L.append(hdr)
    L.append("|---|---|" + "---|" * len(data))
    for t in themes:
        row = f"| {t} | {words[t][:45]} |"
        for name in data:
            c = data[name].get(t, (0, 0, 0))[0]
            row += f" {c} ({shares[name][themes.index(t)]:.1%}) |"
        L.append(row)

    # ── 3. Похожесть распределений ──
    L.append("\n## 3. Похожесть распределений (JS-дивергенция / косинус долей)\n")
    names = list(data)
    L.append("| |" + "".join(f" {n} |" for n in names))
    L.append("|---|" + "---|" * len(names))
    for a in names:
        row = f"| {a} |"
        for b in names:
            js = js_divergence(shares[a], shares[b])
            cos = float(shares[a] @ shares[b] /
                        (np.linalg.norm(shares[a]) * np.linalg.norm(shares[b]) + 1e-12))
            row += f" {js:.3f} / {cos:.3f} |"
        L.append(row)

    # ── 4. XML по типам/функциям (V2) ──
    cur.execute(f"""
        SELECT c.type_named_id, c.doc_function, COUNT(*) n,
               AVG(a.similarity)::numeric(5,3), SUM(a.is_unknown::int) unk
        FROM {T_SUBJECT_ASSIGN} a
        JOIN temp_xml_canonical c ON c.id = a.xml_id
        GROUP BY 1, 2 ORDER BY n DESC""")
    L.append(f"\n## 4. XML (V2) по типам документов и функциям\n")
    L.append("| TypeNamedId | Функция | n | avg sim | unknown |")
    L.append("|---|---|---|---|---|")
    for tname, func, n, s, unk in cur.fetchall():
        L.append(f"| {tname} | {func} | {n} | {s} | {unk} ({unk/n:.1%}) |")

    # ── 5. Топ-темы XML (V3) ──
    L.append(f"\n## 5. Топ-10 тем XML (V3, очищенный тематический)\n")
    L.append("| Тема | n | avg sim | Топ-слова |")
    L.append("|---|---|---|---|")
    xml3 = data.get('XML V3 (темат. очищенный)', {})
    for t, (c, s, u) in sorted(xml3.items(), key=lambda kv: -kv[1][0])[:10]:
        L.append(f"| {t} | {c} ({shares['XML V3 (темат. очищенный)'][themes.index(t)]:.1%}) | {s:.3f} | {words[t][:50]} |")

    # ── 6. Миграция V2 → V3 (куда ушли документы темы 26) ──
    cur.execute("""
        SELECT a3.assigned_theme t3, COUNT(*) n
        FROM temp_xml_subject_assign a2
        JOIN temp_xml_subject_clean_assign a3 ON a3.xml_id = a2.xml_id
        WHERE a2.assigned_theme = 26
        GROUP BY 1 ORDER BY n DESC LIMIT 12""")
    L.append(f"\n## 6. Куда перешли документы темы 26 после очистки (V2→V3)\n")
    L.append("| V3 тема | n | Топ-слова V3 |")
    L.append("|---|---|---|")
    for t3, n in cur.fetchall():
        L.append(f"| {t3} | {n} | {words.get(t3, '')[:55]} |")

    report = "\n".join(L)
    REPORTS_DIR.mkdir(exist_ok=True)
    out = REPORTS_DIR / f"xml_vs_pdf_themes_{datetime.datetime.now():%Y%m%d_%H%M}.md"
    out.write_text(report, encoding='utf-8')
    print(report)
    print(f"\n✅ отчёт: {out}")
    conn.close()


if __name__ == '__main__':
    main()
