#!/usr/bin/env python3
# update_confluence_with_charts.py - Direct REST API call to update page with embedded charts
import base64
import json
import urllib.request
import urllib.error
from pathlib import Path

CONFLUENCE_BASE = "https://bwiki.beeline.ru"
PAGE_ID = "2225219675"
CHARTS_DIR = Path(__file__).parent / 'reports'

# Read and encode PNG files
def encode_png(filepath):
    with open(filepath, 'rb') as f:
        return base64.b64encode(f.read()).decode('ascii')

pie_png = CHARTS_DIR / 'packet_distribution_pie.png'
bar_png = CHARTS_DIR / 'packet_by_type_bar.png'

print("Reading chart files...")
pie_b64 = encode_png(pie_png)
bar_b64 = encode_png(bar_png)
print(f"Pie chart: {len(pie_b64)} chars base64")
print(f"Bar chart: {len(bar_b64)} chars base64")

# Generate Confluence storage format with embedded images
content_html = f'''<h2>Обзор</h2>
<p><strong>Период анализа:</strong> H1 2026 (январь–июнь), база <code>events_26.doc_in</code></p>
<p><strong>Всего документов:</strong> 662 315 | <strong>Уникальных пакетов:</strong> 136 649 | <strong>Средний размер пакета:</strong> 2.78 документа</p>

<h2>Общий баланс: в пакетах vs без пакетов</h2>
<div style="text-align: center; margin: 20px 0;"><img src="data:image/png;base64,{pie_b64}" alt="Pie chart: packet distribution" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);" /></div>

<table>
<caption>Числовые значения</caption>
<thead>
<tr><th>Категория</th><th>Документов</th><th>Доля</th></tr>
</thead>
<tbody>
<tr><td><strong>В пакетах</strong> (<code>PacketId IS NOT NULL</code>)</td><td>379 464</td><td><strong>57.29%</strong></td></tr>
<tr><td>Без пакетов (<code>PacketId IS NULL</code>)</td><td>282 851</td><td>42.71%</td></tr>
<tr><td><strong>Итого</strong></td><td><strong>662 315</strong></td><td><strong>100%</strong></td></tr>
</tbody>
</table>

<h2>Сравнение с полным годом 2025 (events_25)</h2>
<table>
<caption>Динамика доли документов без пакетов</caption>
<thead>
<tr><th>Период</th><th>В пакетах</th><th>Без пакетов</th><th>Доля без пакетов</th></tr>
</thead>
<tbody>
<tr><td>2025 год (полный)</td><td>965 035</td><td>617 413</td><td><strong>39.02%</strong></td></tr>
<tr><td>H1 2026</td><td>379 464</td><td>282 851</td><td><strong>42.71%</strong></td></tr>
<tr><td><strong>Изменение</strong></td><td>&mdash;</td><td>&mdash;</td><td><span style="color: #ff4757; font-weight: bold;">▲ +3.69 п.п.</span></td></tr>
</tbody>
</table>
<p><em>Доля одиночных документов растёт — возможно, изменение бизнес-процессов или подключение новых контрагентов с прямыми отправлениями.</em></p>

<h2>Распределение по типам документов (H1 2026)</h2>
<div style="text-align: center; margin: 20px 0;"><img src="data:image/png;base64,{bar_b64}" alt="Bar chart: documents by type" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);" /></div>

<h3>Типы, которые почти всегда идут в пакетах (&gt;90%)</h3>
<table>
<thead><tr><th>Тип документа</th><th>В пакетах</th><th>Без пакетов</th><th>% в пакете</th></tr></thead>
<tbody>
<tr><td><strong>Акт приемки выполненных работ КС-2</strong></td><td>29 784</td><td>12</td><td><span style="color: #00ff88; font-weight: bold;">99.96%</span></td></tr>
<tr><td><strong>Справка о стоимости выполненных работ КС-3</strong></td><td>29 729</td><td>11</td><td><span style="color: #00ff88; font-weight: bold;">99.96%</span></td></tr>
<tr><td><strong>Акт на взаимозачет</strong></td><td>2 115</td><td>2</td><td><span style="color: #00ff88; font-weight: bold;">99.91%</span></td></tr>
<tr><td><strong>Корректировочный счет-фактура</strong></td><td>2 392</td><td>77</td><td><span style="color: #00ff88; font-weight: bold;">96.88%</span></td></tr>
<tr><td><strong>Детализация услуг</strong></td><td>1 150</td><td>46</td><td><span style="color: #00ff88; font-weight: bold;">96.15%</span></td></tr>
</tbody>
</table>

<h3>Типы, которые почти всегда идут отдельно (&lt;20% в пакетах)</h3>
<table>
<thead><tr><th>Тип документа</th><th>В пакетах</th><th>Без пакетов</th><th>% в пакете</th></tr></thead>
<tbody>
<tr><td><strong>Акт МХ-3 (возврат)</strong></td><td>4 255</td><td>50 766</td><td><span style="color: #ffd93d; font-weight: bold;">7.73%</span></td></tr>
<tr><td><strong>Акт МХ-1 (складской)</strong></td><td>5 340</td><td>34 689</td><td><span style="color: #ffd93d; font-weight: bold;">13.34%</span></td></tr>
<tr><td><strong>Акт сверки</strong></td><td>408</td><td>6 504</td><td><span style="color: #ffd93d; font-weight: bold;">5.90%</span></td></tr>
</tbody>
</table>

<h2>Ключевые выводы</h2>
<ol>
<li><strong>Большинство документов (57%)</strong> поступают в пакетах — это основной режим работы системы.</li>
<li><strong>Акт возврата МХ-3 и складской акт МХ-1</strong> почти всегда идут отдельно (6–13% в пакетах) — вероятно, это реактивные документы, формируемые по факту приёмки/возврата товара.</li>
<li><strong>УПД делятся поровну:</strong> 62K в пакетах vs 105K отдельно — зависит от контрагента и сценария отгрузки.</li>
<li><strong>Доля одиночных документов растёт</strong> (с 39% в 2025 до 43% в H1 2026) — стоит мониторить этот тренд.</li>
</ol>
'''

print(f"\nTotal HTML content size: {len(content_html)} chars")

# Prepare REST API request
url = f"{CONFLUENCE_BASE}/rest/api/content/{PAGE_ID}"
data = {
    "title": "Анализ пакетов документов (PacketId) — events_26, H1 2026",
    "type": "page",
    "status": "current",
    "body": {
        "storage": {
            "value": content_html,
            "representation": "storage"
        }
    },
    "version": {
        "number": 4  # Increment version
    }
}

print("\nSending update request to Confluence...")
req = urllib.request.Request(
    url,
    data=json.dumps(data).encode('utf-8'),
    headers={
        'Content-Type': 'application/json',
        'Accept': 'application/json'
    },
    method='PUT'
)

try:
    with urllib.request.urlopen(req, timeout=60) as response:
        result = json.loads(response.read().decode('utf-8'))
        print(f"SUCCESS! Page updated.")
        print(f"Page ID: {result.get('id')}")
        print(f"Version: {result.get('version', {}).get('number')}")
except urllib.error.HTTPError as e:
    print(f"HTTP Error {e.code}: {e.reason}")
    print(e.read().decode('utf-8'))
except Exception as e:
    print(f"Error: {e}")