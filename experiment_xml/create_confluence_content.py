#!/usr/bin/env python3
# create_confluence_content.py - Generate HTML with embedded small charts
import base64
from pathlib import Path

CHARTS_DIR = Path(__file__).parent / 'reports'

def encode_png(filepath):
    with open(filepath, 'rb') as f:
        return base64.b64encode(f.read()).decode('ascii')

pie_b64 = encode_png(CHARTS_DIR / 'packet_pie_small.png')
bar_b64 = encode_png(CHARTS_DIR / 'packet_bar_small.png')

print(f"Pie base64: {len(pie_b64)} chars")
print(f"Bar base64: {len(bar_b64)} chars")

html_content = f'''<h2>Обзор</h2>
<p><strong>Период анализа:</strong> H1 2026 (январь–июнь), база <code>events_26.doc_in</code></p>
<p><strong>Всего документов:</strong> 662 315 | <strong>Уникальных пакетов:</strong> 136 649 | <strong>Средний размер пакета:</strong> 2.78 документа</p>

<h2>Общий баланс: в пакетах vs без пакетов</h2>
<div style="text-align: center; margin: 20px 0;"><img src="data:image/png;base64,{pie_b64}" alt="Pie chart" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);" /></div>

<table>
<thead><tr><th>Категория</th><th>Документов</th><th>Доля</th></tr></thead>
<tbody>
<tr><td><strong>В пакетах</strong> (<code>PacketId IS NOT NULL</code>)</td><td>379 464</td><td><strong>57.29%</strong></td></tr>
<tr><td>Без пакетов (<code>PacketId IS NULL</code>)</td><td>282 851</td><td>42.71%</td></tr>
<tr><td><strong>Итого</strong></td><td><strong>662 315</strong></td><td><strong>100%</strong></td></tr>
</tbody>
</table>

<h2>Сравнение с полным годом 2025 (events_25)</h2>
<table>
<thead><tr><th>Период</th><th>В пакетах</th><th>Без пакетов</th><th>Доля без пакетов</th></tr></thead>
<tbody>
<tr><td>2025 год (полный)</td><td>965 035</td><td>617 413</td><td><strong>39.02%</strong></td></tr>
<tr><td>H1 2026</td><td>379 464</td><td>282 851</td><td><strong>42.71%</strong></td></tr>
<tr><td><strong>Изменение</strong></td><td>&mdash;</td><td>&mdash;</td><td><span style="color: #ff4757; font-weight: bold;">▲ +3.69 п.п.</span></td></tr>
</tbody>
</table>

<h2>Распределение по типам документов (H1 2026)</h2>
<div style="text-align: center; margin: 20px 0;"><img src="data:image/png;base64,{bar_b64}" alt="Bar chart" style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1);" /></div>

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

# Write to file for verification
output_file = CHARTS_DIR.parent / '_confluence_html_content.txt'
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(html_content)
print(f"\nHTML content written to: {output_file}")
print(f"Total size: {len(html_content)} chars ({len(html_content)/1024:.1f} KB)")