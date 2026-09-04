#!/usr/bin/env python3
# generate_small_charts.py - Generate smaller PNG charts for Confluence embedding
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / 'reports'

# Data
pie_data = {'В пакетах': 379464, 'Без пакетов': 282851}
colors = ['#00d9ff', '#ff6b6b']

bar_data = [
    {'title': 'УПД', 'packets': 62416, 'no_packet': 104831},
    {'title': 'Неформализованный документ', 'packets': 139685, 'no_packet': 43141},
    {'title': 'Счет-фактура', 'packets': 45740, 'no_packet': 10864},
    {'title': 'Акт МХ-3 (возврат)', 'packets': 4255, 'no_packet': 50766},
    {'title': 'Счет (проформа)', 'packets': 30884, 'no_packet': 22170},
    {'title': 'Акт МХ-1 (складской)', 'packets': 5340, 'no_packet': 34689},
    {'title': 'Акт КС-2', 'packets': 29784, 'no_packet': 12},
    {'title': 'Справка КС-3', 'packets': 29729, 'no_packet': 11},
    {'title': 'Акт сверки', 'packets': 408, 'no_packet': 6504},
]

# ── Small Pie Chart (600x400) ────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
wedges, texts, autotexts = ax.pie(
    pie_data.values(), labels=pie_data.keys(), colors=colors,
    autopct='%1.1f%%', startangle=90, explode=(0.05, 0)
)
ax.set_title('Документы: в пакетах vs без пакетов', fontsize=11, fontweight='bold')
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(10)
plt.tight_layout()
pie_small = OUTPUT_DIR / 'packet_pie_small.png'
fig.savefig(pie_small, dpi=100, bbox_inches='tight', facecolor='white')
print(f"Pie: {pie_small} ({pie_small.stat().st_size/1024:.1f} KB)")
plt.close(fig)

# ── Small Bar Chart (800x500) ────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
x_pos = range(len(bar_data))
bars1 = ax.bar([x-0.2 for x in x_pos], [d['packets'] for d in bar_data], 0.4, label='В пакетах', color='#00d9ff')
bars2 = ax.bar([x+0.2 for x in x_pos], [d['no_packet'] for d in bar_data], 0.4, label='Без пакетов', color='#ff6b6b')
ax.set_title('Документы по типам (топ-9)', fontsize=11, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([d['title'] for d in bar_data], rotation=45, ha='right', fontsize=8)
ax.legend(fontsize=9)
plt.tight_layout()
bar_small = OUTPUT_DIR / 'packet_bar_small.png'
fig.savefig(bar_small, dpi=100, bbox_inches='tight', facecolor='white')
print(f"Bar: {bar_small} ({bar_small.stat().st_size/1024:.1f} KB)")
plt.close(fig)

print("\nSmall charts generated!")