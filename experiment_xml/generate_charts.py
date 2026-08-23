#!/usr/bin/env python3
# generate_charts.py - Generate PNG charts for Confluence report
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / 'reports'
OUTPUT_DIR.mkdir(exist_ok=True)

# Data
pie_data = {
    'В пакетах': 379464,
    'Без пакетов': 282851
}

bar_data = [
    {'type': 'UniversalTransferDocument', 'title': 'УПД', 'packets': 62416, 'no_packet': 104831},
    {'type': 'Nonformalized', 'title': 'Неформализованный документ', 'packets': 139685, 'no_packet': 43141},
    {'type': 'Invoice', 'title': 'Счет-фактура', 'packets': 45740, 'no_packet': 10864},
    {'type': 'ReturnInventoryAcceptanceCertificate', 'title': 'Акт МХ-3 (возврат)', 'packets': 4255, 'no_packet': 50766},
    {'type': 'ProformaInvoice', 'title': 'Счет (проформа)', 'packets': 30884, 'no_packet': 22170},
    {'type': 'StorageInventoryAcceptanceCertificate', 'title': 'Акт МХ-1 (складской)', 'packets': 5340, 'no_packet': 34689},
    {'type': 'PerformedWorkAcceptanceCertificate', 'title': 'Акт КС-2 (выполненные работы)', 'packets': 29784, 'no_packet': 12},
    {'type': 'PerformedWorkCostCertificate', 'title': 'Справка КС-3 (смета работ)', 'packets': 29729, 'no_packet': 11},
    {'type': 'XmlAcceptanceCertificate', 'title': 'Акт (XmlAcceptanceCertificate)', 'packets': 15838, 'no_packet': 1180},
    {'type': 'ReconciliationAct', 'title': 'Акт сверки', 'packets': 408, 'no_packet': 6504},
    {'type': 'AcceptanceCertificate', 'title': 'Акт (AcceptanceCertificate)', 'packets': 5411, 'no_packet': 1417},
    {'type': 'UniversalCorrectionDocument', 'title': 'УКД', 'packets': 2249, 'no_packet': 2066},
    {'type': 'InvoiceCorrection', 'title': 'Корректировочный СФ', 'packets': 2392, 'no_packet': 77},
    {'type': 'ActOfsetting', 'title': 'Акт на взаимозачет', 'packets': 2115, 'no_packet': 2},
    {'type': 'ServiceDetails', 'title': 'Детализация услуг', 'packets': 1150, 'no_packet': 46}
]

# Set style
plt.style.use('default')
colors = ['#00d9ff', '#ff6b6b']  # cyan and coral

# ── Pie Chart ────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 8))
wedges, texts, autotexts = ax.pie(
    pie_data.values(), 
    labels=pie_data.keys(),
    colors=colors,
    autopct='%1.1f%%',
    startangle=90,
    explode=(0.05, 0),
    shadow=True
)
ax.set_title('Распределение документов по наличию PacketId (H1 2026)', fontsize=14, fontweight='bold', pad=20)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')
plt.tight_layout()
pie_path = OUTPUT_DIR / 'packet_distribution_pie.png'
fig.savefig(pie_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"OK: {pie_path}")
plt.close(fig)

# ── Stacked Bar Chart ────────────────────────
fig, ax = plt.subplots(figsize=(16, 12))
x_pos = np.arange(len(bar_data))
width = 0.35

packets_vals = [d['packets'] for d in bar_data]
no_packet_vals = [d['no_packet'] for d in bar_data]

titles_short = []
for t in [d['title'] for d in bar_data]:
    if len(t) > 25:
        titles_short.append(t[:22] + '...')
    else:
        titles_short.append(t)

bars1 = ax.bar(x_pos - width/2, packets_vals, width, label='В пакетах', color='#00d9ff', edgecolor='white', linewidth=0.5)
bars2 = ax.bar(x_pos + width/2, no_packet_vals, width, label='Без пакетов', color='#ff6b6b', edgecolor='white', linewidth=0.5)

ax.set_ylabel('Количество документов', fontsize=12)
ax.set_title('Документы в пакетах vs без пакетов по типам (H1 2026)', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x_pos)
ax.set_xticklabels(titles_short, rotation=45, ha='right', fontsize=9)
ax.legend(loc='upper right', fontsize=10)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 100:  # Only show labels for significant values
            ax.annotate(f'{int(height):,}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=7)

plt.tight_layout()
bar_path = OUTPUT_DIR / 'packet_by_type_bar.png'
fig.savefig(bar_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"OK: {bar_path}")
plt.close(fig)

print("\nDone! Charts saved to:", OUTPUT_DIR)