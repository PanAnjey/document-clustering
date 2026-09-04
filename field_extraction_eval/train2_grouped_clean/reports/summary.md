# «Правильное» дообучение на TRAIN2 — результаты

Сиды разбиения: [42, 123, 2024]. Групповой сплит (без утечки шаблонов), 52 заведомо ошибочные метки удалены. База Qwen3.5-4B, LoRA r=16, 1 эпоха — рецепт как в train2_full.

## Точность (среднее по сидам, в скобках min–max)

| Замер | Общая | На независимых | На док. с близнецами | macro-F1 |
|---|---|---|---|---|
| Qwen3.5-4B вхолодную | 73.7% (69.6%–76.7%) | 66.9% (64.3%–69.5%) | 78.7% (71.7%–82.4%) | 53.2% (52.2%–55.0%) |
| Qwen3.5-4B + LoRA | 88.1% (82.7%–92.3%) | 93.7% (92.2%–96.8%) | 84.0% (75.6%–89.1%) | 75.1% (70.8%–81.4%) |

Прирост от дообучения: **+14.4 п.п.** (73.7% → 88.1%).
Предыдущие замеры той же модели: наивный (случайный сплит, утечка) 90.3% общая; групповой сплит + грубая чистка (52 метки) 85.8% на независимых / 69.7% общая.
Честное число для сравнения с эмбеддингами — точность на независимых документах: **93.7%** (92.2%–96.8%).

## По классам (recall, среднее по сидам)

| Класс | n test | вхолодную | + LoRA | Δ |
|---|--:|--:|--:|--:|
| ContractAppendix | 20 | 96% | 70% | -26% |
| Claim | 3 | 100% | 87% | -13% |
| TrustConnectionRequest | 5 | 7% | 0% | -7% |
| Notification | 23 | 100% | 95% | -5% |
| ReconciliationAct | 35 | 100% | 99% | -1% |
| ActDisagreement | 4 | 25% | 25% | +0% |
| PowerOfAttorney | 33 | 100% | 100% | +0% |
| ServiceDetails | 42 | 82% | 82% | +0% |
| Torg12 | 4 | 18% | 18% | +0% |
| ProformaInvoice | 46 | 91% | 100% | +9% |
| SupplementaryAgreement | 53 | 83% | 97% | +13% |
| Contract | 28 | 81% | 96% | +15% |
| PriceListAgreement | 12 | 0% | 43% | +43% |
| AcceptanceCertificate | 18 | 28% | 93% | +65% |
| Letter | 28 | 15% | 99% | +84% |
| CertificateRegistry | 9 | 0% | 100% | +100% |

## Топ-путаницы дообученной модели (сид 42)

- PriceListAgreement → ServiceDetails: ×11
- Torg12 → ServiceDetails: ×5
- Contract → ContractAppendix: ×3
- ServiceDetails → Contract: ×3
- TrustConnectionRequest → PowerOfAttorney: ×3
- ActDisagreement → ServiceDetails: ×2
- Claim → Notification: ×2
- TrustConnectionRequest → Letter: ×2
- AcceptanceCertificate → SupplementaryAgreement: ×1
- AcceptanceCertificate → ServiceDetails: ×1