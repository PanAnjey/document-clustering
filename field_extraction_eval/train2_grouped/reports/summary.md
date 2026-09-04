# «Правильное» дообучение на TRAIN2 — результаты

Сиды разбиения: [42, 123, 2024]. Групповой сплит (без утечки шаблонов), 52 заведомо ошибочные метки удалены. База Qwen3.5-4B, LoRA r=16, 1 эпоха — рецепт как в train2_full.

## Точность (среднее по сидам, в скобках min–max)

| Замер | Общая | На независимых | На док. с близнецами | macro-F1 |
|---|---|---|---|---|
| Qwen3.5-4B вхолодную | 54.4% (50.9%–60.9%) | 64.7% (62.5%–66.9%) | 49.5% (43.1%–59.1%) | 45.2% (42.0%–50.5%) |
| Qwen3.5-4B + LoRA | 69.7% (65.8%–73.9%) | 85.8% (83.3%–88.0%) | 61.8% (57.5%–68.0%) | 65.1% (61.0%–69.3%) |

Прирост от дообучения: **+15.2 п.п.** (54.4% → 69.7%).
Старый замер (случайный сплит с утечкой шаблонов): дообученная 90.3%, вхолодную 69.2%.
Честное число для сравнения с эмбеддингами — точность на независимых документах: **85.8%** (83.3%–88.0%).

## По классам (recall, среднее по сидам)

| Класс | n test | вхолодную | + LoRA | Δ |
|---|--:|--:|--:|--:|
| TrustConnectionRequest | 4 | 25% | 0% | -25% |
| ActDisagreement | 3 | 44% | 28% | -17% |
| Notification | 30 | 100% | 88% | -12% |
| ContractAppendix | 10 | 64% | 57% | -7% |
| Claim | 2 | 100% | 100% | +0% |
| PowerOfAttorney | 29 | 100% | 100% | +0% |
| ReconciliationAct | 34 | 100% | 100% | +0% |
| Torg12 | 70 | 8% | 16% | +9% |
| ProformaInvoice | 49 | 89% | 98% | +9% |
| Contract | 51 | 58% | 68% | +10% |
| SupplementaryAgreement | 43 | 64% | 74% | +10% |
| PriceListAgreement | 14 | 0% | 21% | +21% |
| ServiceDetails | 39 | 58% | 84% | +26% |
| AcceptanceCertificate | 36 | 11% | 62% | +51% |
| Letter | 9 | 19% | 81% | +63% |
| CertificateRegistry | 15 | 0% | 100% | +100% |

## Топ-путаницы дообученной модели (сид 42)

- Torg12 → AcceptanceCertificate: ×66
- Contract → SupplementaryAgreement: ×19
- PriceListAgreement → ServiceDetails: ×11
- ServiceDetails → AcceptanceCertificate: ×5
- SupplementaryAgreement → Contract: ×5
- Torg12 → ServiceDetails: ×4
- AcceptanceCertificate → ServiceDetails: ×3
- ActDisagreement → ReconciliationAct: ×3
- ServiceDetails → Contract: ×3
- TrustConnectionRequest → ServiceDetails: ×2