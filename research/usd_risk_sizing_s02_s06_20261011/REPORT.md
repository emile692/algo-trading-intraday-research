# S02–S06 — sizing par budget de risque USD

**Verdict : `TRADEOFF`.** Campagne isolée; aucune référence gelée n'a été modifiée.

## Audit et conventions

- Reproduction des six ledgers : erreur absolue maximale `0` USD.
- S01 est intrinsèquement risk-sized à 250 USD dans son fichier de trades, mais le book actif additionne une copie de cette sleeve déjà dimensionnée. La référence de cette campagne conserve donc S01 telle quelle dans toutes les branches.
- S02 : le stop publié est l'open RTH; l'overlay 240 minutes gelé l'a effacé du ledger. Il sert ici d'ancre de sizing, sans réécrire les sorties gelées. Cette divergence empêche d'interpréter le risque prévu comme une perte garantie.
- Quantité décidée avec le niveau d'ordre connu; les ordres au prochain open utilisent le dernier close observable ou une distance ATR déjà figée. Un gap de fill ne déclenche jamais de rescaling; son dépassement est mesuré dans `risk_audit.csv`.
- Allocation: commission round-trip plus deux ticks nominaux round-trip. Les stress ajoutent 1 ou 2 ticks par côté. Quantité zéro = signal ignoré.
- Les périodes 2024–2026 ont déjà servi aux décisions MNQ/MGC/MBT; aucune revendication de nouvel OOS indépendant.

## Activité de sizing

| Sleeve | Budget | Exécutés | Ignorés | Qté moyenne | Qté max | Risque prévu moyen | Risque fill moyen | Fill > budget |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mnq_volatility_breakout | 100 | 0 | 291 | nan | 0 | nan | nan | 0 |
| mnq_compression_breakout | 100 | 71 | 39 | 1.07 | 3 | 78.0 | 78.5 | 0 |
| mnq_atr_trend_flip | 100 | 275 | 384 | 1.00 | 2 | 81.4 | 81.4 | 0 |
| mgc_volatility_breakout | 100 | 0 | 65 | nan | 0 | nan | nan | 0 |
| mgc_gap_continuation | 100 | 6 | 89 | 1.00 | 1 | 91.7 | 92.0 | 1 |
| mnq_volatility_breakout | 150 | 0 | 291 | nan | 0 | nan | nan | 0 |
| mnq_compression_breakout | 150 | 100 | 10 | 1.36 | 5 | 107.6 | 108.3 | 1 |
| mnq_atr_trend_flip | 150 | 510 | 149 | 1.17 | 3 | 111.3 | 111.3 | 0 |
| mgc_volatility_breakout | 150 | 0 | 65 | nan | 0 | nan | nan | 0 |
| mgc_gap_continuation | 150 | 26 | 69 | 1.04 | 2 | 125.7 | 126.4 | 1 |
| mnq_volatility_breakout | 200 | 0 | 291 | nan | 0 | nan | nan | 0 |
| mnq_compression_breakout | 200 | 106 | 4 | 1.97 | 7 | 157.8 | 158.8 | 0 |
| mnq_atr_trend_flip | 200 | 608 | 51 | 1.53 | 4 | 153.1 | 153.1 | 0 |
| mgc_volatility_breakout | 200 | 0 | 65 | nan | 0 | nan | nan | 0 |
| mgc_gap_continuation | 200 | 43 | 52 | 1.14 | 2 | 153.5 | 154.2 | 1 |
| mnq_volatility_breakout | 250 | 16 | 275 | 1.00 | 1 | 242.5 | 243.3 | 2 |
| mnq_compression_breakout | 250 | 108 | 2 | 2.46 | 9 | 198.6 | 199.8 | 2 |
| mnq_atr_trend_flip | 250 | 640 | 19 | 1.90 | 5 | 194.1 | 194.1 | 0 |
| mgc_volatility_breakout | 250 | 8 | 57 | 1.00 | 1 | 232.5 | 234.1 | 0 |
| mgc_gap_continuation | 250 | 55 | 40 | 1.22 | 3 | 179.9 | 180.7 | 0 |

## Books — nominal

| Scénario | PnL net | Sharpe | PF | maxDD | Pire jour | Pires 5 jours |
|---|---:|---:|---:|---:|---:|---:|
| reference__six__slip0 | 33,707 | 1.61 | 1.45 | -5,419 | -1,234 | -5,283 |
| reference__nine__slip0 | 43,225 | 1.83 | 1.49 | -6,014 | -1,378 | -6,147 |
| budget_100__six__slip0 | 5,438 | 0.83 | 1.18 | -2,202 | -480 | -2,058 |
| budget_100__nine__slip0 | 14,957 | 1.49 | 1.29 | -3,129 | -898 | -3,716 |
| budget_150__six__slip0 | 7,936 | 0.86 | 1.17 | -3,695 | -715 | -3,240 |
| budget_150__nine__slip0 | 17,454 | 1.39 | 1.27 | -4,769 | -1,106 | -4,188 |
| budget_200__six__slip0 | 8,011 | 0.67 | 1.13 | -6,165 | -751 | -3,578 |
| budget_200__nine__slip0 | 17,530 | 1.18 | 1.23 | -6,613 | -1,172 | -4,872 |
| budget_250__six__slip0 | 10,386 | 0.69 | 1.13 | -6,917 | -968 | -4,301 |
| budget_250__nine__slip0 | 19,905 | 1.13 | 1.22 | -7,727 | -1,314 | -5,449 |

## Topstep — bootstrap mobile apparié, 60 séances

| Scénario | PASS | MLL | EXPIRY | Jours vers PASS | DD médian |
|---|---:|---:|---:|---:|---:|
| budget_100__nine__slip0 | 34.1% | 27.4% | 38.5% | 39 | -1,559 |
| budget_100__six__slip0 | 5.1% | 10.3% | 84.5% | 33 | -1,192 |
| budget_150__nine__slip0 | 40.1% | 39.9% | 20.0% | 34 | -1,823 |
| budget_150__six__slip0 | 18.8% | 30.2% | 51.0% | 33 | -1,641 |
| budget_200__nine__slip0 | 38.1% | 51.8% | 10.1% | 29 | -2,033 |
| budget_200__six__slip0 | 24.5% | 52.4% | 23.1% | 31 | -2,023 |
| budget_250__nine__slip0 | 36.6% | 58.2% | 5.2% | 25 | -2,063 |
| budget_250__six__slip0 | 27.9% | 63.0% | 9.1% | 26 | -2,078 |
| reference__nine__slip0 | 45.2% | 48.1% | 6.7% | 23 | -2,055 |
| reference__six__slip0 | 38.2% | 49.2% | 12.5% | 27 | -2,039 |

## Exposition

| Variante | Book | Total micro max | MNQ | MGC | MBT | Limite dépassée |
|---|---|---:|---:|---:|---:|---|
| reference | six | 4 | 4 | 2 | 0 | False |
| reference | nine | 7 | 4 | 2 | 3 | False |
| budget_100 | six | 3 | 3 | 1 | 0 | False |
| budget_100 | nine | 6 | 3 | 1 | 3 | False |
| budget_150 | six | 5 | 5 | 2 | 0 | False |
| budget_150 | nine | 7 | 5 | 2 | 3 | False |
| budget_200 | six | 7 | 7 | 2 | 0 | False |
| budget_200 | nine | 9 | 7 | 2 | 3 | False |
| budget_250 | six | 9 | 9 | 3 | 0 | False |
| budget_250 | nine | 11 | 9 | 3 | 3 | False |

## Lecture

Le verdict porte sur la grille commune entière, sans choisir le budget qui maximise le PASS. Les CSV contiennent les résultats annuels, les sleeves, les trois stress, les horizons 35/60 et les deltas appariés. Les intervalles utilisent des chemins par blocs communs aux variantes afin de préserver la dépendance sérielle et l'appariement; les fenêtres historiques chevauchantes restent descriptives.

## Reproduction

```powershell
.\.venv\Scripts\python.exe research/usd_risk_sizing_s02_s06_20261011/run_campaign.py
```

Figures: `figures/performance_comparison.png`, `figures/topstep_frontier_60d.png`. Tables principales: `sizing_activity.csv`, `performance_metrics.csv`, `yearly_results.csv`, `exposure_summary.csv`, `topstep_summary.csv`, `topstep_paired_deltas.csv`, `risk_audit.csv`, `resized_trades.csv`.
