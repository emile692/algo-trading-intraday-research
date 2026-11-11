# MCL — validation restreinte de deux mécanismes

## Décision

| Mécanisme | Décision | Motif principal |
|---|---|---|
| `short_horizon_pullback_reversion` | **ABANDON AS SLEEVE** | Les cinq représentants n'ont que 1–11 trades IS et 1–5 trades OOS. La diagonale suit une rareté de queue, mais ne fournit pas un échantillon permettant d'établir un phénomène persistant. |
| `ma_crossover_trend` | **ABANDON AS SLEEVE** | Plateau nominal commun, mais toutes les variantes perdent en 2024 et l'avantage disparaît sous un stress de coûts raisonnable : 2/5 positives à deux ticks, 0/5 à trois ticks ou avec commission doublée. |

Les 14 autres mécanismes MCL restent `FROZEN_NOT_REOPENED`. Aucune nouvelle grille, aucun nouveau seuil et aucune sélection du meilleur Sharpe n'ont été introduits.

## Protocole verrouillé

- Configuration figée avant calcul : `4db861905c6ae2c65b3609cc2a2d69e529cdbd47495e2946ed4d28b69efac1b9`.
- IS : `2021-07-12 → 2023-12-31`; OOS : `2024-01-01 → 2026-03-19`.
- Même moteur, session MCL, taille fixe, conventions de fill et coûts nominaux que la campagne initiale.
- 5 représentants par mécanisme, choisis par géométrie du paysage et non par classement de performance.
- 40 runs : 10 représentants × 4 scénarios de coûts.
- Aucun changement des 16 spécifications sources frozen pendant l'exécution.

Représentants RSI : `(4,2), (5,5), (7,10), (10,15), (14,20)`. Ils matérialisent toute la diagonale observée.

Représentants MA : médoïde `(5,45)` et quatre cellules centrales proches `(3,45), (8,45), (5,60), (3,60)`. Le maximum de bord `(12,45)` est volontairement exclu.

## Résultats nominaux IS/OOS

Le max drawdown est exprimé avec le signe négatif du moteur.

| Mécanisme | Représentant | Période | Trades | PnL USD | PF | Sharpe | MaxDD USD | Win rate |
|---|---|---|---:|---:|---:|---:|---:|---:|
| MA | `ma_f3_s45` | IS | 239 | 1 126.5 | 1.128 | 0.437 | -1 634.5 | 37.2% |
| MA | `ma_f3_s45` | OOS | 244 | 27.0 | 1.004 | 0.024 | -701.5 | 41.4% |
| MA | `ma_f3_s60` | IS | 204 | 630.0 | 1.079 | 0.264 | -1 762.0 | 39.7% |
| MA | `ma_f3_s60` | OOS | 220 | 326.0 | 1.060 | 0.242 | -669.0 | 41.8% |
| MA | `ma_f5_s45` | IS | 206 | 1 199.0 | 1.154 | 0.480 | -1 250.5 | 40.3% |
| MA | `ma_f5_s45` | OOS | 192 | 580.0 | 1.125 | 0.423 | -545.5 | 46.4% |
| MA | `ma_f5_s60` | IS | 171 | 1 504.5 | 1.222 | 0.606 | -1 340.0 | 42.7% |
| MA | `ma_f5_s60` | OOS | 182 | 286.0 | 1.067 | 0.229 | -856.5 | 47.3% |
| MA | `ma_f8_s45` | IS | 167 | 1 985.5 | 1.311 | 0.814 | -1 063.5 | 45.5% |
| MA | `ma_f8_s45` | OOS | 168 | 437.0 | 1.108 | 0.330 | -870.5 | 49.4% |
| RSI | `rsi_diag_p4_e2` | IS | 1 | 58.5 | inf | 0.644 | 0.0 | 100.0% |
| RSI | `rsi_diag_p4_e2` | OOS | 1 | 5.5 | inf | 0.679 | 0.0 | 100.0% |
| RSI | `rsi_diag_p5_e5` | IS | 6 | 19.0 | 1.106 | 0.074 | -103.5 | 66.7% |
| RSI | `rsi_diag_p5_e5` | OOS | 5 | 117.5 | 5.434 | 0.739 | -26.5 | 80.0% |
| RSI | `rsi_diag_p7_e10` | IS | 11 | 452.5 | 2.534 | 0.837 | -147.0 | 63.6% |
| RSI | `rsi_diag_p7_e10` | OOS | 5 | 96.5 | 4.217 | 0.594 | -26.5 | 60.0% |
| RSI | `rsi_diag_p10_e15` | IS | 4 | 332.0 | 5.340 | 0.830 | -76.5 | 75.0% |
| RSI | `rsi_diag_p10_e15` | OOS | 4 | 30.0 | 2.132 | 0.424 | -26.5 | 75.0% |
| RSI | `rsi_diag_p14_e20` | IS | 1 | 156.5 | inf | 0.644 | 0.0 | 100.0% |
| RSI | `rsi_diag_p14_e20` | OOS | 1 | 18.5 | inf | 0.679 | 0.0 | 100.0% |

Les PF infinis et les Sharpes positifs des points RSI à un trade ne sont pas des preuves de robustesse.

## Diagonale RSI et percentile implicite

Le percentile est mesuré séparément dans les univers éligibles au filtre SMA200 : queue basse lorsque `close > SMA200`, queue haute lorsque `close < SMA200`.

| Représentant | Fréquence de queue IS | Fréquence de queue OOS | Crosses frais IS/OOS |
|---|---:|---:|---:|
| `(4,2)` | 0.008% | 0.008% | 1 / 1 |
| `(5,5)` | 0.061% | 0.075% | 6 / 6 |
| `(7,10)` | 0.129% | 0.100% | 12 / 5 |
| `(10,15)` | 0.046% | 0.050% | 5 / 5 |
| `(14,20)` | 0.008% | 0.008% | 1 / 1 |

La diagonale possède bien une interprétation en termes de rareté : augmenter la période RSI exige un seuil moins extrême en valeur absolue. En revanche, elle n'est pas un contour de quantile constant. Sa fréquence varie d'un facteur 16 entre le centre et les extrémités, et reste partout trop faible pour valider un sleeve.

## Stabilité locale

### Pullback RSI

- Part de voisins 3×3 positifs dans les deux périodes : 40–50% seulement.
- Le Sharpe médian OOS du voisinage est négatif pour `(4,2)` et `(7,10)`.
- Chaque voisinage contient des Sharpes OOS compris entre `-1.21` et `-1.63` au pire.
- Les cinq cellules positives suivent une diagonale étroite entourée de cellules négatives : il ne s'agit pas d'un plateau bidimensionnel.

### Crossover MA

- Le médoïde `(5,45)` et `(5,60)` ont 5 voisins positifs sur 9 ; les trois autres représentants en ont 4 sur 9.
- Sharpe médian OOS du voisinage : `0.024` pour le médoïde, `0.109` pour `(5,60)`, négatif pour `(3,45)` et `(8,45)`.
- Les pires voisins OOS vont de `-0.25` à `-0.76`.
- La composante positive est réelle, mais son centre n'est pas entouré d'une majorité forte de cellules performantes.

## Performance temporelle

Chaque cellule contient `PnL USD / Sharpe annuel`. 2021 et 2026 sont partielles.

| Représentant | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|
| `ma_f3_s45` | 473 / 1.33 | 1 349 / 0.99 | -696 / -0.82 | -225 / -0.35 | 343 / 0.44 | -91 / -0.38 |
| `ma_f3_s60` | 412 / 1.32 | 390 / 0.30 | -171 / -0.25 | -165 / -0.27 | 307 / 0.54 | 185 / 0.93 |
| `ma_f5_s45` | 501 / 1.53 | 1 286 / 0.95 | -587 / -0.77 | -131 / -0.25 | 458 / 0.72 | 253 / 1.26 |
| `ma_f5_s60` | 449 / 1.24 | 870 / 0.64 | 186 / 0.31 | -385 / -0.72 | 278 / 0.47 | 393 / 2.92 |
| `ma_f8_s45` | 766 / 2.18 | 1 061 / 0.80 | 159 / 0.26 | -428 / -0.82 | 469 / 0.71 | 397 / 2.77 |
| `rsi_diag_p4_e2` | 0 / n/a | 0 / n/a | 59 / 1.01 | 6 / 1.01 | 0 / n/a | 0 / n/a |
| `rsi_diag_p5_e5` | -32 / -0.53 | 106 / 1.29 | -55 / -0.48 | -8 / -0.25 | 126 / 1.23 | 0 / n/a |
| `rsi_diag_p7_e10` | 249 / 1.62 | 297 / 1.33 | -94 / -0.68 | -8 / -0.25 | 105 / 1.00 | 0 / n/a |
| `rsi_diag_p10_e15` | 35 / 0.38 | 297 / 1.38 | 0 / n/a | -8 / -0.25 | 38 / 1.08 | 0 / n/a |
| `rsi_diag_p14_e20` | 0 / n/a | 157 / 1.01 | 0 / n/a | 19 / 1.01 | 0 / n/a | 0 / n/a |

Les cinq variantes MA sont négatives en 2024. Leur OOS positif agrégé provient du rebond 2025 et, pour quatre variantes, du début 2026. Les sous-périodes semestrielles confirment des épisodes communs négatifs ; seulement 6 à 8 semestres sur 10 sont positifs selon le représentant.

Le Sharpe glissant 126 séances du MA est positif pendant 54–65% des fenêtres, mais atteint des minima de `-1.90` à `-2.71`. Le rolling Sharpe du RSI n'est pas interprétable : avec presque toutes les journées à zéro et quelques trades isolés, il produit des valeurs discrètes artificielles.

## Stress coûts et slippage OOS

| Représentant | Nominal | 2 ticks | 3 ticks | 2 ticks + commission ×2 |
|---|---:|---:|---:|---:|
| `ma_f3_s45` | 27 | -461 | -949 | -1 071 |
| `ma_f3_s60` | 326 | -114 | -554 | -664 |
| `ma_f5_s45` | 580 | 196 | -188 | -284 |
| `ma_f5_s60` | 286 | -78 | -442 | -533 |
| `ma_f8_s45` | 437 | 101 | -235 | -319 |
| `rsi_diag_p4_e2` | 6 | 4 | 2 | 1 |
| `rsi_diag_p5_e5` | 118 | 108 | 98 | 95 |
| `rsi_diag_p7_e10` | 97 | 87 | 77 | 74 |
| `rsi_diag_p10_e15` | 30 | 22 | 14 | 12 |
| `rsi_diag_p14_e20` | 19 | 17 | 15 | 14 |

Les points RSI restent arithmétiquement positifs sous stress parce qu'ils ne négocient presque jamais ; ce résultat ne résout pas le défaut d'échantillon.

Pour le MA, le médoïde `(5,45)` survit à deux ticks mais échoue à trois ticks et avec commission doublée. Aucun représentant ne reste positif sous tous les scénarios préspécifiés.

## Conclusion

### `short_horizon_pullback_reversion`

La diagonale est explicable comme adaptation grossière du seuil à la contraction de la distribution du RSI lorsque sa période augmente. Elle ne représente toutefois ni un quantile constant ni une zone robuste. Le nombre de trades est deux ordres de grandeur sous le garde-fou de 100 trades par période. **Décision : abandon comme sleeve MCL.**

### `ma_crossover_trend`

Le plateau nominal n'est pas un pic isolé, mais son avantage est trop petit par rapport aux coûts et dépend d'un régime temporel : année 2024 négative pour 5/5 représentants, puis récupération récente. Le médoïde échoue au garde-fou de résistance à tous les scénarios de coûts. **Décision : abandon comme sleeve MCL, sans nouvelle optimisation.**

Cette décision archive la branche MCL actuelle ; elle ne supprime ni ne modifie les mécanismes frozen communs utilisés sur les autres actifs.

## Livrables

- `period_metrics.csv` : métriques IS/OOS nominales et stressées.
- `yearly_metrics.csv`, `halfyear_metrics.csv`, `rolling_sharpe_126.csv`.
- `local_stability.csv`, `rsi_extreme_quantiles.csv`.
- `trades/` : transactions nominales des dix représentants.
- `charts/` : rolling Sharpe, PnL annuel, stress coûts et percentile implicite RSI.
