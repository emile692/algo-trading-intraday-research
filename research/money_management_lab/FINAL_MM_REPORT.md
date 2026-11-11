# Final Money Management Report

## Protocole final

- Signaux d'entrée, paramètres alpha, coûts, exécution et quantité **1 MNQ** inchangés.
- IS : **2019-05-06 → 2023-12-31** ; Validation : **2024-01-01 → 2026-03-19**.
- Seuls `time_stop` pour volatility breakout et `stop_atr` pour ATR trend flip ont été rejoués.
- Les choix finaux ci-dessous proviennent de l'instruction explicite de verrouillage donnée après revue des études IS/Validation.
- Aucun nouveau paramètre et aucun nouveau backtest ne sont introduits par ce verrouillage.
- SHA-256 grille : `52b0bc3af0aae437b8e94c3d357cb01955f5d0124e246e22b3a8d93a3dbf4ebe`.
- SHA-256 décisions de verrouillage : `11b1b467fcb4e34d902232eca1407a11f176afc276d55b6dad3a5983f303b81e`.
- SHA-256 règles finales : `faed04bf191b2dc1cea41e2b005a9c5dcec9a5149a914d270a370dd171444655`.

## compression_breakout

Statut final : **`MM_LOCKED_BASELINE`**.

Sortie baseline verrouillee explicitement par l'utilisateur.

La sortie baseline est conservée définitivement, conformément à la décision préalable. Aucun nouveau money management n'a été exécuté.

## volatility_breakout

Statut final : **`MM_LOCKED_MODIFIED`**.

Time stop 240 minutes verrouille explicitement par l'utilisateur apres revue IS/Validation.

Règle verrouillée : **`time_stop`**, `max_holding_minutes = 240`.

Valeurs testées : `[60, 90, 120, 150, 180, 210, 240, 270, 300, 360]`.
Zones éligibles détectées : `[[120.0, 150.0], [240.0, 270.0]]`.

| Valeur | Sharpe IS/VAL | Expectancy IS/VAL | PF IS/VAL | DD IS/VAL | Trades IS/VAL | Éligible |
|---:|---:|---:|---:|---:|---:|---:|
| 60 | -0.01/1.04 | -0.34/21.79 | 0.99/1.39 | -3,532/-1,969 | 608/291 | non |
| 90 | 0.21/0.94 | 2.56/22.17 | 1.05/1.32 | -3,440/-3,065 | 608/291 | non |
| 120 | 0.54/1.12 | 7.54/26.29 | 1.14/1.36 | -2,714/-3,230 | 608/291 | oui |
| 150 | 0.48/0.99 | 7.08/27.07 | 1.12/1.33 | -3,189/-3,817 | 608/291 | oui |
| 180 | 0.37/0.99 | 5.67/29.25 | 1.09/1.34 | -3,416/-4,318 | 608/291 | non |
| 210 | 0.47/1.02 | 7.64/29.86 | 1.12/1.33 | -4,384/-5,206 | 608/291 | non |
| 240 | 0.68/1.26 | 12.15/38.53 | 1.19/1.44 | -4,656/-3,554 | 608/291 | oui |
| 270 | 0.83/1.14 | 14.98/35.59 | 1.24/1.38 | -4,107/-3,763 | 608/291 | oui |
| 300 | 0.83/0.86 | 15.75/26.58 | 1.24/1.28 | -4,231/-4,419 | 608/291 | non |
| 360 | 0.62/0.99 | 12.45/31.25 | 1.18/1.32 | -4,870/-4,028 | 608/291 | non |

## atr_trend_flip

Statut final : **`MM_LOCKED_MODIFIED`**.

Stop 1.0 ATR verrouille explicitement par l'utilisateur apres revue IS/Validation.

Règle verrouillée : **`atr_stop`**, `stop_atr = 1.0`.

Valeurs testées : `[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]`.
Zones éligibles détectées : `[[1.0]]`.

| Valeur | Sharpe IS/VAL | Expectancy IS/VAL | PF IS/VAL | DD IS/VAL | Trades IS/VAL | Éligible |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5 | -0.42/0.51 | -2.98/6.22 | 0.92/1.13 | -7,628/-3,870 | 1522/689 | non |
| 0.75 | 0.20/0.93 | 1.39/13.06 | 1.03/1.21 | -4,342/-3,203 | 1495/672 | non |
| 1 | 0.34/0.83 | 2.81/12.56 | 1.05/1.17 | -3,574/-4,056 | 1460/659 | oui |
| 1.25 | 0.14/0.96 | 1.09/16.09 | 1.02/1.20 | -5,122/-4,090 | 1419/635 | non |
| 1.5 | 0.29/0.74 | 2.73/13.04 | 1.04/1.15 | -4,742/-4,085 | 1354/608 | non |
| 1.75 | 0.38/0.66 | 3.95/11.94 | 1.05/1.12 | -6,683/-4,452 | 1294/589 | non |
| 2 | 0.38/0.57 | 4.18/10.96 | 1.05/1.11 | -7,382/-4,233 | 1234/553 | non |

## Verrouillage terminal

Cette étape clôt le tuning des sorties pour ces trois mécanismes. Toute étude ultérieure devra traiter les règles ci-dessus comme des spécifications figées.

## Livrables

- `final_mm_results.csv`
- `final_mm_locked_exit_rules.json` et son SHA-256
- dossier `final_mm_curves/`
- `run_final_mm_study.py`
