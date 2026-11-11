# MCL — campagne brute des 16 mécanismes frozen

## Protocole

- Données Databento `MCL.v.0`, OHLCV 1 minute, du lancement du produit au 24 septembre 2026 exclusif.
- Coût Databento : **6,455839 USD**, un seul appel payant.
- Fenêtres : IS `2021-07-12–2023-12-31`, OOS `2024-01-01–2026-03-19`, full `2021-07-12–2026-03-19`.
- Une spécification canonique par mécanisme, aucune grille, aucun réglage après performance.
- Contrat : 100 barils, tick 0,01, valeur du tick 1 USD, commission 1,25 USD/côté, slippage 1 tick/côté.
- Session énergie : `09:00–14:30 America/New_York`; une seule unité, aucun pyramiding.
- Seules les sessions core complètes sont conservées; aucun bar synthétique. Le 18 septembre 2024 est explicitement exclu après l'avertissement de qualité Databento.
- Les 16 fichiers frozen sources sont restés inchangés.

## Résultats IS/OOS

| Mécanisme | Trades IS | PnL IS | Sharpe IS | PF IS | Trades OOS | PnL OOS | Sharpe OOS | PF OOS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| atr_trend_flip | 379 | 332 | 0,11 | 1,02 | 350 | 251 | 0,12 | 1,02 |
| channel_extreme_reversion | 1 400 | -7 248 | -2,00 | 0,78 | 1 315 | -5 061 | -2,53 | 0,74 |
| compression_breakout | 154 | -1 035 | -0,91 | 0,75 | 113 | -505 | -0,68 | 0,73 |
| gap_continuation | 15 | 893 | 0,41 | 1,68 | 13 | -906 | -1,22 | 0,19 |
| gap_fade | 500 | -2 646 | -0,99 | 0,82 | 446 | -1 943 | -1,36 | 0,78 |
| ma_crossover_trend | 247 | 691 | 0,27 | 1,07 | 233 | -254 | -0,15 | 0,96 |
| momentum_confirmed_breakout | 107 | -2 212 | -1,03 | 0,65 | 109 | -1 265 | -1,03 | 0,67 |
| range_structure_reversion | 27 | -62 | -0,09 | 0,93 | 11 | 241 | 0,38 | 1,92 |
| regression_equilibrium_reversion | 111 | -702 | -0,31 | 0,87 | 100 | -682 | -0,72 | 0,75 |
| rolling_range_breakout | 259 | -3 169 | -0,93 | 0,77 | 236 | -3 437 | -1,35 | 0,62 |
| session_range_breakout | 597 | -3 895 | -1,51 | 0,79 | 524 | -2 883 | -2,09 | 0,72 |
| short_horizon_pullback_reversion | 532 | -2 760 | -1,03 | 0,81 | 517 | -2 403 | -1,44 | 0,74 |
| statistical_zscore_reversion | 149 | -146 | -0,05 | 0,98 | 122 | 1 837 | 0,81 | 1,58 |
| trend_pullback_ma_structure | 611 | -3 636 | -0,86 | 0,86 | 552 | -4 315 | -1,72 | 0,73 |
| volatility_breakout | 275 | 1 258 | 0,41 | 1,12 | 239 | -1 157 | -0,66 | 0,83 |
| vwap_equilibrium_reversion | 2 056 | -12 986 | -3,88 | 0,68 | 1 641 | -6 200 | -2,60 | 0,71 |

## Lecture

- `atr_trend_flip` est le seul mécanisme positif IS et OOS, mais PF ≈1,02 et Sharpe ≈0,1 : marge économique trop faible après coûts.
- `statistical_zscore_reversion` est intéressant OOS (Sharpe 0,81), mais échoue IS. Le retenir sur cette seule amélioration serait une sélection post-hoc.
- `gap_continuation`, `ma_crossover_trend` et `volatility_breakout` sont positifs IS puis négatifs OOS.
- `range_structure_reversion` ne compte que 11 trades OOS et était négatif IS.
- Les autres mécanismes sont négatifs dans les deux segments.

## Décision de phase 1

**Aucune sleeve MCL n'est retenue.** Aucun candidat ne combine avantage économique suffisant, stabilité IS/OOS et taille d'échantillon crédible. En conséquence, aucun test d'ajout au portefeuille MNQ/MGC et aucune optimisation locale ne sont lancés.

Cette conclusion vaut pour la transposition directe des 16 mécanismes frozen sur MCL, pas pour toute hypothèse énergie possible. Une éventuelle suite devrait être un audit de frameworks spécifiquement énergie, séparé et préspecifié.
