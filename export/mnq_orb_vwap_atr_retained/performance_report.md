# MNQ ORB + VWAP + ATR — rapport de performance

Version corrigée le 18 septembre 2026 : stop et objectif sont évalués seulement avant la sortie de 16:00 New York. L'audit historique du cutoff n'a pas été récupéré ; le hash du journal de trades corrigé est toutefois reproduit dans `summary.json`.

## Périmètre et méthode

- Stratégie retenue : `orb_vwap_atr`, appelée `baseline_fixed` dans la campagne de recherche.
- OR15, long-only, clôture de breakout, entrée à l'ouverture de la barre suivante, VWAP continu et ensemble ATR(14) avec vote 0,50.
- Stop sous l'opening range avec tampon de 2 ticks ; objectif 2R ; sortie horaire à 16:00 (New York) ; au plus un trade par jour.
- Fichier MNQ 1 minute : `data\processed\parquet\MNQ_c_0_1m_20260321_094501.parquet` ; SHA-256 `3d67f747ca7dfab84d97cdb5e8356610b0ca5ad0d97c4592e2d1b11b9eebf174`.
- IS : 2019-05-05 au 2024-02-27 (1500 séances) ; OOS : 2024-02-28 au 2026-03-20 (643 séances). Découpage chronologique 70/30.
- Capital de référence : 50,000 $ ; risque nominal : 0.50 % par trade.
- Coûts du backtest : 1.25 $ de commission par côté et 1 tick de slippage par côté ; tick de 0.25 point.
- Même génération de signaux, jeu de données et séparation IS/OOS que la campagne ORB existante ; sorties recoupées avec l'audit du cutoff. L'export historique `baseline_fixed` précède cette correction.

## Résultats nets du backtest

| Période | Trades | PnL net ($) | Sharpe | Drawdown max ($) | Profit factor | Taux gagnant | Espérance/trade ($) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| IS | 584 | 12,524.50 | 0.962 | -2,061.50 | 1.244 | 49.0 % | 21.45 |
| OOS | 239 | 6,013.00 | 1.153 | -928.50 | 1.318 | 49.0 % | 25.16 |
| Ensemble | 823 | 18,537.50 | 1.016 | -2,061.50 | 1.264 | 49.0 % | 22.52 |

## Variantes historiques

Les résultats de la variante complète, de la comparaison 3-state et des autres campagnes n'ont pas été rejoués après cette correction. Leur comparaison chiffrée avec la référence corrigée est suspendue.

## Limites de lecture

- Résultats historiques sur un seul jeu de données MNQ ; aucune performance future ou réussite de challenge n'en découle.
- Les coûts sont fixes dans le moteur de recherche ; les fills, la latence, le spread et les incidents broker réels peuvent différer.
- Le backtest risque 0,50 % de 50 000 $, soit 250 $ nominaux par trade. La configuration d'exécution risque 0,50 % d'une base nominale de 150 000 $, soit 750 $ avant limites et arrondis. Ces PnL ne sont donc pas une projection de la configuration d'exécution ; comparer signaux et ordres sur les mêmes barres avant toute extrapolation.
- Le journal des trades et les métriques machine lisibles accompagnent ce rapport pour permettre l'audit.

## Reproduction

Depuis la racine du dépôt : `python scripts/report_mnq_orb_vwap_atr.py`.
