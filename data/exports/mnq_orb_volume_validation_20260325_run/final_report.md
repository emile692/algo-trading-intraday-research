# MNQ ORB Volume Validation Report

## Baseline

- Baseline ORB conservee: OR30 / direction `long` / RR `1.50` / VWAP confirmation `True`.
- Filtre ATR conserve tel quel via l'ensemble `majority_50` calibre uniquement sur IS.
- Sizing retire pour cette campagne: backtest rerun en `fixed_contracts=1` pour isoler l'alpha signal volume.
- Dataset: `MNQ_c_0_1m_20260321_094501.parquet` | sessions IS/OOS: `1222` / `525`.

## Baseline Readout

- Baseline OOS: net pnl `3279.58` | Sharpe `1.122` | Sortino `0.593` | PF `1.394` | expectancy `24.66` | maxDD `-1195.58`.
- Baseline OOS trades/days: `133` trades | `133` jours trades | exposition temps `14.98%`.
- Trade universe pour l'etude volume: `427` sessions baseline ATR selectionnees et effectivement tradables.

## Main Findings

- Meilleur candidat volume: `regime_mid__volume_percentile_prev_20` | verdict `mixed_positive` | OOS Sharpe delta `-0.014` | OOS expectancy delta `+4.41` | retention pnl `84.2%` | trade coverage `71.4%` | maxDD improvement `8.5%`.
- Famille / feature la plus prometteuse: `participation` / `breakout_same_minute_rvol_20` via `regime_high__breakout_same_minute_rvol_20` (`insufficient_oos`).
- Driver principal: hit rate delta `-1.1 pts`, stop-hit delta `+2.1 pts`, avg win delta `+7.57`, avg loss delta `+7.54`.
- Sous-groupe directionnel le plus sensible: `long` pour `filter_drop_low__breakout_vol_vs_or_mean` (expectancy delta `+3.80`, hit delta `+0.2 pts`).
- Sous-groupe timing le plus sensible: `mid` pour `filter_drop_low__breakout_vol_vs_or_mean` (expectancy delta `+19.49`, hit delta `+5.8 pts`).
- Variants robustes OOS: `0` | protecteurs mais couteux: `1` | faux positifs IS-only: `2`.

## Reponses Directes

- Le volume apporte-t-il un edge standalone par rapport a la baseline ORB + ATR ? Non, pas de facon assez robuste en OOS; au mieux un signal exploratoire ou un filtre partiel.
- Cet edge est-il surtout un filtre de selection ou un vrai moteur de performance ? Plutot un filtre de selection.
- Les gains viennent-ils surtout du hit rate, du drawdown ou d'une reduction des faux breakouts ? Driver principal: hit rate delta `-1.1 pts`, stop-hit delta `+2.1 pts`, avg win delta `+7.57`, avg loss delta `+7.54`.
- Les resultats sont-ils assez robustes pour une phase 2 volume + 3-state sizing ? Non, mieux vaut ne pas combiner avec le 3-state tant que le gain volume seul reste ambigu.

## Notes Methodologiques

- Tous les seuils volume sont calibres sur IS seulement puis reappliques tels quels en OOS.
- Les features breakout utilisent uniquement des donnees connues a la cloture de la barre de signal ou des historiques strictement anterieurs.
- Les references same-minute et cum-volume historiques utilisent `shift(1)` avant rolling pour exclure la session courante.
- Interaction OR15 vs OR30 non testee ici car la baseline est runnee en OR fixe unique.

## Exports

- `selected_trade_volume_features.csv`
- `feature_bucket_summary.csv`
- `screening_summary.csv`
- `validation_summary.csv`
- `full_variant_results.csv`
- `feature_importance_like_summary.csv`
- `interaction_summary.csv`
- `final_verdict.json`
- `variants/<variant>/...`
