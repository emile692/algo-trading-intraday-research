# MNQ ORB VIX/VVIX Validation Report

## Baseline

- Baseline ORB conservee: OR30 / direction `both` / RR `2.00` / VWAP confirmation `True`.
- Filtre ATR conserve tel quel via l'ensemble `majority_50` calibre uniquement sur IS.
- Sizing retire pour cette campagne: backtest rerun en `fixed_contracts=1` pour isoler le bloc VIX/VVIX.
- Dataset: `MNQ_c_0_1m_20260321_094501.parquet` | sessions IS/OOS d'origine: `1222` / `525`.

## Data Coverage

- Sessions baseline ATR selectionnees: `1134`.
- Sessions avec contexte VIX/VVIX exploitable: `1120`.
- Sessions exclues faute de contexte daily t-1: `14`.
- Fin de couverture source VIX: `2026-02-27` | fin de couverture source VVIX: `2026-03-25`.

## Baseline Readout

- Baseline OOS sur fenetre couverte: net pnl `10206.08` | Sharpe `2.036` | Sortino `2.341` | PF `1.355` | expectancy `30.65` | maxDD `-3752.76`.
- Baseline OOS trades/days: `333` trades | `333` jours trades | exposition temps `69.22%`.
- Univers effectivement teste: `1120` trades / sessions baseline avec contexte daily exploitable.

## Main Findings

- Meilleur candidat: `filter_drop_low__vvix_pct_63_t1` | verdict `robust_positive` | OOS Sharpe delta `+0.192` | OOS PF delta `+0.136` | OOS expectancy delta `+12.39` | retention pnl `97.4%` | trade coverage `69.4%` | maxDD improvement `44.1%`.
- Family / feature la plus prometteuse: `vvix_percentile` / `vvix_pct_63_t1` via `filter_drop_low__vvix_pct_63_t1` (`robust_positive`).
- Meilleur VIX standalone: `filter_drop_low__vix_pct_63_t1` (`protective_filter`), validation score `0.850`, OOS Sharpe delta `-0.072`.
- Meilleur VVIX standalone: `regime_low__vvix_level_t1` (`protective_filter`), validation score `2.485`, OOS Sharpe delta `+0.110`.
- Meilleur regime croise: `regime_elevated_stressed__vix_x_vvix` (`insufficient_oos`), validation score `0.239`, maxDD improvement `90.2%`.
- Driver principal: `better_day_selection` | hit delta `-0.4 pts`, stop-hit delta `-1.3 pts`, avg win delta `+27.16`, avg loss delta `-1.46`.
- Sous-groupe directionnel le plus sensible: `short` pour `filter_drop_low__vvix_pct_126_t1` (expectancy delta `+18.66`, hit delta `+3.3 pts`).
- Sous-groupe timing le plus sensible: `late` pour `filter_drop_low__vvix_pct_63_t1` (expectancy delta `+27.85`, hit delta `+5.7 pts`).
- Variants robustes OOS: `3` | protecteurs mais couteux: `30` | faux positifs IS-only: `4`.

## Reponses Directes

- Le VIX apporte-t-il un edge standalone comme filtre de regime ? Oui, avec confirmation OOS suffisante.
- Le VVIX ajoute-t-il une information complementaire utile au VIX ? Oui, au moins sous forme de regime croise / ratio ou de filtre complementaire.
- Le gain eventuel vient-il d'une meilleure selection des jours, d'une baisse du drawdown, d'une reduction des faux breakouts ou d'une vraie hausse du PF / Sharpe ? `better_day_selection`.
- Les resultats justifient-ils une phase 2 avec meilleur filtre VIX/VVIX + sizing 3-state ? Oui.
- L'interet est-il plutot alpha, filtrage defensif ou simple controle de regime ? `defensive_filter`.

## Notes Methodologiques

- Les regles d'entree/sortie ORB, les couts, le slippage, la commission et le filtre ATR baseline restent inchanges.
- Les features VIX/VVIX sont construites strictement en daily t-1; aucune cloture du jour courant n'est utilisee pour decider du trade du meme jour.
- Tous les seuils et buckets sont calibres sur IS seulement puis figes sur OOS.
- Cette campagne n'inclut aucun sizing dynamique 3-state.
- OR15 vs OR30 n'est pas re-optimise ici: la baseline est gardee figee sur sa reference tradable actuelle.

## Exports

- `selected_trade_vix_vvix_features.csv`
- `regime_summary.csv`
- `screening_summary.csv`
- `validation_summary.csv`
- `full_variant_results.csv`
- `feature_importance_like_summary.csv`
- `interaction_summary.csv`
- `final_verdict.json`
- `variants/<variant>/...`
