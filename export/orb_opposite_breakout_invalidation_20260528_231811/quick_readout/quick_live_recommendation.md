# Quick Live Readout

## Scope

- Source unique: fichiers existants dans `export/orb_opposite_breakout_invalidation_20260528_231811`.
- Aucun backtest relance. Aucun recalcul de campagne.
- Verdict global principal construit sur `MNQ + MES`.
- `M2K` est inclus comme check complementaire car les configs baseline / close_1m / n_closes necessaires sont presentes.
- `MGC` est exclu du verdict global principal pour garder un readout centre sur `MNQ + MES`, comme demande.

## Verdict

- Verdict clair: **invalider apres downside first breakout**.
- Gagnant recherche sur `MNQ + MES`: `invalidate_on_opposite_touch__buffer_0`.
- Gagnant invalidation sur `MNQ`: `invalidate_on_opposite_touch__buffer_0`.
- Meilleur reclaim separe sur `MNQ + MES`: `allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_false`.
- Reference whole-basket existante dans `ranking_robust.csv`: `invalidate_on_opposite_close_1m__buffer_1`.

## Answers

- La baseline gagne-t-elle de l'argent sur les journees ou le first breakout est downside ? Non sur `MNQ + MES`: net `-15253.00`, Sharpe `-1.221`, drawdown `-17674.75`.
- Les trades apres downside first breakout ameliorent-ils ou degradent-ils Sharpe / PnL / drawdown ? Ils degradent nettement la baseline sur `MNQ + MES`: subset downside net `-15253.00` contre `23862.25` hors downside, Sharpe `-1.221` contre `1.041`, drawdown `-17674.75` contre `-5496.50`.
- Quelle config d'invalidation est la meilleure pour `MNQ` ? `invalidate_on_opposite_touch__buffer_0` avec net `18677.50`, Sharpe `1.037`, maxDD `-2442.00`.
- Quelle config est la plus robuste sur `MNQ + MES` ? `invalidate_on_opposite_touch__buffer_0` avec net `25137.50`, Sharpe `1.148`, maxDD `-4985.00`.
- Recommandation live pour le repo execution: garder le moteur ORB inchange, ajouter une policy d'invalidation simple en option, et deployer d'abord une variante conservative `close_1m / buffer 0` plutot que d'embarquer une logique reclaim dans la strategie principale.

## Baseline Vs Best Invalidation

- Baseline `MNQ + MES`: net `8609.25`, Sharpe `0.251`, maxDD `-8316.00`.
- Best invalidation `MNQ + MES`: `invalidate_on_opposite_touch__buffer_0` net `25137.50`, Sharpe `1.148`, maxDD `-4985.00`.
- Delta net: `16528.25`.
- Delta Sharpe: `0.897`.
- Drawdown improvement: `3331.00`.

## Reclaim Read

- Best reclaim `MNQ + MES`: `allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_false` net `8734.75`, Sharpe `0.255`, maxDD `-8190.50`.
- Conclusion reclaim: legerement meilleur que la baseline brute, mais tres inferieur a la meilleure invalidation simple. Ce n'est pas la recommandation live principale.

## Runtime Context

- backtest/evaluation: `1207.260s`
- writing_exports: `770.231s`
- preprocessing/session_features: `492.088s`

## Recommendation YAML

```yaml
opposite_breakout_policy:
  enabled: true
  mode: invalidate_for_day
  confirmation: close_1m
  buffer_ticks: 0
  confirm_bars: 1
  allow_reclaim: false
```
