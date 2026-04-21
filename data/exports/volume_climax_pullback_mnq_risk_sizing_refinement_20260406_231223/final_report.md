# Volume Climax Pullback MNQ Risk Sizing - Refinement Report

## Scope
- Symbol: `MNQ` only.
- Base alpha reused unchanged: `dynamic_exit_atr_target_1p0_ts2_vq0p95_bf0p5_ra1p2`.
- Reference V3 run: `D:\Business\Trading\VSCODE\algo-trading-intraday-research\data\exports\volume_climax_pullback_v3_run\volume_climax_pullback_v3_20260402_184702`.
- Dataset: `repo latest MNQ source`.
- Sessions: full `1747` | IS `1222` | OOS `525`.
- Sizing logic unchanged from the prior campaign. Only the local grid around the previous winner was refined.

## Prop Score
- Formula on OOS: `prop_score = 4 * min(net_pnl/3000, 2) + 3 * min(sharpe, 2.5) - 3 * min(maxDD/2000, 3) - 5 * min(max_daily_DD/1000, 3) - 2 * min(worst_trade/200, 3) - 0.5 * nb_days_below_-500 - 6 * 1[pass=False]`.
- Interpretation: reward enough OOS profit to clear a 50k target, reward clean Sharpe, penalize drawdown, penalize daily drawdown strongly, and penalize any non-pass configuration heavily.

## OOS Winner
- Punctual OOS winner by `prop_score`: `risk_pct_0p0015__max_contracts_4__skip_trade_if_too_small_true`.
- OOS metrics: net `6151.67` | CAGR `5.77%` | Sharpe `1.541` | maxDD `497.50` | max daily DD `571.50` | prop_score `8.16`.
- Versus previous winner tag: net `-667.90` | Sharpe `-0.199` | maxDD `-139.00`.

```text
                                         campaign_variant_name  oos_net_pnl_usd  oos_sharpe  oos_max_drawdown_usd  oos_max_daily_drawdown_usd  oos_prop_score  oos_pass_target_3000_usd_without_breaching_2000_dd
risk_pct_0p0015__max_contracts_4__skip_trade_if_too_small_true         6151.675    1.540608               497.500                     571.500        8.158074                                                True
risk_pct_0p0025__max_contracts_2__skip_trade_if_too_small_true         5651.975    1.837363               548.500                     548.500        8.082805                                                True
risk_pct_0p0015__max_contracts_5__skip_trade_if_too_small_true         6965.875    1.524516               535.000                     609.000        7.751047                                                True
risk_pct_0p0025__max_contracts_3__skip_trade_if_too_small_true         6819.575    1.739649               636.500                     644.950        7.599446                                                True
risk_pct_0p0015__max_contracts_6__skip_trade_if_too_small_true         7770.075    1.497854               581.725                     655.725        7.352351                                                True
risk_pct_0p0025__max_contracts_4__skip_trade_if_too_small_true         7920.800    1.691914               653.000                     707.000        7.041243                                                True
risk_pct_0p0015__max_contracts_3__skip_trade_if_too_small_true         4962.200    1.542505               474.175                     548.175        6.941643                                                True
risk_pct_0p0025__max_contracts_5__skip_trade_if_too_small_true         9357.600    1.687010               732.000                     738.550        6.750280                                                True
risk_pct_0p0025__max_contracts_6__skip_trade_if_too_small_true        10426.075    1.656324               732.000                     742.600        6.637972                                                True
risk_pct_0p0020__max_contracts_6__skip_trade_if_too_small_true         9409.550    1.577245               725.325                     799.325        6.447123                                                True
```

## Robust Zone
- Recommended cluster id: `1` | size `2` | scale `zone_etruite_mais_non_isolee`.
- Zone range: risk_pct `0.0015` -> `0.0015` | max_contracts `4` -> `5`.
- Zone center: risk_pct `0.0015` | max_contracts `4.50`.
- Zone means: net `6558.77` | Sharpe `1.533` | maxDD `516.25` | prop_score `7.95`.
- Best point inside the zone: `risk_pct_0p0015__max_contracts_4__skip_trade_if_too_small_true`.

## Requested Readout
1. Le gagnant ponctuel OOS: `risk_pct_0p0015__max_contracts_4__skip_trade_if_too_small_true`.
2. La zone robuste OOS: cluster `1` couvrant risk_pct `0.0015` -> `0.0015` et cap `4` -> `5`.
3. Impact marginal de risk_pct autour de 0.25%: voir la coupe `max_contracts=3` ci-dessous. La question utile est de savoir si le score reste propre quand on s'eloigne de `0.0025`.
4. Impact marginal du cap `max_contracts` autour de 3: voir la coupe `risk_pct=0.0025` ci-dessous. La lecture utile est la vitesse de deterioration du drawdown et du `prop_score` quand on desserre le cap.
5. Distribution des tailles pour la meilleure variante: avg `2.36` | median `2.00` | 1c `41.8%` | 2c `14.9%` | 3+c `43.3%`.
6. Trades skippes pour la meilleure variante: `35` soit `34.3%` des tentatives.
7. Robustesse: `zone_etruite_mais_non_isolee`.
8. Reference de recherche: `risk_pct_0p0015__max_contracts_4__skip_trade_if_too_small_true` | mode prop-safe: `risk_pct_0p0015__max_contracts_4__skip_trade_if_too_small_true` | mode plus agressif mais defendable: `risk_pct_0p0025__max_contracts_6__skip_trade_if_too_small_true`.
9. Verdict final: `retenir_une_zone_parametrique`.

### Slice: risk_pct around 0.25% with max_contracts=3
```text
                                         campaign_variant_name  risk_pct  max_contracts  oos_net_pnl_usd  oos_sharpe  oos_max_drawdown_usd  oos_max_daily_drawdown_usd  oos_prop_score
risk_pct_0p0015__max_contracts_3__skip_trade_if_too_small_true    0.0015            3.0         4962.200    1.542505               474.175                     548.175        6.941643
risk_pct_0p0020__max_contracts_3__skip_trade_if_too_small_true    0.0020            3.0         5553.500    1.601962               732.675                     806.675        5.938166
risk_pct_0p0025__max_contracts_3__skip_trade_if_too_small_true    0.0025            3.0         6819.575    1.739649               636.500                     644.950        7.599446
risk_pct_0p0030__max_contracts_3__skip_trade_if_too_small_true    0.0030            3.0         6792.475    1.616171               946.500                     946.500        5.076264
risk_pct_0p0035__max_contracts_3__skip_trade_if_too_small_true    0.0035            3.0         7622.900    1.663280               955.475                     985.975        4.586751
risk_pct_0p0040__max_contracts_3__skip_trade_if_too_small_true    0.0040            3.0         8232.975    1.691477              1100.500                    1100.500        3.761180
```

### Slice: max_contracts around 3 with risk_pct=0.0025
```text
                                         campaign_variant_name  risk_pct  max_contracts  oos_net_pnl_usd  oos_sharpe  oos_max_drawdown_usd  oos_max_daily_drawdown_usd  oos_prop_score
risk_pct_0p0025__max_contracts_2__skip_trade_if_too_small_true    0.0025            2.0         5651.975    1.837363                 548.5                      548.50        8.082805
risk_pct_0p0025__max_contracts_3__skip_trade_if_too_small_true    0.0025            3.0         6819.575    1.739649                 636.5                      644.95        7.599446
risk_pct_0p0025__max_contracts_4__skip_trade_if_too_small_true    0.0025            4.0         7920.800    1.691914                 653.0                      707.00        7.041243
risk_pct_0p0025__max_contracts_5__skip_trade_if_too_small_true    0.0025            5.0         9357.600    1.687010                 732.0                      738.55        6.750280
risk_pct_0p0025__max_contracts_6__skip_trade_if_too_small_true    0.0025            6.0        10426.075    1.656324                 732.0                      742.60        6.637972
```
