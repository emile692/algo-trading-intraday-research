# Volume Climax Pullback M2K Risk Sizing - Refinement Report

## Scope
- Symbol: `M2K` only.
- Base alpha reused unchanged: `dynamic_exit_atr_target_1p0_ts4_vq0p95_bf0p5_ra1p2`.
- Reference V3 run: `D:\Business\Trading\VSCODE\algo-trading-intraday-research\data\exports\volume_climax_pullback_v3_run\volume_climax_pullback_v3_20260402_184702`.
- Dataset: `repo latest M2K source`.
- Sessions: full `1747` | IS `1222` | OOS `525`.
- Sizing logic unchanged from the prior campaign. Only the local grid around the previous winner was refined.

## Prop Score
- Formula on OOS: `prop_score = 4 * min(net_pnl/3000, 2) + 3 * min(sharpe, 2.5) - 3 * min(maxDD/2000, 3) - 5 * min(max_daily_DD/1000, 3) - 2 * min(worst_trade/200, 3) - 0.5 * nb_days_below_-500 - 6 * 1[pass=False]`.
- Interpretation: reward enough OOS profit to clear a 50k target, reward clean Sharpe, penalize drawdown, penalize daily drawdown strongly, and penalize any non-pass configuration heavily.

## OOS Winner
- Punctual OOS winner by `prop_score`: `risk_pct_0p0030__max_contracts_6__skip_trade_if_too_small_true`.
- OOS metrics: net `5543.43` | CAGR `5.21%` | Sharpe `1.742` | maxDD `706.80` | max daily DD `706.80` | prop_score `6.35`.
- Versus previous winner tag: net `+2499.05` | Sharpe `+0.033` | maxDD `+305.30`.

```text
                                         campaign_variant_name  oos_net_pnl_usd  oos_sharpe  oos_max_drawdown_usd  oos_max_daily_drawdown_usd  oos_prop_score  oos_pass_target_3000_usd_without_breaching_2000_dd
risk_pct_0p0030__max_contracts_6__skip_trade_if_too_small_true         5543.425    1.742289                 706.8                       706.8        6.348901                                                True
risk_pct_0p0020__max_contracts_6__skip_trade_if_too_small_true         4432.950    1.685967                 556.5                       556.5        6.151252                                                True
risk_pct_0p0030__max_contracts_5__skip_trade_if_too_small_true         4847.375    1.751826                 613.3                       613.3        6.057194                                                True
risk_pct_0p0020__max_contracts_5__skip_trade_if_too_small_true         4068.350    1.715262                 518.0                       518.0        6.028252                                                True
risk_pct_0p0035__max_contracts_6__skip_trade_if_too_small_true         5750.050    1.720594                 742.5                       742.5        6.002265                                                True
risk_pct_0p0020__max_contracts_4__skip_trade_if_too_small_true         3601.800    1.742719                 456.0                       456.0        5.971558                                                True
risk_pct_0p0040__max_contracts_5__skip_trade_if_too_small_true         5187.075    1.734240                 649.0                       649.0        5.900319                                                True
risk_pct_0p0040__max_contracts_6__skip_trade_if_too_small_true         5919.275    1.718586                 742.5                       742.5        5.881874                                                True
risk_pct_0p0025__max_contracts_5__skip_trade_if_too_small_true         4488.125    1.703919                 588.5                       588.5        5.845672                                                True
risk_pct_0p0030__max_contracts_4__skip_trade_if_too_small_true         4052.175    1.736725                 519.8                       519.8        5.634374                                                True
```

## Robust Zone
- Recommended cluster id: `2` | size `2` | scale `zone_etruite_mais_non_isolee`.
- Zone range: risk_pct `0.0030` -> `0.0030` | max_contracts `5` -> `6`.
- Zone center: risk_pct `0.0030` | max_contracts `5.50`.
- Zone means: net `5195.40` | Sharpe `1.747` | maxDD `660.05` | prop_score `6.20`.
- Best point inside the zone: `risk_pct_0p0030__max_contracts_6__skip_trade_if_too_small_true`.

## Requested Readout
1. Le gagnant ponctuel OOS: `risk_pct_0p0030__max_contracts_6__skip_trade_if_too_small_true`.
2. La zone robuste OOS: cluster `2` couvrant risk_pct `0.0030` -> `0.0030` et cap `5` -> `6`.
3. Impact marginal de risk_pct autour de `0.0025`: voir la coupe `max_contracts=3` ci-dessous. La question utile est de savoir si le score reste propre quand on s'eloigne de `0.0025`.
4. Impact marginal du cap `max_contracts` autour de `3`: voir la coupe `risk_pct=0.0025` ci-dessous. La lecture utile est la vitesse de deterioration du drawdown et du `prop_score` quand on desserre le cap.
5. Distribution des tailles pour la meilleure variante: avg `5.36` | median `6.00` | 1c `0.0%` | 2c `5.1%` | 3+c `94.9%`.
6. Trades skippes pour la meilleure variante: `0` soit `0.0%` des tentatives.
7. Robustesse: `zone_etruite_mais_non_isolee`.
8. Reference de recherche: `risk_pct_0p0030__max_contracts_6__skip_trade_if_too_small_true` | mode prop-safe: `risk_pct_0p0030__max_contracts_6__skip_trade_if_too_small_true` | mode plus agressif mais defendable: `risk_pct_0p0040__max_contracts_6__skip_trade_if_too_small_true`.
9. Verdict final: `retenir_une_zone_parametrique`.

### Slice: risk_pct around 0.0025 with max_contracts=3
```text
                                         campaign_variant_name  risk_pct  max_contracts  oos_net_pnl_usd  oos_sharpe  oos_max_drawdown_usd  oos_max_daily_drawdown_usd  oos_prop_score
risk_pct_0p0015__max_contracts_3__skip_trade_if_too_small_true    0.0015            3.0         2539.875    1.654855                365.45                      365.45       -0.834360
risk_pct_0p0020__max_contracts_3__skip_trade_if_too_small_true    0.0020            3.0         2971.400    1.787271                367.00                      367.00       -0.156820
risk_pct_0p0025__max_contracts_3__skip_trade_if_too_small_true    0.0025            3.0         3044.375    1.709270                401.50                      401.50        5.367228
risk_pct_0p0030__max_contracts_3__skip_trade_if_too_small_true    0.0030            3.0         3129.350    1.698479                417.00                      417.00        5.057404
risk_pct_0p0035__max_contracts_3__skip_trade_if_too_small_true    0.0035            3.0         2959.950    1.562202                462.00                      462.00       -2.184795
risk_pct_0p0040__max_contracts_3__skip_trade_if_too_small_true    0.0040            3.0         2959.950    1.562202                462.00                      462.00       -2.184795
```

### Slice: max_contracts around 3 with risk_pct=0.0025
```text
                                         campaign_variant_name  risk_pct  max_contracts  oos_net_pnl_usd  oos_sharpe  oos_max_drawdown_usd  oos_max_daily_drawdown_usd  oos_prop_score
risk_pct_0p0025__max_contracts_2__skip_trade_if_too_small_true    0.0025            2.0         1973.300    1.562202                 308.0                       308.0       -1.894328
risk_pct_0p0025__max_contracts_3__skip_trade_if_too_small_true    0.0025            3.0         3044.375    1.709270                 401.5                       401.5        5.367228
risk_pct_0p0025__max_contracts_4__skip_trade_if_too_small_true    0.0025            4.0         3792.075    1.709856                 495.0                       495.0        5.543167
risk_pct_0p0025__max_contracts_5__skip_trade_if_too_small_true    0.0025            5.0         4488.125    1.703919                 588.5                       588.5        5.845672
risk_pct_0p0025__max_contracts_6__skip_trade_if_too_small_true    0.0025            6.0         4793.025    1.636608                 682.0                       682.0        5.367524
```
