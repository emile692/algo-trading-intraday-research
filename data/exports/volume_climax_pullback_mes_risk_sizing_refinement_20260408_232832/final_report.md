# Volume Climax Pullback MES Risk Sizing - Refinement Report

## Scope
- Symbol: `MES` only.
- Base alpha reused unchanged: `dynamic_exit_mixed_ts4_vq0p95_bf0p6_ra1p5`.
- Reference V3 run: `D:\Business\Trading\VSCODE\algo-trading-intraday-research\data\exports\volume_climax_pullback_v3_run\volume_climax_pullback_v3_20260402_184702`.
- Dataset: `repo latest MES source`.
- Sessions: full `1747` | IS `1222` | OOS `525`.
- Sizing logic unchanged from the prior campaign. Only the local grid around the previous winner was refined.

## Prop Score
- Formula on OOS: `prop_score = 4 * min(net_pnl/3000, 2) + 3 * min(sharpe, 2.5) - 3 * min(maxDD/2000, 3) - 5 * min(max_daily_DD/1000, 3) - 2 * min(worst_trade/200, 3) - 0.5 * nb_days_below_-500 - 6 * 1[pass=False]`.
- Interpretation: reward enough OOS profit to clear a 50k target, reward clean Sharpe, penalize drawdown, penalize daily drawdown strongly, and penalize any non-pass configuration heavily.

## OOS Winner
- Punctual OOS winner by `prop_score`: `risk_pct_0p0020__max_contracts_6__skip_trade_if_too_small_true`.
- OOS metrics: net `5738.81` | CAGR `5.39%` | Sharpe `1.346` | maxDD `552.50` | max daily DD `552.50` | prop_score `6.85`.
- Versus previous winner tag: net `+1926.27` | Sharpe `-0.158` | maxDD `+36.97`.

```text
                                         campaign_variant_name  oos_net_pnl_usd  oos_sharpe  oos_max_drawdown_usd  oos_max_daily_drawdown_usd  oos_prop_score  oos_pass_target_3000_usd_without_breaching_2000_dd
risk_pct_0p0020__max_contracts_6__skip_trade_if_too_small_true      5738.812500    1.345825               552.500                     552.500        6.847975                                                True
risk_pct_0p0015__max_contracts_6__skip_trade_if_too_small_true      4964.046875    1.249596               398.750                     398.750        6.775643                                                True
risk_pct_0p0020__max_contracts_5__skip_trade_if_too_small_true      5045.828125    1.385554               462.500                     462.500        6.678184                                                True
risk_pct_0p0025__max_contracts_6__skip_trade_if_too_small_true      6578.578125    1.466151               641.250                     641.250        6.667829                                                True
risk_pct_0p0015__max_contracts_5__skip_trade_if_too_small_true      4514.218750    1.313403               373.750                     373.750        6.579792                                                True
risk_pct_0p0025__max_contracts_5__skip_trade_if_too_small_true      5719.343750    1.475838               641.250                     641.250        6.322680                                                True
risk_pct_0p0025__max_contracts_4__skip_trade_if_too_small_true      4948.859375    1.507102               559.875                     559.875        6.018098                                                True
risk_pct_0p0020__max_contracts_4__skip_trade_if_too_small_true      4326.593750    1.423258               462.500                     462.500        5.832317                                                True
risk_pct_0p0015__max_contracts_4__skip_trade_if_too_small_true      3830.640625    1.348310               373.750                     373.750        5.823075                                                True
risk_pct_0p0030__max_contracts_5__skip_trade_if_too_small_true      5953.875000    1.460441               780.000                     780.000        5.487324                                                True
```

## Robust Zone
- Recommended cluster id: `1` | size `4` | scale `zone_etruite_mais_non_isolee`.
- Zone range: risk_pct `0.0015` -> `0.0025` | max_contracts `5` -> `6`.
- Zone center: risk_pct `0.0020` | max_contracts `5.75`.
- Zone means: net `5581.82` | Sharpe `1.362` | maxDD `513.75` | prop_score `6.74`.
- Best point inside the zone: `risk_pct_0p0020__max_contracts_6__skip_trade_if_too_small_true`.

## Requested Readout
1. Le gagnant ponctuel OOS: `risk_pct_0p0020__max_contracts_6__skip_trade_if_too_small_true`.
2. La zone robuste OOS: cluster `1` couvrant risk_pct `0.0015` -> `0.0025` et cap `5` -> `6`.
3. Impact marginal de risk_pct autour de `0.0025`: voir la coupe `max_contracts=3` ci-dessous. La question utile est de savoir si le score reste propre quand on s'eloigne de `0.0025`.
4. Impact marginal du cap `max_contracts` autour de `3`: voir la coupe `risk_pct=0.0025` ci-dessous. La lecture utile est la vitesse de deterioration du drawdown et du `prop_score` quand on desserre le cap.
5. Distribution des tailles pour la meilleure variante: avg `4.28` | median `6.00` | 1c `17.2%` | 2c `10.9%` | 3+c `71.9%`.
6. Trades skippes pour la meilleure variante: `1` soit `1.5%` des tentatives.
7. Robustesse: `zone_etruite_mais_non_isolee`.
8. Reference de recherche: `risk_pct_0p0020__max_contracts_6__skip_trade_if_too_small_true` | mode prop-safe: `risk_pct_0p0020__max_contracts_6__skip_trade_if_too_small_true` | mode plus agressif mais defendable: `risk_pct_0p0035__max_contracts_6__skip_trade_if_too_small_true`.
9. Verdict final: `retenir_une_zone_parametrique`.

### Slice: risk_pct around 0.0025 with max_contracts=3
```text
                                         campaign_variant_name  risk_pct  max_contracts  oos_net_pnl_usd  oos_sharpe  oos_max_drawdown_usd  oos_max_daily_drawdown_usd  oos_prop_score
risk_pct_0p0015__max_contracts_3__skip_trade_if_too_small_true    0.0015            3.0      3047.656250    1.367716            373.750000                  373.750000        4.837315
risk_pct_0p0020__max_contracts_3__skip_trade_if_too_small_true    0.0020            3.0      3524.859375    1.443691            405.000000                  405.000000        5.310884
risk_pct_0p0025__max_contracts_3__skip_trade_if_too_small_true    0.0025            3.0      3812.546875    1.503555            515.531250                  515.531250        4.943109
risk_pct_0p0030__max_contracts_3__skip_trade_if_too_small_true    0.0030            3.0      3775.000000    1.470884            653.390625                  653.390625        3.598945
risk_pct_0p0035__max_contracts_3__skip_trade_if_too_small_true    0.0035            3.0      3741.859375    1.416771            824.281250                  824.281250        1.931632
risk_pct_0p0040__max_contracts_3__skip_trade_if_too_small_true    0.0040            3.0      3563.109375    1.336595            928.031250                  928.031250        0.478393
```

### Slice: max_contracts around 3 with risk_pct=0.0025
```text
                                         campaign_variant_name  risk_pct  max_contracts  oos_net_pnl_usd  oos_sharpe  oos_max_drawdown_usd  oos_max_daily_drawdown_usd  oos_prop_score
risk_pct_0p0025__max_contracts_2__skip_trade_if_too_small_true    0.0025            2.0      2540.109375    1.439388             557.43750                   557.43750       -3.218367
risk_pct_0p0025__max_contracts_3__skip_trade_if_too_small_true    0.0025            3.0      3812.546875    1.503555             515.53125                   515.53125        4.943109
risk_pct_0p0025__max_contracts_4__skip_trade_if_too_small_true    0.0025            4.0      4948.859375    1.507102             559.87500                   559.87500        6.018098
risk_pct_0p0025__max_contracts_5__skip_trade_if_too_small_true    0.0025            5.0      5719.343750    1.475838             641.25000                   641.25000        6.322680
risk_pct_0p0025__max_contracts_6__skip_trade_if_too_small_true    0.0025            6.0      6578.578125    1.466151             641.25000                   641.25000        6.667829
```
