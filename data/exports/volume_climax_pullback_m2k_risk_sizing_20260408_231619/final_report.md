# Volume Climax Pullback M2K Risk Sizing - Final Report

## Scope
- Symbol: `M2K` only.
- Base alpha reused unchanged: `dynamic_exit_atr_target_1p0_ts4_vq0p95_bf0p5_ra1p2`.
- Resolved alpha variant object: `dynamic_exit_atr_target_1p0_ts4_vq0p95_bf0p5_ra1p2`.
- Reference V3 run: `D:\Business\Trading\VSCODE\algo-trading-intraday-research\data\exports\volume_climax_pullback_v3_run\volume_climax_pullback_v3_20260402_184702`.
- Dataset: `repo latest M2K source`.
- Sessions: full `1747` | IS `1222` | OOS `525`.
- OOS-only runs restart from the same 50k capital while keeping signals precomputed on the full leak-free history.

## Baseline Vs Sizing
- Baseline OOS: net `986.65` | CAGR `0.95%` | Sharpe `1.562` | maxDD `154.00`.
- Best risk-sized OOS row by net PnL: `risk_pct_0p0100__max_contracts_15__skip_trade_if_too_small_true` | net `14980.35` | CAGR `13.50%` | Sharpe `1.689` | maxDD `1985.00`.

```text
                                           campaign_variant_name  oos_net_pnl_usd  oos_cagr_pct  oos_sharpe  oos_max_drawdown_usd  oos_pass_target_3000_usd_without_breaching_2000_dd  oos_avg_contracts_entered
 risk_pct_0p0100__max_contracts_15__skip_trade_if_too_small_true        14980.350     13.497572    1.688828                1985.0                                                True                  14.494949
risk_pct_0p0100__max_contracts_15__skip_trade_if_too_small_false        14980.350     13.497572    1.688828                1985.0                                                True                  14.494949
 risk_pct_0p0075__max_contracts_15__skip_trade_if_too_small_true        14065.225     12.722500    1.707317                1785.0                                                True                  13.797980
risk_pct_0p0075__max_contracts_15__skip_trade_if_too_small_false        14065.225     12.722500    1.707317                1785.0                                                True                  13.797980
 risk_pct_0p0050__max_contracts_15__skip_trade_if_too_small_true        11470.125     10.492894    1.640809                1530.5                                                True                  12.262626
risk_pct_0p0050__max_contracts_15__skip_trade_if_too_small_false        11470.125     10.492894    1.640809                1530.5                                                True                  12.262626
 risk_pct_0p0075__max_contracts_10__skip_trade_if_too_small_true        10124.750      9.317837    1.677004                1340.0                                                True                   9.787879
risk_pct_0p0075__max_contracts_10__skip_trade_if_too_small_false        10124.750      9.317837    1.677004                1340.0                                                True                   9.787879
```

## Readout
1. Baseline 1 contrat vs risk sizing: no risk-sized row improved OOS net PnL without a material maxDD trade-off.
2. Variantes qui ameliorent CAGR / net PnL sans degrader excessivement le maxDD: aucune claire.
3. Cadre prop 50k: meilleure lecture defensive = `risk_pct_0p0025__max_contracts_3__skip_trade_if_too_small_true` | pass flag `True` | trailing DD `401.50` | jours <= -1000 USD `0`.
4. Impact de `skip_trade_if_too_small`: forcer 1 contrat change en moyenne l'OOS net PnL de `+0.00` USD, le maxDD OOS de `+0.00` USD, et ajoute `+0.00` trade(s) OOS.
5. Impact du cap `max_contracts`: `cap=3` -> `risk_pct_0p0025__max_contracts_3__skip_trade_if_too_small_true` net `3044.38` / Sharpe `1.709` / cap-hit `90.9%`; `cap=5` -> `risk_pct_0p0050__max_contracts_5__skip_trade_if_too_small_true` net `5102.65` / Sharpe `1.645` / cap-hit `94.9%`; `cap=10` -> `risk_pct_0p0075__max_contracts_10__skip_trade_if_too_small_true` net `10124.75` / Sharpe `1.677` / cap-hit `90.9%`; `cap=15` -> `risk_pct_0p0100__max_contracts_15__skip_trade_if_too_small_true` net `14980.35` / Sharpe `1.689` / cap-hit `83.8%`.
6. Stabilite de la courbe d'equity: le sizing augmente surtout la dispersion sans produire de vainqueur OOS assez propre.

## Recommendation
- Verdict final: `ne pas retenir`.
- Raison: aucune variante risk-based ne bat la baseline OOS avec un compromis rendement / drawdown suffisamment propre.
