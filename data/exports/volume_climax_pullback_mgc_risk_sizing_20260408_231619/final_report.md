# Volume Climax Pullback MGC Risk Sizing - Final Report

## Scope
- Symbol: `MGC` only.
- Base alpha reused unchanged: `regime_filtered_ema_mild_atr_20_80_compression_off_atr_target_1p0_ts3_vq0p95_bf0p5_ra1p2`.
- Resolved alpha variant object: `regime_filtered_ema_mild_atr_20_80_compression_off_atr_target_1p0_ts3_vq0p95_bf0p5_ra1p2`.
- Reference V3 run: `D:\Business\Trading\VSCODE\algo-trading-intraday-research\data\exports\volume_climax_pullback_v3_run\volume_climax_pullback_v3_20260402_184702`.
- Dataset: `repo latest MGC source`.
- Sessions: full `3128` | IS `2189` | OOS `939`.
- OOS-only runs restart from the same 50k capital while keeping signals precomputed on the full leak-free history.

## Baseline Vs Sizing
- Baseline OOS: net `1515.20` | CAGR `0.70%` | Sharpe `1.184` | maxDD `152.50`.
- Best risk-sized OOS row by net PnL: `risk_pct_0p0100__max_contracts_15__skip_trade_if_too_small_true` | net `8499.90` | CAGR `3.72%` | Sharpe `0.949` | maxDD `1723.20`.

```text
                                           campaign_variant_name  oos_net_pnl_usd  oos_cagr_pct  oos_sharpe  oos_max_drawdown_usd  oos_pass_target_3000_usd_without_breaching_2000_dd  oos_avg_contracts_entered
 risk_pct_0p0100__max_contracts_15__skip_trade_if_too_small_true          8499.90      3.720073    0.949418                1723.2                                                True                  11.205882
risk_pct_0p0100__max_contracts_15__skip_trade_if_too_small_false          8499.90      3.720073    0.949418                1723.2                                                True                  11.205882
risk_pct_0p0075__max_contracts_15__skip_trade_if_too_small_false          7358.00      3.245498    0.908939                1743.7                                                True                   9.882353
 risk_pct_0p0100__max_contracts_10__skip_trade_if_too_small_true          7033.60      3.109356    1.024176                1211.0                                                True                   8.441176
risk_pct_0p0100__max_contracts_10__skip_trade_if_too_small_false          7033.60      3.109356    1.024176                1211.0                                                True                   8.441176
 risk_pct_0p0075__max_contracts_15__skip_trade_if_too_small_true          6899.60      3.052946    0.857278                1743.7                                                True                  10.151515
risk_pct_0p0050__max_contracts_15__skip_trade_if_too_small_false          6317.65      2.806774    0.868321                1425.8                                                True                   8.000000
risk_pct_0p0075__max_contracts_10__skip_trade_if_too_small_false          5919.00      2.637011    0.960683                1168.7                                                True                   7.676471
```

## Readout
1. Baseline 1 contrat vs risk sizing: no risk-sized row improved OOS net PnL without a material maxDD trade-off.
2. Variantes qui ameliorent CAGR / net PnL sans degrader excessivement le maxDD: aucune claire.
3. Cadre prop 50k: meilleure lecture defensive = `risk_pct_0p0100__max_contracts_3__skip_trade_if_too_small_true` | pass flag `True` | trailing DD `457.50` | jours <= -1000 USD `0`.
4. Impact de `skip_trade_if_too_small`: forcer 1 contrat change en moyenne l'OOS net PnL de `+406.36` USD, le maxDD OOS de `+0.00` USD, et ajoute `+1.00` trade(s) OOS.
5. Impact du cap `max_contracts`: `cap=3` -> `risk_pct_0p0100__max_contracts_3__skip_trade_if_too_small_true` net `3628.80` / Sharpe `1.244` / cap-hit `94.3%`; `cap=5` -> `risk_pct_0p0100__max_contracts_5__skip_trade_if_too_small_true` net `5134.00` / Sharpe `1.205` / cap-hit `88.6%`; `cap=10` -> `risk_pct_0p0100__max_contracts_10__skip_trade_if_too_small_true` net `7033.60` / Sharpe `1.024` / cap-hit `62.9%`; `cap=15` -> `risk_pct_0p0100__max_contracts_15__skip_trade_if_too_small_true` net `8499.90` / Sharpe `0.949` / cap-hit `48.6%`.
6. Stabilite de la courbe d'equity: le sizing augmente surtout la dispersion sans produire de vainqueur OOS assez propre.

## Recommendation
- Verdict final: `ne pas retenir`.
- Raison: aucune variante risk-based ne bat la baseline OOS avec un compromis rendement / drawdown suffisamment propre.
