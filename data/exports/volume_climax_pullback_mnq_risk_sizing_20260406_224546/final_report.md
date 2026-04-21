# Volume Climax Pullback MNQ Risk Sizing - Final Report

## Scope
- Symbol: `MNQ` only.
- Base alpha reused unchanged: `dynamic_exit_atr_target_1p0_ts2_vq0p95_bf0p5_ra1p2`.
- Resolved alpha variant object: `dynamic_exit_atr_target_1p0_ts2_vq0p95_bf0p5_ra1p2`.
- Reference V3 run: `D:\Business\Trading\VSCODE\algo-trading-intraday-research\data\exports\volume_climax_pullback_v3_run\volume_climax_pullback_v3_20260402_184702`.
- Dataset: `repo latest MNQ source`.
- Sessions: full `1747` | IS `1222` | OOS `525`.
- OOS-only runs restart from the same 50k capital while keeping signals precomputed on the full leak-free history.

## Baseline Vs Sizing
- Baseline OOS: net `3038.57` | CAGR `2.89%` | Sharpe `1.451` | maxDD `790.00`.
- Best risk-sized OOS row by net PnL: `risk_pct_0p0100__max_contracts_15__skip_trade_if_too_small_true` | net `35511.05` | CAGR `29.60%` | Sharpe `1.666` | maxDD `4930.00`.

```text
                                           campaign_variant_name  oos_net_pnl_usd  oos_cagr_pct  oos_sharpe  oos_max_drawdown_usd  oos_pass_target_3000_usd_without_breaching_2000_dd  oos_avg_contracts_entered
 risk_pct_0p0100__max_contracts_15__skip_trade_if_too_small_true        35511.050     29.597216    1.665586                4930.0                                               False                  10.495050
risk_pct_0p0100__max_contracts_15__skip_trade_if_too_small_false        35511.050     29.597216    1.665586                4930.0                                               False                  10.495050
 risk_pct_0p0075__max_contracts_15__skip_trade_if_too_small_true        31308.200     26.479712    1.667319                3702.0                                               False                   9.000000
risk_pct_0p0075__max_contracts_15__skip_trade_if_too_small_false        31308.200     26.479712    1.667319                3702.0                                               False                   9.000000
 risk_pct_0p0100__max_contracts_10__skip_trade_if_too_small_true        28279.650     24.181288    1.730177                3868.0                                               False                   7.960396
risk_pct_0p0100__max_contracts_10__skip_trade_if_too_small_false        28279.650     24.181288    1.730177                3868.0                                               False                   7.960396
risk_pct_0p0050__max_contracts_15__skip_trade_if_too_small_false        26298.025     22.652435    1.663745                2191.5                                                True                   7.089109
 risk_pct_0p0050__max_contracts_15__skip_trade_if_too_small_true        26144.025     22.532767    1.654060                2191.5                                                True                   7.150000
```

## Readout
1. Baseline 1 contrat vs risk sizing: `5` risk-sized rows improved OOS net PnL while keeping OOS maxDD within +25% of baseline.
2. Variantes qui ameliorent CAGR / net PnL sans degrader excessivement le maxDD: `risk_pct_0p0025__max_contracts_3__skip_trade_if_too_small_true`, `risk_pct_0p0025__max_contracts_3__skip_trade_if_too_small_false`, `risk_pct_0p0025__max_contracts_5__skip_trade_if_too_small_true`, `risk_pct_0p0025__max_contracts_5__skip_trade_if_too_small_false`, `risk_pct_0p0025__max_contracts_10__skip_trade_if_too_small_true`.
3. Cadre prop 50k: meilleure lecture defensive = `risk_pct_0p0025__max_contracts_3__skip_trade_if_too_small_true` | pass flag `True` | trailing DD `644.95` | jours <= -1000 USD `0`.
4. Impact de `skip_trade_if_too_small`: forcer 1 contrat change en moyenne l'OOS net PnL de `-6.81` USD, le maxDD OOS de `+81.50` USD, et ajoute `+2.56` trade(s) OOS.
5. Impact du cap `max_contracts`: `cap=3` -> `risk_pct_0p0100__max_contracts_3__skip_trade_if_too_small_true` net `9210.72` / Sharpe `1.490` / cap-hit `97.0%`; `cap=5` -> `risk_pct_0p0100__max_contracts_5__skip_trade_if_too_small_true` net `14920.80` / Sharpe `1.549` / cap-hit `80.2%`; `cap=10` -> `risk_pct_0p0100__max_contracts_10__skip_trade_if_too_small_true` net `28279.65` / Sharpe `1.730` / cap-hit `56.4%`; `cap=15` -> `risk_pct_0p0100__max_contracts_15__skip_trade_if_too_small_true` net `35511.05` / Sharpe `1.666` / cap-hit `41.6%`.
6. Stabilite de la courbe d'equity: `risk_pct_0p0025__max_contracts_3__skip_trade_if_too_small_true` vs baseline = Sharpe `+0.289`, vol annualisee `+1.75` pts, maxDD `-153.50` USD.

## Recommendation
- Verdict final: `retenir risk_pct_0p0025__max_contracts_3__skip_trade_if_too_small_true`.
- Variante recommandee: `risk_pct_0p0025__max_contracts_3__skip_trade_if_too_small_true` | risk=0.0025, cap=3, skip_small=True.
- Variante agressive: `risk_pct_0p0100__max_contracts_15__skip_trade_if_too_small_true` | risk=0.0100, cap=15, skip_small=True.
- Variante prop-safe: `risk_pct_0p0025__max_contracts_3__skip_trade_if_too_small_true` | risk=0.0025, cap=3, skip_small=True.
