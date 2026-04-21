# Volume Climax Pullback MES Risk Sizing - Final Report

## Scope
- Symbol: `MES` only.
- Base alpha reused unchanged: `dynamic_exit_mixed_ts4_vq0p95_bf0p6_ra1p5`.
- Resolved alpha variant object: `dynamic_exit_mixed_ts4_vq0p95_bf0p6_ra1p5`.
- Reference V3 run: `D:\Business\Trading\VSCODE\algo-trading-intraday-research\data\exports\volume_climax_pullback_v3_run\volume_climax_pullback_v3_20260402_184702`.
- Dataset: `repo latest MES source`.
- Sessions: full `1747` | IS `1222` | OOS `525`.
- OOS-only runs restart from the same 50k capital while keeping signals precomputed on the full leak-free history.

## Baseline Vs Sizing
- Baseline OOS: net `1388.59` | CAGR `1.33%` | Sharpe `1.376` | maxDD `370.59`.
- Best risk-sized OOS row by net PnL: `risk_pct_0p0100__max_contracts_15__skip_trade_if_too_small_true` | net `18406.11` | CAGR `16.35%` | Sharpe `1.430` | maxDD `3410.16`.

```text
                                           campaign_variant_name  oos_net_pnl_usd  oos_cagr_pct  oos_sharpe  oos_max_drawdown_usd  oos_pass_target_3000_usd_without_breaching_2000_dd  oos_avg_contracts_entered
 risk_pct_0p0100__max_contracts_15__skip_trade_if_too_small_true     18406.109375     16.350080    1.430472            3410.15625                                                True                  13.646154
risk_pct_0p0100__max_contracts_15__skip_trade_if_too_small_false     18406.109375     16.350080    1.430472            3410.15625                                                True                  13.646154
 risk_pct_0p0075__max_contracts_15__skip_trade_if_too_small_true     17782.265625     15.836222    1.452489            2517.65625                                                True                  12.769231
risk_pct_0p0075__max_contracts_15__skip_trade_if_too_small_false     17782.265625     15.836222    1.452489            2517.65625                                                True                  12.769231
 risk_pct_0p0050__max_contracts_15__skip_trade_if_too_small_true     15723.703125     14.123018    1.419305            1675.00000                                                True                  11.292308
risk_pct_0p0050__max_contracts_15__skip_trade_if_too_small_false     15723.703125     14.123018    1.419305            1675.00000                                                True                  11.292308
 risk_pct_0p0100__max_contracts_10__skip_trade_if_too_small_true     12475.718750     11.362528    1.387729            3130.93750                                                True                   9.630769
risk_pct_0p0100__max_contracts_10__skip_trade_if_too_small_false     12475.718750     11.362528    1.387729            3130.93750                                                True                   9.630769
```

## Readout
1. Baseline 1 contrat vs risk sizing: no risk-sized row improved OOS net PnL without a material maxDD trade-off.
2. Variantes qui ameliorent CAGR / net PnL sans degrader excessivement le maxDD: aucune claire.
3. Cadre prop 50k: meilleure lecture defensive = `risk_pct_0p0025__max_contracts_3__skip_trade_if_too_small_true` | pass flag `True` | trailing DD `515.53` | jours <= -1000 USD `0`.
4. Impact de `skip_trade_if_too_small`: forcer 1 contrat change en moyenne l'OOS net PnL de `+0.00` USD, le maxDD OOS de `+0.00` USD, et ajoute `+0.00` trade(s) OOS.
5. Impact du cap `max_contracts`: `cap=3` -> `risk_pct_0p0075__max_contracts_3__skip_trade_if_too_small_true` net `4165.78` / Sharpe `1.376` / cap-hit `100.0%`; `cap=5` -> `risk_pct_0p0100__max_contracts_5__skip_trade_if_too_small_true` net `6652.38` / Sharpe `1.361` / cap-hit `98.5%`; `cap=10` -> `risk_pct_0p0100__max_contracts_10__skip_trade_if_too_small_true` net `12475.72` / Sharpe `1.388` / cap-hit `87.7%`; `cap=15` -> `risk_pct_0p0100__max_contracts_15__skip_trade_if_too_small_true` net `18406.11` / Sharpe `1.430` / cap-hit `72.3%`.
6. Stabilite de la courbe d'equity: le sizing augmente surtout la dispersion sans produire de vainqueur OOS assez propre.

## Recommendation
- Verdict final: `ne pas retenir`.
- Raison: aucune variante risk-based ne bat la baseline OOS avec un compromis rendement / drawdown suffisamment propre.
