# Volume Climax Pullback MGC Risk Sizing - Refinement Report

## Scope
- Symbol: `MGC` only.
- Base alpha reused unchanged: `regime_filtered_ema_mild_atr_20_80_compression_off_atr_target_1p0_ts3_vq0p95_bf0p5_ra1p2`.
- Reference V3 run: `D:\Business\Trading\VSCODE\algo-trading-intraday-research\data\exports\volume_climax_pullback_v3_run\volume_climax_pullback_v3_20260402_184702`.
- Dataset: `repo latest MGC source`.
- Sessions: full `3128` | IS `2189` | OOS `939`.
- Sizing logic unchanged from the prior campaign. Only the local grid around the previous winner was refined.

## Prop Score
- Formula on OOS: `prop_score = 4 * min(net_pnl/3000, 2) + 3 * min(sharpe, 2.5) - 3 * min(maxDD/2000, 3) - 5 * min(max_daily_DD/1000, 3) - 2 * min(worst_trade/200, 3) - 0.5 * nb_days_below_-500 - 6 * 1[pass=False]`.
- Interpretation: reward enough OOS profit to clear a 50k target, reward clean Sharpe, penalize drawdown, penalize daily drawdown strongly, and penalize any non-pass configuration heavily.

## OOS Winner
- Punctual OOS winner by `prop_score`: `risk_pct_0p0100__max_contracts_3__skip_trade_if_too_small_true`.
- OOS metrics: net `3628.80` | CAGR `1.64%` | Sharpe `1.244` | maxDD `457.50` | max daily DD `457.50` | prop_score `3.24`.
- Versus previous winner tag: net `+0.00` | Sharpe `+0.000` | maxDD `+0.00`.

```text
                                         campaign_variant_name  oos_net_pnl_usd  oos_sharpe  oos_max_drawdown_usd  oos_max_daily_drawdown_usd  oos_prop_score  oos_pass_target_3000_usd_without_breaching_2000_dd
risk_pct_0p0100__max_contracts_3__skip_trade_if_too_small_true          3628.80    1.244038                 457.5                       457.5        3.241763                                                True
risk_pct_0p0105__max_contracts_3__skip_trade_if_too_small_true          3628.80    1.244038                 457.5                       457.5        3.241763                                                True
risk_pct_0p0095__max_contracts_3__skip_trade_if_too_small_true          3378.55    1.246298                 457.5                       457.5        2.914877                                                True
risk_pct_0p0095__max_contracts_4__skip_trade_if_too_small_true          4435.35    1.234961                 610.0                       610.0        2.513683                                                True
risk_pct_0p0100__max_contracts_4__skip_trade_if_too_small_true          4435.35    1.234961                 610.0                       610.0        2.513683                                                True
risk_pct_0p0105__max_contracts_4__skip_trade_if_too_small_true          4435.35    1.234961                 610.0                       610.0        2.513683                                                True
risk_pct_0p0105__max_contracts_5__skip_trade_if_too_small_true          5241.90    1.217220                 762.5                       762.5        1.759609                                                True
risk_pct_0p0100__max_contracts_5__skip_trade_if_too_small_true          5134.00    1.204768                 762.5                       762.5        1.578387                                                True
risk_pct_0p0085__max_contracts_4__skip_trade_if_too_small_true          3726.70    1.125075                 610.0                       610.0        1.239158                                                True
risk_pct_0p0090__max_contracts_4__skip_trade_if_too_small_true          3726.70    1.125075                 610.0                       610.0        1.239158                                                True
```

## Robust Zone
- No multi-point robust cluster passed the guardrails plus top-quartile filter. The previous point looks isolated.

## Requested Readout
1. Le gagnant ponctuel OOS: `risk_pct_0p0100__max_contracts_3__skip_trade_if_too_small_true`.
2. La zone robuste OOS: aucune zone multi-point claire, le signal de robustesse reste local.
3. Impact marginal de risk_pct autour de `0.0100`: voir la coupe `max_contracts=3` ci-dessous. La question utile est de savoir si le score reste propre quand on s'eloigne de `0.0100`.
4. Impact marginal du cap `max_contracts` autour de `3`: voir la coupe `risk_pct=0.0100` ci-dessous. La lecture utile est la vitesse de deterioration du drawdown et du `prop_score` quand on desserre le cap.
5. Distribution des tailles pour la meilleure variante: avg `2.94` | median `3.00` | 1c `2.9%` | 2c `0.0%` | 3+c `97.1%`.
6. Trades skippes pour la meilleure variante: `1` soit `2.9%` des tentatives.
7. Robustesse: locale et trop etroite pour parler d'une vraie zone.
8. Reference / prop-safe / agressif: voir les tableaux, aucun trio complet ne ressort proprement.
9. Verdict final: `retenir_une_unique_variante`.

### Slice: risk_pct around 0.0100 with max_contracts=3
```text
                                         campaign_variant_name  risk_pct  max_contracts  oos_net_pnl_usd  oos_sharpe  oos_max_drawdown_usd  oos_max_daily_drawdown_usd  oos_prop_score
risk_pct_0p0075__max_contracts_3__skip_trade_if_too_small_true    0.0075            3.0          2920.15    1.138628                 457.5                       457.5       -4.019332
risk_pct_0p0080__max_contracts_3__skip_trade_if_too_small_true    0.0080            3.0          2920.15    1.138628                 457.5                       457.5       -4.019332
risk_pct_0p0085__max_contracts_3__skip_trade_if_too_small_true    0.0085            3.0          2920.15    1.138628                 457.5                       457.5       -4.019332
risk_pct_0p0090__max_contracts_3__skip_trade_if_too_small_true    0.0090            3.0          2920.15    1.138628                 457.5                       457.5       -4.019332
risk_pct_0p0095__max_contracts_3__skip_trade_if_too_small_true    0.0095            3.0          3378.55    1.246298                 457.5                       457.5        2.914877
risk_pct_0p0100__max_contracts_3__skip_trade_if_too_small_true    0.0100            3.0          3628.80    1.244038                 457.5                       457.5        3.241763
risk_pct_0p0105__max_contracts_3__skip_trade_if_too_small_true    0.0105            3.0          3628.80    1.244038                 457.5                       457.5        3.241763
```

### Slice: max_contracts around 3 with risk_pct=0.0100
```text
                                         campaign_variant_name  risk_pct  max_contracts  oos_net_pnl_usd  oos_sharpe  oos_max_drawdown_usd  oos_max_daily_drawdown_usd  oos_prop_score
risk_pct_0p0100__max_contracts_2__skip_trade_if_too_small_true      0.01            2.0          2572.00    1.252984                 305.0                       305.0       -2.364213
risk_pct_0p0100__max_contracts_3__skip_trade_if_too_small_true      0.01            3.0          3628.80    1.244038                 457.5                       457.5        3.241763
risk_pct_0p0100__max_contracts_4__skip_trade_if_too_small_true      0.01            4.0          4435.35    1.234961                 610.0                       610.0        2.513683
risk_pct_0p0100__max_contracts_5__skip_trade_if_too_small_true      0.01            5.0          5134.00    1.204768                 762.5                       762.5        1.578387
risk_pct_0p0100__max_contracts_6__skip_trade_if_too_small_true      0.01            6.0          5505.80    1.144813                 915.0                       915.0        0.118004
```
