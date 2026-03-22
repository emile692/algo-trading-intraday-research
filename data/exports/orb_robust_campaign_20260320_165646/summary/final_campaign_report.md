# ORB Robust Campaign Report

## Baseline

- Dataset: `MNQ_1mim.parquet`
- Baseline config: `{'or_minutes': 15, 'opening_time': '09:30:00', 'direction': 'long', 'one_trade_per_day': True, 'entry_buffer_ticks': 2, 'stop_buffer_ticks': 2, 'target_multiple': 2.0, 'vwap_confirmation': True, 'vwap_column': 'continuous_session_vwap', 'time_exit': '16:00:00', 'account_size_usd': 50000.0, 'risk_per_trade_pct': 0.5, 'tick_size': 0.25, 'entry_on_next_open': True}`
- Sessions: 2019-05-05 to 2026-03-02 (2127 sessions)
- Baseline trades: 1125
- Baseline Sharpe: 0.6899
- Baseline profit factor: 1.1477
- Baseline expectancy: 13.3947
- Baseline max drawdown: -2659.50
- Baseline cumulative PnL: 15069.00

## Tested Hypotheses

- A: OR width normalized by ATR can filter weak or overextended opening regimes.
- B: ATR regime bounds at signal time can reduce poor volatility conditions.
- C: Breakout structure quality improves when continuous VWAP slope confirms long direction.
- D: Price-to-VWAP distance normalized by ATR helps avoid weak/overstretched entries.

## Phase 3 Snapshot

```text
                         name                 block  n_trades  profit_factor  sharpe_ratio  expectancy  max_drawdown  phase3_robustness_score  signal_credible_phase3
                b_atr_q20_q90          B_atr_regime       788       1.294469      1.089292   25.470812       -2083.5                 0.461809                    True
                 b_atr_ge_q30          B_atr_regime       787       1.205877      0.802025   19.207751       -2298.5                 0.184623                    True
              a_ratio_q10_q90   A_or_width_over_atr       899       1.183978      0.764521   16.807564       -2080.0                 0.141790                    True
                b_atr_q35_q85          B_atr_regime       563       1.334338      1.031750   28.859680       -2636.0                 0.459118                   False
d_price_vwap_dist_atr_q30_q85 D_price_vwap_distance       618       1.201626      0.688967   18.301780       -2074.5                 0.135888                   False
     c_vwap_slope_atr_q40_q95  C_breakout_structure       617       1.215111      0.725875   19.137763       -3346.5                 0.097309                   False
              a_ratio_q30_q85   A_or_width_over_atr       618       1.190665      0.652770   17.110032       -2986.0                 0.039393                   False
              a_ratio_q20_q80   A_or_width_over_atr       675       1.147513      0.541727   13.723704       -2315.5                -0.040502                   False
   c_vwap_slope_atr_ge_median  C_breakout_structure       561       1.166131      0.544712   14.607843       -2929.0                -0.056040                   False
 d_price_vwap_dist_atr_ge_q20 D_price_vwap_distance       900       1.137102      0.573289   12.197778       -3434.5                -0.110139                   False
```

## Block Conclusions

- A_or_width_over_atr: best=a_ratio_q10_q90 -> signal credible (score=0.1418, PF=1.184, Sharpe=0.765, DD=-2080.00)
- B_atr_regime: best=b_atr_q20_q90 -> signal credible (score=0.4618, PF=1.294, Sharpe=1.089, DD=-2083.50)
- C_breakout_structure: best=c_vwap_slope_atr_q40_q95 -> pas convaincant (score=0.0973, PF=1.215, Sharpe=0.726, DD=-3346.50)
- D_price_vwap_distance: best=d_price_vwap_dist_atr_q30_q85 -> pas convaincant (score=0.1359, PF=1.202, Sharpe=0.689, DD=-2074.50)

## Final Validation

b_atr_q20_q90 shows robust evidence of beating the baseline.

## Recommendation

- Promote a candidate only if multi-metric improvements are preserved out-of-sample and not concentrated in one year.
- If no candidate passes those checks, keep the baseline and continue with similarly parsimonious regime filters.
