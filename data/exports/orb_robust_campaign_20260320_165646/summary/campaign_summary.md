# Campaign Summary

## Baseline Recall

- Trades: 1125
- Profit factor: 1.1477
- Sharpe: 0.6899
- Expectancy: 13.3947
- Max drawdown: -2659.50
- Cumulative PnL: 15069.00

## Hypotheses Tested

- A: OR width / ATR regime window.
- B: ATR regime bounds at signal time.
- C: Continuous VWAP slope quality filters.
- D: Price-VWAP distance normalized by ATR.

## Rejected (Not Convincing)

- C_breakout_structure: best variant `c_vwap_slope_atr_q40_q95` did not show robust multi-metric evidence.
- D_price_vwap_distance: best variant `d_price_vwap_dist_atr_q30_q85` did not show robust multi-metric evidence.

## Promising

- `b_atr_q20_q90`: PF=1.294, Sharpe=1.089, Expectancy=25.47, DD=-2083.5, Trades=788.
- `b_atr_ge_q30`: PF=1.206, Sharpe=0.802, Expectancy=19.21, DD=-2298.5, Trades=787.
- `a_ratio_q10_q90`: PF=1.184, Sharpe=0.765, Expectancy=16.81, DD=-2080.0, Trades=899.

## Baseline vs Best Candidates (Overall)

```text
      candidate  n_trades  profit_factor  sharpe_ratio  expectancy  max_drawdown  cumulative_pnl
a_ratio_q10_q90       899       1.183978      0.764521   16.807564       -2080.0         15110.0
   b_atr_ge_q30       787       1.205877      0.802025   19.207751       -2298.5         15116.5
  b_atr_q20_q90       788       1.294469      1.089292   25.470812       -2083.5         20071.0
 baseline_exact      1125       1.147651      0.689928   13.394667       -2659.5         15069.0
```

## Baseline vs Best Candidates (Out-of-Sample 30%)

```text
      candidate  n_trades  profit_factor  sharpe_ratio  expectancy  max_drawdown  cumulative_pnl
a_ratio_q10_q90       231       1.179173      0.680470   16.125541       -1505.5          3725.0
   b_atr_ge_q30       250       1.230144      0.903674   21.234000       -1461.5          5308.5
  b_atr_q20_q90       222       1.242146      0.879616   21.213964       -1397.5          4709.5
 baseline_exact       288       1.142040      0.612873   13.020833       -1879.0          3750.0
```

## Honest Conclusion

- Based on this campaign, `b_atr_q20_q90` is the strongest robust candidate and shows credible evidence of beating the baseline.
- Caveat: it does so with lower participation/trade count than baseline, so production sizing should be stress-tested.

## Next Steps

- Re-run the same winning filter on a fully held-out future slice once available.
- Stress-test execution assumptions (fees/slippage) for the finalist vs baseline.
- Keep model complexity flat; avoid combining many filters unless incremental evidence is stable.
