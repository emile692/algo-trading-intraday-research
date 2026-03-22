# Meta-Risk Control Campaign

## Scope

- Entry signal logic unchanged (overlay only on risk size / trading permission).
- No look-ahead in control state updates (strict chronological daily updates).
- IS/OOS split preserved; ATR filter thresholds calibrated on IS and frozen on OOS.

## Dataset & Split

- Dataset: `MNQ_1mim.parquet`
- Sessions total: 1761 (2019-05-06 -> 2026-03-02)
- IS sessions: 1232
- OOS sessions: 529 (from 2024-02-12)

## Baseline Strategy (unchanged signal)

- OR minutes: `15`
- Side mode: `long_only`
- Target multiple: `5.0`
- Stop buffer ticks: `2`
- Risk per trade (base): `0.25%`
- EMA filter: `ema_only` / EMA30
- ATR IS band: `atr_14` in [8.3214, 102.3929]

## Variants

- 1_baseline
- 2_half_after_2_losses
- 3_skip_after_3_losses
- 4_local_drawdown_scaling

## Ranking Priority

1) challenge pass rate
2) median days to +6%
3) drawdown
4) Sharpe/PF/expectancy

## Final Ranking

```text
 selection_rank            variant_label  challenge_pass_rate  challenge_median_days_to_target  oos_max_drawdown  oos_sharpe  oos_profit_factor  oos_expectancy  oos_pnl  oos_trades
              1               1_baseline                0.173                            423.0            -779.0    0.970865           1.579544       30.607143   1714.0          56
              2    3_skip_after_3_losses                0.173                            423.0            -779.0    0.970865           1.579544       30.607143   1714.0          56
              3    2_half_after_2_losses                0.000                              NaN             -61.5   -0.690849           0.000000      -61.500000    -61.5           1
              4 4_local_drawdown_scaling                0.000                              NaN             -61.5   -0.690849           0.000000      -61.500000    -61.5           1
```

## Winner

- Selected: `1_baseline` (pass_rate=17.30%, median_days=423.0, OOS_DD=-779.00).

## Honest Conclusion

- Le pass rate challenge reste faible sur toutes les variantes: amélioration relative oui, robustesse absolue encore limitée.

## Full Comparative Table

```text
           variant_label  overall_pnl  overall_trades  overall_win_rate  overall_profit_factor  overall_expectancy  overall_sharpe  overall_max_drawdown  oos_pnl  oos_trades  oos_win_rate  oos_profit_factor  oos_expectancy  oos_sharpe  oos_max_drawdown  challenge_pass_rate  challenge_median_days_to_target
              1_baseline       4457.5             261          0.448276               1.326246           17.078544        0.722458                -934.5   1714.0          56      0.446429           1.579544       30.607143    0.970865            -779.0                0.173                            423.0
   2_half_after_2_losses        533.5              66          0.393939               1.154213            8.083333        0.184553                -934.5    -61.5           1      0.000000           0.000000      -61.500000   -0.690849             -61.5                0.000                              NaN
   3_skip_after_3_losses       4580.5             260          0.450000               1.338294           17.617308        0.743469                -934.5   1714.0          56      0.446429           1.579544       30.607143    0.970865            -779.0                0.173                            423.0
4_local_drawdown_scaling       -156.0              44          0.409091               0.928160           -3.545455       -0.082173                -695.0    -61.5           1      0.000000           0.000000      -61.500000   -0.690849             -61.5                0.000                              NaN
```
