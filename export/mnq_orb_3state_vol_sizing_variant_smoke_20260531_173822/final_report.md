# MNQ ORB 3-State Vol-Sizing Variant Smoke

## Objective

Quickly test whether the retained MNQ ORB 3-state vol sizing stays robust around `fast=15 / slow=60`,
while changing only the vol-ratio construction and keeping the ORB signal, entries, exits, costs, sessions, filters, invalidation, and risk model unchanged.

- Reference export reused: `D:\Business\Trading\VSCODE\algo-trading-intraday-research\data\exports\mnq_orb_regime_filter_sizing_20260325_150405`
- Symbol: `MNQ`
- Dataset: `D:\Business\Trading\VSCODE\algo-trading-intraday-research\data\processed\parquet\MNQ_c_0_1m_20260321_094501.parquet`
- Aggregation rule: `majority_50`
- Fixed three-state multipliers for every variant: `low=0.50x`, `mid=1.00x`, `high=0.25x`

## Ranked By OOS Sharpe

| variant_name | sharpe | net_pnl | max_drawdown | profit_factor | num_trades | delta_sharpe_vs_single_15_60 |
| --- | --- | --- | --- | --- | --- | --- |
| median_plateau_compact | 2.688 | 31710.000 | -2774.000 | 1.924 | 263 | 0.274 |
| median_fast15_slow_60_70_80 | 2.562 | 30938.000 | -2774.000 | 1.859 | 263 | 0.149 |
| single_15_70 | 2.501 | 29719.000 | -2774.000 | 1.840 | 261 | 0.087 |
| single_15_60 | 2.414 | 27959.000 | -2420.500 | 1.801 | 257 | 0.000 |
| single_14_60 | 2.413 | 27546.500 | -2339.000 | 1.791 | 261 | -0.001 |
| single_15_80 | 2.410 | 29049.000 | -2774.000 | 1.786 | 262 | -0.004 |
| single_16_60 | 2.380 | 27493.500 | -2420.500 | 1.785 | 257 | -0.034 |
| single_16_75 | 2.166 | 25095.500 | -2692.500 | 1.678 | 262 | -0.248 |

## Ranked By Prop-Safe Robustness

| variant_name | pass_prop_constraints | max_loss_limit_buffer_usd | daily_loss_limit_breach_freq | sharpe | max_drawdown |
| --- | --- | --- | --- | --- | --- |
| single_14_60 | yes | 1017.500 | 0.000 | 2.413 | -2339.000 |
| single_15_80 | yes | 770.000 | 0.000 | 2.410 | -2774.000 |
| median_plateau_compact | yes | 626.500 | 0.000 | 2.688 | -2774.000 |
| median_fast15_slow_60_70_80 | yes | 256.500 | 0.000 | 2.562 | -2774.000 |
| single_15_70 | yes | 256.500 | 0.000 | 2.501 | -2774.000 |
| single_16_75 | yes | 256.500 | 0.000 | 2.166 | -2692.500 |
| single_16_60 | yes | 217.000 | 0.000 | 2.380 | -2420.500 |
| single_15_60 | yes | 166.000 | 0.000 | 2.414 | -2420.500 |

## Explicit Comparison Vs `single_15_60`

- Baseline `single_15_60`: Sharpe `2.414`, net PnL `27959.0`, max drawdown `-2420.5`.
- Nearby single variants span about Sharpe `0.335` and net PnL `4623.5` OOS, which is a practical read on how flat the 15/60 neighborhood is.
- Final verdict: passer à `median_plateau_compact`.
- Rationale: median_plateau_compact keeps or improves OOS Sharpe versus single_15_60 (2.688 vs 2.414) while improving the risk/stability profile.

## OOS Bucket Contribution Snapshot

| variant_name | bucket_label | risk_multiplier | num_trades | net_pnl | avg_trade_pnl |
| --- | --- | --- | --- | --- | --- |
| median_fast15_slow_60_70_80 | high | 0.250 | 31 | 679.500 | 21.919 |
| median_fast15_slow_60_70_80 | low | 0.500 | 127 | 4192.500 | 33.012 |
| median_fast15_slow_60_70_80 | mid | 1.000 | 105 | 26066.000 | 248.248 |
| median_plateau_compact | high | 0.250 | 31 | 388.000 | 12.516 |
| median_plateau_compact | low | 0.500 | 132 | 4340.000 | 32.879 |
| median_plateau_compact | mid | 1.000 | 100 | 26982.000 | 269.820 |
| single_14_60 | high | 0.250 | 33 | 523.000 | 15.848 |
| single_14_60 | low | 0.500 | 134 | 5757.000 | 42.963 |
| single_14_60 | mid | 1.000 | 94 | 21266.500 | 226.239 |
| single_15_60 | high | 0.250 | 31 | 637.500 | 20.565 |
| single_15_60 | low | 0.500 | 128 | 4649.500 | 36.324 |
| single_15_60 | mid | 1.000 | 98 | 22672.000 | 231.347 |
| single_15_70 | high | 0.250 | 33 | 460.500 | 13.955 |
| single_15_70 | low | 0.500 | 127 | 4192.500 | 33.012 |
| single_15_70 | mid | 1.000 | 101 | 25066.000 | 248.178 |
| single_15_80 | high | 0.250 | 30 | 302.000 | 10.067 |
| single_15_80 | low | 0.500 | 129 | 3472.500 | 26.919 |
| single_15_80 | mid | 1.000 | 103 | 25274.500 | 245.383 |
| single_16_60 | high | 0.250 | 33 | 558.000 | 16.909 |
| single_16_60 | low | 0.500 | 123 | 3972.500 | 32.297 |
| single_16_60 | mid | 1.000 | 101 | 22963.000 | 227.356 |
| single_16_75 | high | 0.250 | 31 | 705.000 | 22.742 |
| single_16_75 | low | 0.500 | 125 | 3720.500 | 29.764 |
| single_16_75 | mid | 1.000 | 106 | 20670.000 | 195.000 |
