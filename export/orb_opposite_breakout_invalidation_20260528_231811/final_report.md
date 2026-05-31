# ORB Opposite Breakout Invalidation Campaign

## 1. Executive summary

- Symbols tested: `MNQ, MES, M2K, MGC`
- Period requested: `2018-01-01` -> `2026-12-31`
- Policies tested: 37
- Run status: `completed`
- Configs completed: 37/37
- Verdict: **invalidate on opposite breakout**

## 2. Verdict clair

- invalidate on opposite breakout

## 3. Best config overall

- Config: `invalidate_on_opposite_close_1m__buffer_1`
- Family: `invalidate_for_day`
- Strategy tag: `orb_long_only`
- OOS Sharpe: 0.324
- OOS net pnl: 4803.25
- Robust score: 0.421

## 4. Best config per asset

- M2K: `invalidate_on_opposite_close_1m__buffer_4` | net `-5293.00` | Sharpe `-0.279` | DD `-9954.50`
- MES: `invalidate_on_opposite_touch__buffer_0` | net `6460.00` | Sharpe `0.319` | DD `-4438.75`
- MGC: `invalidate_on_opposite_touch__buffer_0` | net `644.00` | Sharpe `0.042` | DD `-5498.00`
- MNQ: `invalidate_on_opposite_touch__buffer_0` | net `18677.50` | Sharpe `1.037` | DD `-2442.00`

## 5. Baseline decomposition

- M2K / first_breakout_downside_then_reclaim: trades 385, net -1415.00, Sharpe -0.284
- M2K / first_breakout_upside: trades 820, net -6164.50, Sharpe -0.541
- M2K / no_downside_breakout_before_trade: trades 820, net -6164.50, Sharpe -0.541
- MES / first_breakout_downside_then_reclaim: trades 539, net -12947.50, Sharpe -1.686
- MES / first_breakout_upside: trades 815, net 6016.25, Sharpe 0.470
- MES / no_downside_breakout_before_trade: trades 815, net 6016.25, Sharpe 0.470
- MGC / first_breakout_downside_then_reclaim: trades 349, net -11001.00, Sharpe -2.384
- MGC / first_breakout_upside: trades 512, net -1518.50, Sharpe -0.199
- MGC / no_downside_breakout_before_trade: trades 512, net -1518.50, Sharpe -0.199
- MNQ / first_breakout_downside_then_reclaim: trades 396, net -2305.50, Sharpe -0.450
- MNQ / first_breakout_upside: trades 721, net 17846.00, Sharpe 1.683
- MNQ / no_downside_breakout_before_trade: trades 721, net 17846.00, Sharpe 1.683

## 6. Impact of removing trades after downside first breakout

- invalidate_on_opposite_close_1m__buffer_1: baseline pnl 646.50, filtered pnl 4803.25, removed pnl -11489.75, trades removed 4537, Sharpe delta 0.288, DD delta 18090.25, daily loss breach delta 0, PF delta 0.068
- invalidate_on_opposite_touch__buffer_0: baseline pnl 646.50, filtered pnl 4972.25, removed pnl -32133.75, trades removed 1835, Sharpe delta 0.323, DD delta 20171.00, daily loss breach delta 0, PF delta 0.101
- invalidate_on_opposite_close_1m__buffer_0: baseline pnl 646.50, filtered pnl 4765.75, removed pnl -26414.00, trades removed 1605, Sharpe delta 0.289, DD delta 18932.50, daily loss breach delta 0, PF delta 0.076
- invalidate_on_opposite_touch__buffer_1: baseline pnl 646.50, filtered pnl 4635.25, removed pnl -11489.75, trades removed 4537, Sharpe delta 0.296, DD delta 18521.00, daily loss breach delta 0, PF delta 0.086
- invalidate_on_opposite_touch__buffer_4: baseline pnl 646.50, filtered pnl 4561.25, removed pnl -11489.75, trades removed 4537, Sharpe delta 0.285, DD delta 16132.75, daily loss breach delta 0, PF delta 0.077

## 7. Robustness by year

- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_false / M2K / 2019: net -2697.00, trades 118, maxDD -3552.00, invalidations 0
- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_false / M2K / 2020: net -1511.50, trades 177, maxDD -3091.00, invalidations 0
- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_false / M2K / 2021: net 1800.50, trades 177, maxDD -2488.00, invalidations 0
- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_false / M2K / 2022: net -1708.00, trades 177, maxDD -4125.50, invalidations 0
- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_false / M2K / 2023: net -689.50, trades 170, maxDD -3770.50, invalidations 0
- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_false / M2K / 2024: net -868.50, trades 166, maxDD -2585.00, invalidations 0
- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_false / M2K / 2025: net -1082.50, trades 182, maxDD -3463.00, invalidations 0
- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_false / M2K / 2026: net -823.00, trades 38, maxDD -1880.00, invalidations 0
- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_false / MES / 2019: net -4111.25, trades 132, maxDD -5547.50, invalidations 0
- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_false / MES / 2020: net -413.75, trades 204, maxDD -2231.25, invalidations 0
- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_false / MES / 2021: net 5521.25, trades 199, maxDD -1230.00, invalidations 0
- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_false / MES / 2022: net -2533.75, trades 189, maxDD -4777.50, invalidations 0
- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_false / MES / 2023: net -1216.25, trades 202, maxDD -3381.25, invalidations 0
- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_false / MES / 2024: net -1280.00, trades 199, maxDD -3786.25, invalidations 0
- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_false / MES / 2025: net -2137.50, trades 188, maxDD -3290.00, invalidations 0
- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_false / MES / 2026: net -522.50, trades 40, maxDD -968.75, invalidations 0
- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_false / MGC / 2018: net -6884.50, trades 94, maxDD -7380.00, invalidations 0
- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_false / MGC / 2019: net -810.00, trades 99, maxDD -2550.00, invalidations 0
- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_false / MGC / 2020: net 385.00, trades 107, maxDD -2609.50, invalidations 0
- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_false / MGC / 2021: net -2854.00, trades 103, maxDD -2955.50, invalidations 0

## 8. Prop-firm risk view

- invalidate_on_opposite_close_1m__buffer_1: prop_pass=False, daily_loss_limit_breaches=0
- invalidate_on_opposite_touch__buffer_0: prop_pass=False, daily_loss_limit_breaches=0
- invalidate_on_opposite_close_1m__buffer_0: prop_pass=False, daily_loss_limit_breaches=0
- invalidate_on_opposite_touch__buffer_1: prop_pass=False, daily_loss_limit_breaches=0
- invalidate_on_opposite_touch__buffer_4: prop_pass=False, daily_loss_limit_breaches=0

## 9. Recommendation for live execution repo

- Keep the execution repo ORB baseline unchanged unless the selected policy improves OOS Sharpe and drawdown together.
- If the best reclaim variant remains competitive, wire it as a separate strategy family rather than mixing it into the ORB quality filter.

## 10. Suggested YAML fields for execution config

### Best overall policy

```yaml
opposite_breakout_policy:
  enabled: true
  mode: invalidate_for_day
  confirmation: close_1m
  buffer_ticks: 1
  confirm_bars: 1
  allow_reclaim: false
```

### Best reclaim policy

```yaml
opposite_breakout_policy:
  enabled: true
  mode: reclaim_required
  confirmation: touch
  buffer_ticks: 1
  require_reclaim_or_low: false
  require_reclaim_vwap: false
  reclaim_confirm_bars: 2
```

## 11. Run traceability

- Checkpoint results by symbol: `export\orb_opposite_breakout_invalidation_20260528_231811\checkpoint_results_by_symbol.csv`
- Checkpoint results by config: `export\orb_opposite_breakout_invalidation_20260528_231811\checkpoint_results_by_config.csv`
- Checkpoint trades by config: `export\orb_opposite_breakout_invalidation_20260528_231811\checkpoint_trades_by_config.csv`
