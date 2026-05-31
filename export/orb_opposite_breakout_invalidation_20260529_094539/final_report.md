# ORB Opposite Breakout Invalidation Campaign

## 1. Executive summary

- Symbols tested: `MNQ`
- Period requested: `2018-01-01` -> `2026-12-31`
- Policies tested: 17
- Run status: `completed`
- Configs completed: 17/17
- Verdict: **invalidate on opposite breakout**

## 2. Verdict clair

- invalidate on opposite breakout

## 3. Best config overall

- Config: `invalidate_on_opposite_n_closes_1m__buffer_2__confirm_3`
- Family: `invalidate_for_day`
- Strategy tag: `orb_long_only`
- OOS Sharpe: 1.633
- OOS net pnl: 19252.60
- Robust score: 2.275

## 4. Best config per asset

- MNQ: `invalidate_on_opposite_n_closes_1m__buffer_0__confirm_2` | net `30191.76` | Sharpe `0.872` | DD `-6843.70`

## 5. Baseline decomposition

- MNQ / first_breakout_downside_then_reclaim: trades 207, net 1915.42, Sharpe 0.248
- MNQ / first_breakout_upside: trades 578, net 25494.22, Sharpe 1.051
- MNQ / no_downside_breakout_before_trade: trades 578, net 25494.22, Sharpe 1.051

## 6. Impact of removing trades after downside first breakout

- invalidate_on_opposite_n_closes_1m__buffer_2__confirm_3: baseline pnl 14973.08, filtered pnl 19252.60, removed pnl -2946.98, trades removed 169, Sharpe delta 0.466, DD delta 1813.06, daily loss breach delta 0, PF delta 0.050
- invalidate_on_opposite_close_5m__buffer_2: baseline pnl 14973.08, filtered pnl 18569.82, removed pnl 873.78, trades removed 176, Sharpe delta 0.427, DD delta 597.54, daily loss breach delta 0, PF delta 0.028
- invalidate_on_opposite_close_5m__buffer_4: baseline pnl 14973.08, filtered pnl 18569.82, removed pnl 873.78, trades removed 176, Sharpe delta 0.427, DD delta 597.54, daily loss breach delta 0, PF delta 0.028
- invalidate_on_opposite_close_5m__buffer_0: baseline pnl 14973.08, filtered pnl 18237.54, removed pnl 764.68, trades removed 179, Sharpe delta 0.399, DD delta 597.54, daily loss breach delta 0, PF delta 0.029
- invalidate_on_opposite_close_5m__buffer_1: baseline pnl 14973.08, filtered pnl 18237.54, removed pnl 1206.06, trades removed 177, Sharpe delta 0.399, DD delta 597.54, daily loss breach delta 0, PF delta 0.026

## 7. Robustness by year

- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_false / MNQ / 2019: net 405.86, trades 11, maxDD -1435.96, invalidations 0
- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_false / MNQ / 2020: net -1767.38, trades 118, maxDD -5211.46, invalidations 0
- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_false / MNQ / 2021: net 14741.92, trades 126, maxDD -3166.00, invalidations 0
- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_false / MNQ / 2022: net -1613.98, trades 114, maxDD -5190.16, invalidations 0
- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_false / MNQ / 2023: net 1236.36, trades 149, maxDD -7335.24, invalidations 0
- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_false / MNQ / 2024: net 14352.94, trades 141, maxDD -5951.28, invalidations 0
- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_false / MNQ / 2025: net 1221.26, trades 118, maxDD -7651.42, invalidations 0
- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_false / MNQ / 2026: net -601.12, trades 7, maxDD -690.72, invalidations 0
- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_true / MNQ / 2019: net 405.86, trades 11, maxDD -1435.96, invalidations 0
- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_true / MNQ / 2020: net -1020.42, trades 117, maxDD -4719.62, invalidations 0
- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_true / MNQ / 2021: net 14254.88, trades 125, maxDD -3166.00, invalidations 0
- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_true / MNQ / 2022: net -935.02, trades 113, maxDD -5190.16, invalidations 0
- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_true / MNQ / 2023: net 328.08, trades 148, maxDD -7403.84, invalidations 0
- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_true / MNQ / 2024: net 14845.82, trades 139, maxDD -5951.28, invalidations 0
- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_true / MNQ / 2025: net -234.52, trades 117, maxDD -7975.86, invalidations 0
- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_true / MNQ / 2026: net -601.12, trades 7, maxDD -690.72, invalidations 0
- baseline_no_opposite_invalidation / MNQ / 2019: net 405.86, trades 11, maxDD -1435.96, invalidations 0
- baseline_no_opposite_invalidation / MNQ / 2020: net -1767.38, trades 118, maxDD -5211.46, invalidations 0
- baseline_no_opposite_invalidation / MNQ / 2021: net 14741.92, trades 126, maxDD -3166.00, invalidations 0
- baseline_no_opposite_invalidation / MNQ / 2022: net -2180.20, trades 115, maxDD -5464.90, invalidations 0

## 8. Prop-firm risk view

- invalidate_on_opposite_n_closes_1m__buffer_2__confirm_3: prop_pass=False, daily_loss_limit_breaches=0
- invalidate_on_opposite_close_5m__buffer_2: prop_pass=False, daily_loss_limit_breaches=0
- invalidate_on_opposite_close_5m__buffer_4: prop_pass=False, daily_loss_limit_breaches=0
- invalidate_on_opposite_close_5m__buffer_0: prop_pass=False, daily_loss_limit_breaches=0
- invalidate_on_opposite_close_5m__buffer_1: prop_pass=False, daily_loss_limit_breaches=0

## 9. Recommendation for live execution repo

- Keep the execution repo ORB baseline unchanged unless the selected policy improves OOS Sharpe and drawdown together.
- If the best reclaim variant remains competitive, wire it as a separate strategy family rather than mixing it into the ORB quality filter.

## 10. Suggested YAML fields for execution config

### Best overall policy

```yaml
opposite_breakout_policy:
  enabled: true
  mode: invalidate_for_day
  confirmation: n_closes_1m
  buffer_ticks: 2
  confirm_bars: 3
  allow_reclaim: false
```

### Best reclaim policy

```yaml
opposite_breakout_policy:
  enabled: true
  mode: reclaim_required
  confirmation: touch
  buffer_ticks: 0
  require_reclaim_or_low: true
  require_reclaim_vwap: false
  reclaim_confirm_bars: 1
```

## 11. Run traceability

- Checkpoint results by symbol: `export\orb_opposite_breakout_invalidation_20260529_094539\checkpoint_results_by_symbol.csv`
- Checkpoint results by config: `export\orb_opposite_breakout_invalidation_20260529_094539\checkpoint_results_by_config.csv`
- Checkpoint trades by config: `export\orb_opposite_breakout_invalidation_20260529_094539\checkpoint_trades_by_config.csv`
