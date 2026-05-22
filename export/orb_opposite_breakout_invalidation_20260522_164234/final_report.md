# ORB Opposite Breakout Invalidation Campaign

## 1. Executive summary

- Symbols tested: `MNQ`
- Period requested: `2018-01-01` -> `2026-12-31`
- Policies tested: 7
- Verdict: **use reclaim strategy separately**

## 2. Verdict clair

- use reclaim strategy separately

## 3. Best config overall

- Config: `allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_true`
- Family: `reclaim_conservative`
- Strategy tag: `failed_breakdown_reclaim_long`
- OOS Sharpe: -0.436
- OOS net pnl: -104.50
- Robust score: -2.595

## 4. Best config per asset

- MNQ: `invalidate_on_opposite_touch__buffer_0` | net `36.00` | Sharpe `0.154` | DD `-453.00`

## 5. Baseline decomposition

- MNQ / first_breakout_upside: trades 4, net -104.50, Sharpe -1.383
- MNQ / no_downside_breakout_before_trade: trades 4, net -104.50, Sharpe -1.383

## 6. Impact of removing trades after downside first breakout

- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_true: baseline pnl -104.50, filtered pnl -104.50, removed pnl 0.00, trades removed 0, Sharpe delta 0.000, DD delta 0.00, daily loss breach delta 0, PF delta 0.000
- allow_reclaim_after_opposite_breakout_strict__buffer_0__reclaim_1: baseline pnl -104.50, filtered pnl -104.50, removed pnl 0.00, trades removed 0, Sharpe delta 0.000, DD delta 0.00, daily loss breach delta 0, PF delta 0.000
- baseline_no_opposite_invalidation: baseline pnl -104.50, filtered pnl -104.50, removed pnl 0.00, trades removed 0, Sharpe delta 0.000, DD delta 0.00, daily loss breach delta 0, PF delta 0.000
- invalidate_on_opposite_close_1m__buffer_0: baseline pnl -104.50, filtered pnl 36.00, removed pnl -140.50, trades removed 1, Sharpe delta 0.590, DD delta 0.00, daily loss breach delta 0, PF delta 0.256
- invalidate_on_opposite_close_5m__buffer_0: baseline pnl -104.50, filtered pnl 36.00, removed pnl -140.50, trades removed 1, Sharpe delta 0.590, DD delta 0.00, daily loss breach delta 0, PF delta 0.256

## 7. Robustness by year

- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_true / MNQ / 2026: net -104.50, trades 4, maxDD -453.00, invalidations 0
- allow_reclaim_after_opposite_breakout_strict__buffer_0__reclaim_1 / MNQ / 2026: net -104.50, trades 4, maxDD -453.00, invalidations 0
- baseline_no_opposite_invalidation / MNQ / 2026: net -104.50, trades 4, maxDD -453.00, invalidations 0
- invalidate_on_opposite_close_1m__buffer_0 / MNQ / 2026: net 36.00, trades 3, maxDD -453.00, invalidations 21
- invalidate_on_opposite_close_5m__buffer_0 / MNQ / 2026: net 36.00, trades 3, maxDD -453.00, invalidations 20
- invalidate_on_opposite_n_closes_1m__buffer_0__confirm_2 / MNQ / 2026: net 36.00, trades 3, maxDD -453.00, invalidations 20
- invalidate_on_opposite_touch__buffer_0 / MNQ / 2026: net 36.00, trades 3, maxDD -453.00, invalidations 23

## 8. Prop-firm risk view

- allow_reclaim_after_opposite_breakout_conservative__buffer_0__require_vwap_true: prop_pass=False, daily_loss_limit_breaches=0
- allow_reclaim_after_opposite_breakout_strict__buffer_0__reclaim_1: prop_pass=False, daily_loss_limit_breaches=0
- baseline_no_opposite_invalidation: prop_pass=False, daily_loss_limit_breaches=0
- invalidate_on_opposite_close_1m__buffer_0: prop_pass=False, daily_loss_limit_breaches=0
- invalidate_on_opposite_close_5m__buffer_0: prop_pass=False, daily_loss_limit_breaches=0

## 9. Recommendation for live execution repo

- Keep the execution repo ORB baseline unchanged unless the selected policy improves OOS Sharpe and drawdown together.
- If the best reclaim variant remains competitive, wire it as a separate strategy family rather than mixing it into the ORB quality filter.

## 10. Suggested YAML fields for execution config

### Best overall policy

```yaml
opposite_breakout_policy:
  enabled: true
  mode: reclaim_required
  confirmation: touch
  buffer_ticks: 0
  require_reclaim_or_low: true
  require_reclaim_vwap: true
  reclaim_confirm_bars: 1
```

### Best reclaim policy

```yaml
opposite_breakout_policy:
  enabled: true
  mode: reclaim_required
  confirmation: touch
  buffer_ticks: 0
  require_reclaim_or_low: true
  require_reclaim_vwap: true
  reclaim_confirm_bars: 1
```
