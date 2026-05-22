# Hybrid Execution Validation Report

## A. Added
- Added an explicit hybrid backtest mode on the V2 engine with `execution_timeframe="1min"`.
- Added `entry_timing` support for `next_execution_bar_open` and `same_timestamp_execution_open`.
- Added `protective_orders_active_from` support for `after_entry_fill` and `next_execution_bar`.
- Added a dedicated validation campaign exporting baseline and hybrid trade/metric comparisons.

## B. Exact Semantics
- Signal timeframe remains `1H`; execution timeframe is `1min` for hybrid runs on `MNQ`.
- `signal_actionable_time` is the close of the setup bar, which equals the timestamp of the signal row built on left-labeled 1H bars.
- `next_execution_bar_open` enters on the first 1min bar strictly after `signal_actionable_time`.
- `same_timestamp_execution_open` enters on the 1min bar at the same timestamp when present, otherwise the next available 1min bar.
- `after_entry_fill` activates stop/target logic on the entry minute bar itself after the fill; `next_execution_bar` activates them from the following 1min bar.
- If stop and target are both touched inside the same 1min bar, the engine uses the pessimistic `stop_first` convention and logs `stop_ambiguous_first_1m`.
- `time_stop_bars` keeps its 1H business meaning and is converted to `entry_time + N * 60 minutes`; the log stores the exact `time_stop_at` timestamp.
- EOD flat uses the last available RTH minute after the repo's inclusive `09:30 <= timestamp <= 16:00` filter. Current last minute in this run: `2026-03-19 16:00:00-04:00`.
- Daily loss remains a post-trade research metric only; no active intraday daily-loss guard was added in this version.

## C. Files
- Modified: `src/engine/volume_climax_pullback_v2_backtester.py`.
- Added: `src/analytics/volume_climax_pullback_hybrid_execution_validation.py`.
- Added tests covering 1min entry timing, stop/target ambiguity, time stop conversion and EOD flattening.

## D. Tests
- Unit tests are executed separately in the implementation turn and reported from the pytest output.

## E. Exports
- `trades_baseline_1h.csv`
- `trades_hybrid_after_entry_fill.csv`
- `trades_hybrid_next_execution_bar.csv`
- `trade_comparison.csv`
- `metrics_comparison.csv`
- `final_report.md`
- `run_metadata.json`

## F. First Conclusions
- Baseline trades: `315`.
- Hybrid `after_entry_fill` trades: `275` with `323` divergences vs baseline.
- Hybrid `next_execution_bar` trades: `275` with `323` divergences vs baseline.
- Baseline net PnL: `7535.12` USD.
- Hybrid `after_entry_fill` net PnL: `-1675.73` USD.
- Hybrid `next_execution_bar` net PnL: `-1567.28` USD.

## Context
- Dataset: `D:\Business\Trading\VSCODE\algo-trading-intraday-research\data\processed\parquet\MNQ_c_0_1m_20260321_094501.parquet`.
- Variant: `dynamic_exit_atr_target_1p0_ts2_vq0p95_bf0p5_ra1p2`.
- Signal rows: `12035`.
- RTH minute rows: `671037`.
