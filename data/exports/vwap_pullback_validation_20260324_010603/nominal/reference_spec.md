# Reference Spec

## Provenance

- Source run metadata: `D:\Business\Trading\VSCODE\algo-trading-intraday-research\data\exports\vwap_full_smoke_v2\summary\run_metadata.json`
- Source best variant: `vwap_pullback_continuation`
- Source run timestamp: `2026-03-23T00:25:52.630068`

## Audited Strategy Definition

- Strategy name: `vwap_pullback_continuation`
- Time windows: `full_rth`
- Slope lookback / threshold: `5` / `0.0`
- ATR period / buffer / stop buffer: `14` / `0.3` / `0.3`
- Pullback lookback: `8`
- Confirmation threshold: `0.0` ATR above/below `prev_high` / `prev_low`.
- Pullback definition: at least one counter-trend close inside the last pullback window, while the pullback extreme stays within the VWAP regime buffer.
- Confirmation definition: close-based continuation through `prev_high` / `prev_low`, executed on the next bar open.
- Stop logic: pullback extreme +/- `stop_buffer * ATR`.
- Exit logic: VWAP recross = `True`, plus structural stop, plus forced session close.
- Max trades per day: `3`
- Daily kill switches: `max_losses_per_day=None`, `daily_stop_threshold_usd=None`
- Sizing: `fixed_quantity`, fixed quantity `1`, risk per trade `None`
- Costs: commission `1.25` USD / side, slippage `1` tick(s) / side.
- Session assumptions: RTH `[09:30, 16:00)`, flat overnight, dataset `MNQ_c_0_1m_20260321_094501.parquet`.

## Audit Warnings

- `compression_length` is present in the config object but is not consumed by `generate_pullback_continuation_signals`; it is documented as inactive for this variant.
- `use_partial_exit`, `partial_exit_r_multiple`, and `keep_runner_until_close` are currently inactive in the VWAP discrete backtester and are excluded from robustness conclusions.
- The source discovery run selected `vwap_pullback_continuation` as best variant, but that legacy run used same-bar execution for close-based discrete entries. Validation reruns therefore use the corrected next-open semantics and should be treated as the only defendable evidence.
