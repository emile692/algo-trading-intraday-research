# 6J month-end Gotobi — Phase 2 data audit

## Scope and freeze

- Rule: short one 6J contract at 09:31 JST and cover at 09:54 JST on the Phase-1 Japanese month-end calendar.
- Phase-1 gate reproduced before continuing: 136 trades; USD 31.0202 gross expectancy; USD 14.3002 net expectancy; USD 1,944.83 net total.
- No alternate time, general Gotobi rule, post-fix rule, filter, or subgroup was promoted.
- The six existing sleeves were read only for correlations and were not modified.

## Explicit-contract reconstruction

- Each Phase-1 `instrument_id` was resolved to a raw 6J quarterly symbol as-of the event date.
- 136 event-level MBP-1 files and 136 same-date instrument-definition snapshots are present.
- Every MBP-1 and definition ID matches the continuous series' point-in-time underlying ID.
- The instrument definition supplies the actual expiration timestamp; no expiry was inferred from the ticker.
- Explicit bar-open PnL equals Phase-1 continuous PnL on 135 events and differs by exactly one tick on 2018-07-31. No difference exceeds one tick. No back adjustment is used.

## Execution data

- Download scope: only 30 seconds before 09:31 JST through 30 seconds after 09:54 JST for each of the 136 frozen events.
- MBP-1 estimate: 4,225,797 records, 338,063,760 billable bytes, USD 0.566724. The combined pre-download estimate including definitions was USD 0.566812 under a hard USD 0.60 guard.
- Fill rule: first valid displayed bid at or after entry; first valid displayed ask at or after exit. Mid fills are never used.
- Top-of-book sizes, quote delays, boundary spreads, explicit trades, and traded volume are retained in `explicit_contract_trades.csv`.
- The API key was process-only, removed after acquisition, and is not stored in an output or script.

## Known limits

- MBP-1 top depth is displayed level-0 depth, not full queue-position reconstruction. A one-contract marketable order is evaluated against that displayed depth.
- The 2024–2026 block is a frozen retrospective block, not a genuinely unseen holdout, because Phase 1 had already inspected the complete history.
- Full-session volume is calculated from the unadjusted point-in-time continuous mapping for the verified active instrument; event-window volume comes directly from explicit MBP-1 trade records.
