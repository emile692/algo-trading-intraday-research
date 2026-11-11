# H1 Fast Alpha — Phase 0 audit

The experiment is frozen before performance measurement. The six source sleeves are read-only and are identified by `research/portfolio_integration_phase1/portfolio_integration_phase1_config.json`; their canonical reproduced trades remain in `research/portfolio_integration_phase1/sleeve_trades/`.

## Data and timestamps

- MNQ: local `c.0` one-minute data; 09:30–16:00 America/New_York research session.
- MGC: local `v.0` one-minute data; 08:20–13:30 America/New_York research session, matching the frozen COMEX cash-close signal window.
- Input timestamps label the start of UTC one-minute bars. Conversion uses the IANA timezone, including DST.
- Five-minute intervals are left-closed/right-open and anchored to the session start. A bar starting 09:30 contains 09:30–09:34 minute bars and becomes observable at 09:35.
- No order based on a completed bar may execute before the next one-minute open at that completion timestamp.
- Runs reset across sessions, missing aggregate intervals, zero-return bars and contract changes. Overnight/session gaps are never interpreted as five-minute returns.
- An aggregate bar with trades but fewer than five printed one-minute bars remains a valid trade-price OHLC bar; entirely empty intervals are absent and break the run. No price is forward-filled.

## Costs and overlay ambiguity

Frozen repository assumptions are one adverse tick plus $1.25 commission per side: $3.50 round trip for MNQ and $4.50 for MGC. Phase 1 tests predictability gross because H1 is informational, not a standalone trade.

The Concretum paper specifies waiting for one opposite five-minute bar but does not define a maximum wait; it explicitly calls forced urgency controls outside its scope. Therefore no timeout is selected. If Gate 1 passes, the first sleeve-level analysis is an event study ending at each original frozen exit; absence of an opposite completed bar before that exit means an unfilled/missed entry.

Full assumptions and the predeclared Gate 1 criteria are in `frozen_config.json`.
