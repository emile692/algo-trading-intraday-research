# 6J Gotobi Phase 1 — preflight data audit

## Current status

The 6J one-minute data were acquired in one guarded Databento request after a
cost estimate. Phase 1A, Phase 1B, and the conditional Phase 1C are complete;
the terminal verdict is `PROMISING_FOR_VALIDATION`.

No frozen sleeve, prior hypothesis, strategy registry entry, or portfolio file
was modified.

## Market data

- `6J.v.0`: 4,054,663 one-minute bars from 2015-01-01 23:00 UTC through
  2026-10-02 20:59 UTC, zero duplicate timestamps, 48 underlying contracts.
- Paid request count: one; estimated/final billed data cost USD 14.802701 under
  a hard USD 14.85 guard. The API key was not persisted.
- SHA-256: `b6846aea9cbb8cb199b805595f6395e876984dbe67fee7393418c0c888e2f2c4`.
- Databento dataset-condition records are stored locally. Endpoints on degraded
  dates are rejected per window.
- Existing continuous files use Databento `*.v.0`: the highest-volume contract
  is selected point in time. The Gotobi spec retains `instrument_id` and rejects
  any window crossing a mapping transition; it does not back-adjust prices.

## Repository components inspected

- `src/data/downloader.py`: Databento acquisition, cost estimation, parquet
  persistence, and coverage diagnostics.
- `src/data/session.py`: generic local-time filtering, but no Japan-to-CME
  session mapping; the Gotobi campaign uses a dedicated tested mapping.
- `src/engine/execution_model.py`: fixed slippage and commission support, but its
  repository defaults are MNQ-specific and must not be reused implicitly for 6J.
- The frozen Gotobi specification uses the explicit 6J tick and current
  TopstepX round-turn cost.

## Temporal and tradability decision

- 6J is USD per JPY; a rise in USD/JPY is a fall in 6J.
- Tokyo uses `Asia/Tokyo` without DST. Chicago uses `America/Chicago` with DST.
- The 09:55 JST fix occurs on the prior Chicago calendar date at 18:55 CST or
  19:55 CDT, inside the CME/Topstep session carrying the Japanese business date.
- All requested windows are structurally inside Topstep's 17:00–15:10 CT
  permitted session. Actual bar presence remains mandatory because CME holiday
  closures can still make a particular observation non-tradable.

## Calendar

The Japanese business-day calendar uses the Cabinet Office official holiday
CSV plus the Bank of Japan closure from December 31 through January 3. A Gotobi
anchor on a nonbusiness day moves to the preceding Japanese bank business day.
February receives no synthetic day 30; month-end is flagged separately.

## Integrity decision

No interpolation or constant-offset timezone conversion is used. Each measured
window requires both one-minute endpoints, non-degraded source dates, and the
same `instrument_id`. Full estimates and the stopping decision are in
`PHASE1_REPORT.md`.
