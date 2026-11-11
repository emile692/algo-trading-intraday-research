# EIA Natural Gas Storage — Phase 1 data audit

Audit date: 2026-10-03 (Europe/Paris)

## Status

Phase 1 has been completed on a conservative calendar subset. NG one-minute
data are locally available; QG/MNG bars and a contemporaneous consensus history
are not. The event study therefore retains only standard Thursday 10:30 ET
releases confirmed by dated EIA dashboard archives and excludes uncertified
historical holiday exceptions.

## Local market data found

- `algo-trading-macro-research/data/raw/topstep_fut_p1.csv` through
  `topstep_fut_p6.csv` contain Bloomberg-style continuous `NG1 Comdty` daily
  OHLCV observations from 2000-01-04 through 2026-09-15.
- The daily series is useful only as a coverage cross-check. It cannot separate
  the pre-release morning, the 10:30 ET reaction, and the rest of the session.
- `data/downloaded/NG_v_0_1m_20200101_20261003.parquet` now contains
  volume-ranked continuous `NG.v.0` one-minute bars from 2020-01-01 23:00 UTC
  through 2026-10-02 20:59 UTC.
- The NG file contains 2,155,221 rows, zero duplicate timestamps, 82 instrument
  IDs, and occupies 31,229,863 bytes. Its SHA-256 is
  `c0521eb695a513df12545c27980ea82948189d3b34200ccdafd60ce5721ce310`.
- Of 351 Thursdays represented around the announcement window, 348 contain all
  26 bars from 10:25 through 10:50 ET. Broader 08:00–16:00 coverage has a
  480-bar median; 45 of 1,744 sessions fall below 80% coverage.
- Databento warned that several 2020 dates have reduced dataset quality. These
  dates must be identified from dataset-condition metadata and flagged before
  estimating the event study.
- No QG or MNG intraday file was downloaded.
- Existing one-minute futures files under this repository cover 6E, MCL, MES,
  MGC, MNQ, NQ, ZF, and ZN, not natural gas.

## Calendar and surprise data found

- The local economic calendar contains no EIA Natural Gas Storage event.
- Dated official EIA dashboard archives confirm 292 standard Thursday 10:30 ET
  releases through 2025; 289 have every required NG price point.
- Historical holiday exceptions are not consistently represented by the dated
  dashboard PDFs. They are excluded instead of inferred.
- No local actual/consensus storage-surprise history was found. EIA historical
  storage values supply actuals, but consensus is a third-party point-in-time
  field and is not supplied by EIA.

## Existing acquisition path

`src/data/downloader.py` was used with Databento `GLBX.MDP3`, continuous
symbology, and `ohlcv-1m`. One paid `get_range` request was made for NG after a
fresh cost check. Estimated cost was USD 7.868247, below the USD 8.00 hard
guard. The temporary API key was held only in process memory and was removed
from the environment after the request.

## Remaining inputs for any Phase 2

1. A complete point-in-time release archive including historical holiday
   exceptions and their exact release times.
2. Actual and contemporaneous consensus in Bcf, with vendor/source and revision
   semantics documented, if surprise conditioning is required.
3. QG and MNG one-minute bid/ask or trade bars for empirical spread, liquidity,
   and realized-slippage comparisons.

## Integrity decision

The retained sample has 2,155,221 NG bars, zero duplicate timestamps, 82
underlying instrument IDs, and no intraday contract-identity change on a
retained event. Eight overnight event observations crossing a continuous roll
are excluded from that window. The Phase 1 terminal label is `WEAK_EDGE`; see
`PHASE1_REPORT.md` for estimates and rationale.

Official references:

- EIA release schedule: https://ir.eia.gov/ngs/schedule.html
- EIA methodology: https://ir.eia.gov/ngs/methodology.html
- CME NG/QG/MNG comparison: https://www.cmegroup.com/articles/faqs/micro-henry-hub-natural-gas-futures-and-options-frequently-asked-questions.html

