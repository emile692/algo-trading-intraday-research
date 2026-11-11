# ZF data audit

| Field | Result |
|---|---|
| Source | Databento GLBX.MDP3 |
| Symbol | `ZF.v.0` |
| Construction | Volume-ranked front month |
| Resolution | 1 minute, timestamp at bar start |
| Coverage | 2013-05-17 00:00 UTC — 2026-09-18 20:59 UTC |
| Rows | 4,069,036 |
| Timezone | UTC; session audit in America/New_York |
| Audit session | 08:20 inclusive — 15:00 exclusive |
| Sessions | 3,439 |
| Median bars | 399 / 400 |
| Sessions below 80% | 123 (3.58%) |
| Duplicates | 0 |
| Invalid OHLC | 0 |
| Missing required values | 0 |
| Zero-volume rows | 0 |
| Underlying instrument IDs | 55 |
| Roll transitions | 56 |
| Median / p95 / max absolute roll jump | 0.326% / 0.709% / 0.954% |
| Back-adjusted | No |

Databento's daily condition endpoint reports 32 degraded GLBX.MDP3 dates in the requested range. Two 5Y→ZF auction dates coincide with them. Excluding every degraded auction date changes the 5Y combined effect from +2.24 to +2.31 bps and does not affect the decision.

No canonical auction window crosses a detected `instrument_id` transition, so roll discontinuities do not explain the measured event effects.
