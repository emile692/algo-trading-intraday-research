# R1 Conditional VWAP mean reversion: closure

**Status:** `REJECTED`. All 18 pre-registered central cells fail the IS gate; none passes G1.
**Date:** 2026-11-11
**Spec:** [FROZEN_SPECIFICATION.json](FROZEN_SPECIFICATION.json), sha256 `a0742a03…af40f` ([SPEC_LOCK.json](SPEC_LOCK.json)). It was frozen before any post-signal return was computed, and a synthetic self-test passed first.
**OOS (2024-01 → 2026-03):** not computed. MES stage 2 was not run. Diversification was not computed (there is no candidate).

## Hypotheses

Three hypotheses, each tested with two definitions, on MNQ, MGC and MCL separately.

- **H1, VWAP deviation reversion.** Two normalisations:
  - N1: VWAP standard-deviation bands;
  - N2: daily volatility × √(elapsed time).

  Both with an efficiency-ratio trend filter.
- **H2, volatility exhaustion.** A 30-minute move of at least 2.5 σ, then a deceleration:
  - D1: volatility contraction;
  - D2: a stall bar.
- **H3, lunch-hour reversion** (11:30–13:30), on low volume, towards the VWAP or the session-range midpoint.

Common rules:
- Execution: 1-minute bars; market entry at the next minute + 1 tick; stop at the worse of the stop and the open, + 1 tick; limit target filled only on a 1-tick trade-through; stop first within a minute.
- Costs and size: 1.25 USD commission per side, 1 contract.
- Selection: one pre-registered central cell per test, with the grid used only for stability.
- Holdouts: NQ 2010–2019 and CL 2015–2021.

## Results (IS central cells)

| | H1-N1 | H1-N2 | H2-D1 | H2-D2 | H3-VWAP | H3-RANGE |
|---|---|---|---|---|---|---|
| MNQ net USD (t) | −6 690 (−2.24) | −5 982 (−2.37) | +42 (0.03) | −115 (−0.11) | −1 069 (−0.82) | −5 230 (−2.71) |
| MGC | −1 882 (−2.10) | +32 (0.04) | −860 (−1.26) | −1 385 (−1.92) | −542 (−1.61) | −1 036 (−1.86) |
| MCL | −1 902 (−1.70) | −2 794 (−3.16) | −989 (−1.56) | −62 (−0.13) | −1 509 (−2.48) | −2 494 (−2.77) |
| NQ 2010–19 (MNQ economics) | −5 101 (−5.8) | −3 791 (−4.1) | −2 269 (−4.4) | −2 342 (−5.5) | −1 334 (−3.2) | −4 190 (−5.8) |
| CL 2015–21 (MCL economics) | −5 709 (−6.8) | −1 601 (−2.5) | −2 082 (−4.4) | −1 649 (−3.8) | −1 896 (−4.1) | −5 237 (−6.9) |

- Positive after costs: 11 of 126 IS cells (best t 1.07) and 1 of 84 PRE cells.
- All 12 central PRE cells are negative before costs.
- Holm-adjusted p is 1.0 for all 18 tests.

## Reading

There is no exploitable intraday reversion on these futures.
- **MNQ:** price far from VWAP continues more often than it reverts (H1 gross −0.03 to −0.05 R), with or without a trend filter.
- **MGC:** a small gross reversion (+0.02 to +0.07 R) exists but is smaller than a round trip (≈ 0.08 R).
- **MCL:** negative before costs.
- **Lunch hour and low volume** do not make the price more mean-reverting.
- **MNQ H2-D1** is the only timing effect (placebo p 0.018, gross +0.15 R). It nets zero, fails at double cost, and is negative on NQ 2010–2019 (t −4.4).

## Reopening

This is not a reason to try another threshold, normalisation, window or filter. Reopening requires new information (order book, flow or tick data) in a new pre-registered campaign. MNQ H2-D1 with the volume fade (+910 USD, t 0.67) was seen post hoc and must not be adopted. Full report: [final_report.md](final_report.md).
