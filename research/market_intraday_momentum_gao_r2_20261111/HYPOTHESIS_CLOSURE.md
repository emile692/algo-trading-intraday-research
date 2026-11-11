# R2 Market Intraday Momentum (Gao et al. 2018) on futures: closure

**Status:** `REJECTED`. The raw effect was not replicated; no instrument passed any of the seven gates.
**Date:** 2026-11-11
**Spec:** [FROZEN_SPECIFICATION.json](FROZEN_SPECIFICATION.json), sha256 `396c6536…70f098` ([SPEC_LOCK.json](SPEC_LOCK.json)). It was frozen before any return of the strategy was computed, and a synthetic self-test passed first.
**Data read:**
- Campaign A (no parameter): PRE, IS, OOS and EXT.
- Campaigns B and C: PRE and IS only. Their OOS was not computed, because the read rule was not met.

## Hypothesis

The return from the previous close (15:59 ET) to 10:00 ET predicts the sign of the 15:30 → 16:00 ET return.

- Rule: trade 1 contract in the direction of the first-half-hour return, from the 15:30 bar open to the 15:59 bar close.
- Costs: 1 tick plus commission per side.
- Instruments: MNQ, MES, MGC and MCL, each tested separately.
- Benchmarks: NQ 2010–2019, and CL 2015–2021 with MCL economics.

## Results

| | MNQ | MES | MGC | MCL |
|---|---|---|---|---|
| IS gross / cost / net (bp per session) | −0.08 / 1.51 / −1.59 | −0.23 / 2.67 / −2.90 | +0.34 / 2.54 / −2.20 | +0.25 / 5.48 / −5.23 |
| IS net NW t | −1.15 | −2.58 | −6.34 | −6.37 |
| IS slope b (t) | 0.056 (1.40) | 0.040 (1.04) | 0.007 (1.28) | 0.004 (0.65) |
| IS permutation p | 0.50 | 0.53 | 0.18 | 0.39 |
| OOS net bp (t) | −1.87 (−1.71) | −2.02 (−1.93) | −1.36 (−2.65) | −7.51 (−8.40) |
| OOS slope b (t) | −0.023 (−1.32) | −0.024 (−1.11) | 0.013 (1.81) | −0.020 (−1.56) |
| Gates passed (S1–S7) | 0/7 | 0/7 | 0/7 | 0/7 |
| Book Sharpe 1.75 with the strategy (OOS) | 1.50 | 1.58 | 1.64 | 1.60 |

NQ 2010–2019 has a slope of 0.011 (t 0.66) and a net of −1.17 bp (t −1.95). The MNQ IS slope comes from 2020 alone (t 2.29); without 2020 it is −0.013 (t −0.90). None of the 5 variants (Holm over 18 tests) or the 36 regime cells (Holm) is significant, and there is no CONDITIONAL candidate.

## Reading

There is no momentum from the first half hour to the last half hour on these futures, outside crisis years (2012, 2020). This is not a cost problem: gross is about zero and the gross hit rate is about 50 %. What little slope exists comes from the overnight gap, not from the 09:30–10:00 move. The paper's volume claim is reversed here: low volume shows the larger slope. Correlation with the book is low (−0.04 to −0.12), but the strategy loses money, so its marginal value is negative.

## Reopening

Reopening needs a new pre-registered campaign on independent data, for example ES or SPY 1993–2019. Choosing a window, filter or variant from this campaign's tables is excluded. In particular, the 15:25 → 15:55 window, B4 and the high-VIX tercile were seen and must not be adopted. Full report: [final_report.md](final_report.md).
