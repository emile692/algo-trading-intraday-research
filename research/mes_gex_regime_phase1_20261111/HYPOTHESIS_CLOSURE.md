# MES dealer-gamma regime — phase 1 closure

**Status:** `REJECT_MECHANISM` (direction) · `STATE_DEPENDENCE_NOT_ACTIONABLE` (volatility)
**Date:** 2026-11-11
**Spec:** [FROZEN_SPECIFICATION.json](FROZEN_SPECIFICATION.json), sha256 `8cddce24…d9702a` ([SPEC_LOCK.json](SPEC_LOCK.json)), frozen before any return was computed with GEX.
**OOS:** not computed, because the IS gate failed.

## Hypothesis

Low prior-day GEX (relative to its trailing year) means dealers are short gamma, so the day trends. High GEX means they are long gamma, so the day reverts. The prediction: the slope of the 10:30→16:00 return on the 09:30→10:30 return is higher in LOW-GEX regimes than in HIGH-GEX regimes.

## IS results, MES 2019-05 → 2023-12 (1 121 sessions)

| Test | Estimate | HAC t | Gate |
|---|---:|---:|---|
| H2 slope LOW − HIGH (GEX lagged 1 day) | −0,085 | −0,63 | **fail** (wrong sign) |
| H2 slope LOW − HIGH (GEX lagged 2 days) | +0,107 | +0,79 | **fail** (t < 1,0) |
| H2b efficiency LOW − HIGH, VIX-controlled | +0,025 | +3,30 | pass |
| Canonical rule (follow on LOW, fade on HIGH), mean net per trade | −4,84 USD | −0,92 | **fail** |
| H1 ln(RV) LOW − HIGH, controlled for VIX and lagged RV (not a gate) | +0,158 | +2,69 | — |

Rule legs:
- LOW follow: +1,31 USD net per trade over 320 trades, t 0,12.
- HIGH fade: −9,34 USD net per trade over 438 trades, t −1,94.
- Yearly: 2020 and 2021 negative, 2019, 2022 and 2023 slightly positive.

## Reading

- **GEX carries real information on the *shape* of the day, beyond VIX and lagged realised volatility:**
  - LOW-GEX days have about 17 % more realised variance (t 3,1).
  - HIGH-GEX days are choppier, with lower efficiency (t −3,0).
- **It does not predict the *direction* of the afternoon relative to the morning.** Choppy HIGH-GEX days do not reverse the first hour; fading them loses money. The trend/reversal mechanism as specified is rejected.

## What may and may not be reopened

- No rescue variant: no other window, regime cut, normalisation or lag on the same data.
- The volatility dependence could only feed **sizing or risk** (smaller size on LOW-GEX days to protect the MLL). That would be a separate pre-registered hypothesis, tested on the book's PnL, not on MES direction.
- Data caveat: the SqueezeMetrics file has no `available_at` timestamps and may be revised. A positive sizing result would need forward confirmation from a snapshot archived each day.

## Files

- [run_phase1.py](run_phase1.py): runner. It checks the spec hash and input hashes, and includes a numpy Newey-West self-test.
- `results/phase1_results.json`, `results/session_features.csv`, `results/canonical_trades_IS.csv`, `results/canonical_yearly_IS.csv`.
- The GEX source is `data/raw/squeezemetrics/DIX_retrieved_20261111.csv`. It is local only and gitignored, because no redistribution licence was granted.
