# H2 close-of-day intraday momentum — closure record

**Status:** `CLOSED_COST_CONSTRAINED`  
**Closure date:** 2026-09-29  
**Portfolio status:** not promoted; not a seventh sleeve

## What was established

The canonical, non-optimized close-of-day momentum effect is statistically present in the local 6E and ZN samples. It survives the recent-period check, signal permutations, sign inversion and a one-bar execution delay. It is concentrated in the final thirty-minute window rather than the two predefined earlier placebo windows.

The effect is nevertheless too small under the frozen execution assumptions. Mean gross profit per trade—the maximum zero-expectancy all-in round-trip cost—is:

| Instrument | Gross break-even cost | Frozen all-in cost | Net expectancy per contract/trade |
|---|---:|---:|---:|
| 6E | $7.10 | $15.00 | −$7.90 |
| ZN | $5.47 | $33.75 | −$28.28 |

Increasing the number of contracts cannot repair this deficit while costs remain linear per contract. It multiplies both negative expectancy and risk.

## Conditional reopening rule

H2 may be reconsidered only when actual execution records demonstrate that the all-in round-trip cost per contract is sustainably below:

- **$7.10 for 6E**, or
- **$5.47 for ZN**.

These are break-even ceilings, not deployment targets. Costs should be comfortably below the applicable ceiling to allow for estimation error, regime variation and adverse fills. The first reopening step is a prospective replay/shadow validation using actual broker or prop-firm fills.

Reopening does **not** authorize changes to the signal, thirty-minute window, entry/exit convention, filters, thresholds, direction or sizing. Any such modification is a separate research hypothesis. Contract scaling is considered only after positive per-contract net expectancy has been re-established.

## Evidence trail

- `CLOSE_OF_DAY_INTRADAY_MOMENTUM_REPORT.md`
- `cross_asset_summary.csv`
- `cost_sensitivity.csv`
- `permutation_results.csv`
- `placebo_and_execution_delay_results.csv`
- `frozen_config.json`
