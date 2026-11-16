# Macro quadrant regime from free nowcasts — feasibility check

**Date:** 2026-11-11
**Status:** feasibility only. No price, return or P&L series was read. OOS (2024-01 → 2026-03) not touched.
**Question:** can the QuantCube quadrant (sign of the 5-day change of a growth and an inflation nowcast → 4 regimes) be rebuilt from free, point-in-time sources, and would a regime built that way be persistent and well-identified enough to test as a daily conditioner of the intraday sleeves?

## Sources

| Leg | Source | Point-in-time | Update rate in IS |
|---|---|---|---|
| Growth | Atlanta Fed GDPNow, `TrackingArchives` (one target quarter per date) | yes: real-time archive since 2014-05 | value changes on 59 % of business days, max gap 12 days |
| Inflation A | Cleveland Fed Inflation Nowcasting, headline CPI YoY, latest live target month | archive since 2013-08; whether early values are real-time or recomputed is **not confirmed** | value changes on 29 % of business days, median gap 6 days (steps on gasoline and data releases) |
| Inflation B | 10y breakeven = Treasury par nominal 10y − real 10y (home.treasury.gov) | yes (market close) | daily |

- FRED (`T10YIE`) was unreachable from this network, so the breakeven is rebuilt from the Treasury curves.
- Raw files are in `data/downloaded/macro_nowcasts_20261111/` (≈ 30 MB, not committed). Their sha256 values are in `results/feasibility.json`.
- Nowcast series are chained: only revisions within the same target count, and a target roll adds 0.
- China GDP is not available for free, so the growth leg is US-only.
- The regime code is a literal port of the notebook's `ComputeRegimes` with lag 4, toAdj 2. It is causal because toAdj < lag.

## Results (notebook window = 5, plus 10 and 20 for persistence only)

| Inflation source | Window | Period | Switches / year | Median episode (b-days) | Episodes ≤ 5 days |
|---|---|---|---|---|---|
| Cleveland CPI | 5 | IS | 35 | 6 | 49 % |
| Breakeven | 5 | IS | 33 | 6 | 50 % |
| Cleveland CPI | 20 | IS | 16 | 15 | 21 % |
| Breakeven | 20 | IS | 20 | 10 | 24 % |

PRE (2014-07 → 2019-04) is similar. The four quadrants each take 17–33 % of days. Full table: `results/regime_persistence.csv`.

**For comparison, the notebook regime is slower.** Its turnover of 3 885 %/year, with an average Σ|Δw| of 3.24 per switch in its allocation table, implies about 12 switches per year, or about 24 if turnover is counted one-way. With free sources at the notebook's window, the regime switches 2–3 times more often. The GDPNow leg alone flips sign 226 times in IS, about every 5 days.

**The inflation leg is not identified.** The Cleveland CPI nowcast and the breakeven agree on the direction of the change only 54–56 % of the time (windows 5/10/20, IS and PRE), barely above a coin flip. The resulting 4-quadrant regimes agree on 51 % of IS days. Which proxy is used decides the regime.

**Power.** With roughly 300 IS days per quadrant, testing one quadrant against the rest at t = 2 only detects a difference of about 0.13 daily standard deviations of the conditioned P&L. That is an annualised Sharpe gap of about 2. A 2-state growth-only split (about 600 days each) needs about 0.115 sd, or a Sharpe gap of about 1.8.

## Reading

1. **The data is feasible.** Both nowcasts are free, cover PRE and IS, and GDPNow is clearly point-in-time.
2. **The 4-quadrant regime is not worth a phase 1:**
   - The inflation leg depends on the proxy: two reasonable choices disagree half the time.
   - At the notebook's parameters the "regime" is a weekly-changing state, not a macro regime.
   - Only very large conditional effects would be detectable.
3. **The only defensible version is a reduced one.** Use GDPNow alone, 2 states, with a slow window (20) fixed before any return is computed, as a sizing or direction conditioner on the equity-index sleeves. Judge it on marginal value against the book. The prior is weak: GEX regimes already turned out to predict volatility, not direction ([mes_gex_regime_phase1_20261111](../mes_gex_regime_phase1_20261111/HYPOTHESIS_CLOSURE.md)).

## Files

- [feasibility.py](feasibility.py): loaders, chained index, regime port, persistence stats. Needs `openpyxl`, which is not in the project venv.
- `results/feasibility.json`: coverage, inflation-source agreement, raw sha256.
- `results/regime_persistence.csv`, `results/regime_daily_notebook_params.csv`, `results/stdout.txt`.
