# PORTFOLIO INTEGRATION — PHASE 1

## Verdict

The six frozen sleeves behave like approximately **4.48 independent risk dimensions** by correlation-eigenvalue participation ratio, with **4 graph clusters** at the predeclared 0.40 correlation threshold.
The portfolio is **not yet ready for Phase 2 allocation research** if Phase 2 assumes pristine OOS evidence: the MGC lock explicitly labels 2024–2026 as adaptive/reused. It is ready only for a prospective shadow/live-forward collection stage and for implementation checks that do not retune weights.

## Scope and provenance

- Primary common interval: **2024-01-01 → 2026-03-19**.
- Four MNQ sleeves and two MGC sleeves were reproduced from their locked configs/specifications. No sleeve parameter, exit, sizing rule, or source file was changed.
- The three allocations are static diagnostics. They use no mean-return forecast, ML, Sharpe objective, or allocation selection.
- Methodological caveat: the two MGC sleeves are frozen, but their source manifest calls this interval `adaptive_reused_not_pristine_oos`.
- Pearson and Spearman use daily net PnL. Downside correlation uses each series clipped at zero. Losing-day correlation is conditional on a negative equal-sleeve portfolio day. Trade-day overlap is Jaccard overlap; drawdown overlap uses simultaneous underwater days.

## Reproduced sleeves

| Sleeve | Instrument | Trades | PnL USD | Daily vol USD |
|---|---|---:|---:|---:|
| `mnq_orb_vwap_atr` | MNQ | 261 | 6,329.00 | 145.57 |
| `mnq_volatility_breakout` | MNQ | 291 | 11,211.00 | 257.14 |
| `mnq_compression_breakout` | MNQ | 110 | 1,078.50 | 60.40 |
| `mnq_atr_trend_flip` | MNQ | 659 | 8,274.00 | 291.15 |
| `mgc_volatility_breakout` | MGC | 65 | 2,143.00 | 122.92 |
| `mgc_gap_continuation` | MGC | 95 | 4,671.00 | 149.36 |

## Dependence and effective alpha structure

- Mean pairwise correlation among the four MNQ sleeves: **0.233**; maximum: **0.577**.
- When at least two MNQ sleeves are simultaneously active, they point in the same direction **85.7%** of those minutes.
- Maximum simultaneously active sleeves: **4/6**; maximum simultaneous MNQ sleeves: **4/4**.
- Clusters: [mgc_gap_continuation, mgc_volatility_breakout]; [mnq_atr_trend_flip, mnq_volatility_breakout]; [mnq_compression_breakout]; [mnq_orb_vwap_atr].
- The Nasdaq sleeves are not automatically four independent bets: the timestamp overlap and same-direction diagnostics must be read together with daily correlations and drawdown overlap.

Largest contributors to gross negative sleeve contributions on equal-weight portfolio losing days:

| Sleeve | Share | Signed contribution USD |
|---|---:|---:|
| `mnq_atr_trend_flip` | 38.6% | -4,928.00 |
| `mnq_volatility_breakout` | 25.7% | -3,075.75 |
| `mnq_orb_vwap_atr` | 21.6% | -2,805.92 |
| `mgc_gap_continuation` | 7.0% | -819.83 |
| `mgc_volatility_breakout` | 4.3% | -593.17 |
| `mnq_compression_breakout` | 2.9% | -185.50 |

Yearly dependence stability:

- MNQ volatility/ATR correlation ranges from **0.236** to **0.686**.
- MGC volatility/gap correlation ranges from **0.240** to **0.833**.
- The cluster identities are useful diagnostics, but their strength is not stationary year to year. Drawdown overlap is also high because several low-frequency sleeves remain underwater for long stretches; it is not evidence of synchronized entries by itself.

## Naive allocation benchmarks

| Allocation | PnL | Sharpe | Daily PF | Max DD | Worst day | Largest weight | Largest risk contribution |
|---|---:|---:|---:|---:|---:|---:|---:|
| `equal_sleeve_weight` | 5,617.75 | 1.615 | 1.453 | -903.17 | -205.75 | 16.7% | 38.5% |
| `inverse_volatility` | 3,994.27 | 1.634 | 1.469 | -503.78 | -214.65 | 36.3% | 19.7% |
| `equal_risk_contribution` | 4,006.40 | 1.675 | 1.462 | -496.59 | -214.83 | 35.3% | 16.7% |

Yearly benchmark PnL:

| Year | Equal | Inverse vol | ERC |
|---:|---:|---:|---:|
| 2024 | 1,978.50 | 1,176.52 | 1,256.42 |
| 2025 | 2,322.00 | 1,984.21 | 1,967.72 |
| 2026 | 1,317.25 | 833.54 | 782.26 |

Static diagnostic weights:

| Sleeve | Equal | Inverse vol | ERC |
|---|---:|---:|---:|
| `mnq_orb_vwap_atr` | 16.67% | 15.07% | 18.13% |
| `mnq_volatility_breakout` | 16.67% | 8.53% | 7.57% |
| `mnq_compression_breakout` | 16.67% | 36.32% | 35.25% |
| `mnq_atr_trend_flip` | 16.67% | 7.54% | 6.85% |
| `mgc_volatility_breakout` | 16.67% | 17.85% | 17.43% |
| `mgc_gap_continuation` | 16.67% | 14.69% | 14.77% |

Instrument risk contribution:

| Allocation | MNQ | MGC |
|---|---:|---:|
| `equal_sleeve_weight` | 83.6% | 16.4% |
| `inverse_volatility` | 67.6% | 32.4% |
| `equal_risk_contribution` | 66.7% | 33.3% |

Exposure and allocation concentration:

| Allocation | Effective weighted sleeves | Max weighted MNQ gross | Max weighted MGC gross | Max simultaneous sleeves |
|---|---:|---:|---:|---:|
| `equal_sleeve_weight` | 6.00 | 0.667 | 0.333 | 4 |
| `inverse_volatility` | 4.52 | 0.675 | 0.325 | 4 |
| `equal_risk_contribution` | 4.55 | 0.715 | 0.322 | 4 |

## TopstepX 50K constraints

The simulation calls the repository's existing `TopstepCampaignSpec` verbatim: USD 50,000 start, USD 3,000 target, USD 2,000 trailing MLL, 50% best-day consistency, 35- and 60-traded-day horizons, no daily loss limit, and no breakeven lock. No contract cap is present in that ruleset, so none is invented here.

| Allocation | Ruleset | Pass | MLL fail | Expiry | Target hit | Consistency delay | First empirical binder |
|---|---|---:|---:|---:|---:|---:|---|
| `equal_risk_contribution` | `topstepx_50k_extended_60d` | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | `expire` |
| `equal_risk_contribution` | `topstepx_50k_main_35d` | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | `expire` |
| `equal_sleeve_weight` | `topstepx_50k_extended_60d` | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | `expire` |
| `equal_sleeve_weight` | `topstepx_50k_main_35d` | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | `expire` |
| `inverse_volatility` | `topstepx_50k_extended_60d` | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | `expire` |
| `inverse_volatility` | `topstepx_50k_main_35d` | 0.0% | 0.0% | 100.0% | 0.0% | 0.0% | `expire` |

Synchronous-loss stress (all six sleeves realize their individual worst trade together):

| Allocation | Loss | Initial MLL breached | Repetitions to MLL | First immediate constraint |
|---|---:|---|---:|---|
| `equal_sleeve_weight` | -549.42 | False | 4 | `none_immediate` |
| `inverse_volatility` | -465.03 | False | 5 | `none_immediate` |
| `equal_risk_contribution` | -457.09 | False | 5 | `none_immediate` |

On observed rolling starts, **expiry binds first in every case**: none of the three unlevered static benchmarks reaches the USD 3,000 target inside 35 or 60 traded days, and none breaches the USD 2,000 MLL. Under repeated synchronous worst-trade stress, the first risk constraint that can bind is the trailing MLL; there is no repository Topstep daily-loss constraint to bind earlier.

## Phase-2 readiness

**Not ready for parameterized or dynamic Phase 2 allocation.** The correct next step is prospective collection with all six definitions unchanged. Static benchmark results can guide monitoring, but must not be used to choose a best-Sharpe allocation on this reused interval.

## Reproducible artifacts

- `common_timestamp_portfolio.csv`: minute positions, entries/exits, realized PnL, instrument exposure, and simultaneity flags.
- `daily_sleeve_pnl.csv`, `daily_trade_counts.csv`, `sleeve_trades/`: reproduced source series.
- Correlation, downside, loss-day, overlap, drawdown, rolling, yearly, and loss-contribution CSVs.
- Allocation weights, metrics, yearly results, sleeve/instrument risk contributions, concentration, Topstep and stress CSVs.
- `run_metadata.json` records hashes and implementation checks.
