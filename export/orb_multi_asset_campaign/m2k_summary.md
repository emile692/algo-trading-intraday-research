# M2K Ensemble ORB Summary

- Dataset: `M2K_c_0_1m_20260322_134808.parquet`
- Tradable sessions analysed: 1747
- IS/OOS split: 1222 / 525 sessions
- Instrument specs loaded from config: tick_size=0.1, tick_value=0.5, point_value=5.0, commission_per_side=1.25, slippage_ticks=1

## Baseline Transfer

- Baseline transfer rule: `majority_50`
- OOS score=-4.093, PF=0.909, Sharpe=-0.462, Return/DD=-0.581

## Final Recommendations

- Best OOS ensemble: `unanimity_100` (score=-3.186, PF=0.967, Sharpe=-0.155, MaxDD=-9003.00).
- Most robust ensemble: `unanimity_100` (robustness=-3.571).
- Baseline-like ensemble retained if solid: `majority_50`.
- Best point cell: `ATR 26 / q30/q90` (score=-2.998).
- Most robust point cell: `ATR 26 / q30/q90` (local robustness=-3.076).

## Robust Clusters

- No robust cluster detected.

## Campaign Readout

- Transferability verdict: transfer weak / requires caution.
- Baseline MNQ-style majority rule gap vs best ensemble: 0.907 score points.
- Sessions with incomplete opening window: 410.
- Sessions missing part of RTH bars: 593.

## Exports

- Full results: `m2k_results_full.csv`
- Top configs: `m2k_top_configs.csv`
- Aggregation summary: `m2k_aggregation_summary.csv`
- Charts folder: `charts/`
