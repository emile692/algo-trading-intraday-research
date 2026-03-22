# MNQ Ensemble ORB Summary

- Dataset: `MNQ_c_0_1m_20260321_094501.parquet`
- Tradable sessions analysed: 1747
- IS/OOS split: 1222 / 525 sessions
- Instrument specs loaded from config: tick_size=0.25, tick_value=0.5, point_value=2.0, commission_per_side=1.25, slippage_ticks=1

## Baseline Transfer

- Baseline transfer rule: `majority_50`
- OOS score=0.743, PF=1.288, Sharpe=1.281, Return/DD=1.761

## Final Recommendations

- Best OOS ensemble: `majority_50` (score=0.743, PF=1.288, Sharpe=1.281, MaxDD=-13939.00).
- Most robust ensemble: `majority_50` (robustness=-1.263).
- Baseline-like ensemble retained if solid: `majority_50`.
- Best point cell: `ATR 28 / q27/q94` (score=1.188).
- Most robust point cell: `ATR 25 / q25/q94` (local robustness=0.216).

## Robust Clusters

- Cluster 1: 53 cells, ATR 25-30, Qlow 25-30, Qhigh 93-95, avg score=0.953.

## Campaign Readout

- Transferability verdict: transfer acceptable.
- Baseline MNQ-style majority rule gap vs best ensemble: 0.000 score points.
- Sessions with incomplete opening window: 401.
- Sessions missing part of RTH bars: 477.

## Exports

- Full results: `mnq_results_full.csv`
- Top configs: `mnq_top_configs.csv`
- Aggregation summary: `mnq_aggregation_summary.csv`
- Charts folder: `charts/`
