# MNQ ORB Regime Filter And Dynamic Sizing Campaign

## Baseline

- Baseline reused as official reference: `MNQ` / OR30 / direction `both` / RR `2.0` / ensemble `majority_50`.
- Dataset: `MNQ_c_0_1m_20260321_094501.parquet`
- IS/OOS sessions: `1222` / `525`

## Regime Readout

- Nominal OOS: net pnl `35575.00` | Sharpe `1.905` | Sortino `1.863` | PF `1.428` | maxDD `-9765.00`.
- Best volatility splitter in IS: `realized_vol_ratio_15_60` (score spread `4.935`, min bucket obs `265`).
- Best extension splitter in IS: `gap_abs_atr20` (score spread `4.611`, min bucket obs `265`).
- Structural context readout: `weekday_name` selected, but it still needs OOS confirmation.

## Overlay Verdict

- No filter or sizing overlay tested here clearly beats the nominal ensemble once OOS retention and coverage are penalized honestly.

## Exports

- `summary_variants.csv`
- `conditional_bucket_analysis.csv`
- `feature_ranking.csv`
- `regime_state_mappings.csv`
- `selected_session_regimes.csv`
- `variants/<variant>/...`
