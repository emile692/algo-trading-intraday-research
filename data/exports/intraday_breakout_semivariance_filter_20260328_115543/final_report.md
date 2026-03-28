# Intraday Breakout Semivariance Filter Campaign

## Methodology

- Objective: test ex-ante realized semivariance as a meta-signal overlay on top of the existing audited breakout baselines, without changing alpha logic or execution assumptions.
- Universe: `MNQ`, `MES`, `MGC`, `M2K`.
- Data: latest processed `1m` parquet per asset from the repo.
- Fixed horizons: trailing `30m`, `60m`, `90m` semivariance inside the current continuous futures session.
- Session horizon: RTH `09:30` to signal timestamp only.
- Percentile calibration: rolling prior-trade history only, lookback `252` baseline-qualified trades, minimum history `60` trades.
- Percentile inputs: `rs_plus`, `rs_minus`, `rv`, and `abs(rs_imbalance)`.
- Directional adverse mapping: long -> `rs_minus`, short -> `rs_plus`.
- Equal-weight portfolio: four standalone `50k` sleeves aggregated only on the common four-asset overlap window (`1139` sessions, `797` IS / `342` OOS).

## Audited Baselines Used

- `MNQ`: source `D:\Business\Trading\VSCODE\algo-trading-intraday-research\data\exports\mnq_orb_vix_vvix_validation_20260327_run\run_metadata.json`, rule `majority_50`, OR `30m`, direction `both`, grid `[25, 26, 27, 28, 29, 30] x [25, 26, 27, 28, 29, 30] x [90, 91, 92, 93, 94, 95]`.
- `MES`: source `D:\Business\Trading\VSCODE\algo-trading-intraday-research\notebooks\orb_MES_final_ensemble_validation.ipynb`, rule `majority_50`, OR `15m`, direction `long`, grid `[25, 26, 27, 28, 29, 30] x [25, 26, 27, 28, 29, 30] x [90, 91, 92, 93, 94, 95]`.
- `MGC`: source `D:\Business\Trading\VSCODE\algo-trading-intraday-research\notebooks\orb_MGC_final_ensemble_validation.ipynb`, rule `majority_50`, OR `30m`, direction `long`, grid `[20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30] x [25, 26, 27, 28, 29, 30] x [95, 96, 97, 98, 99, 100]`.
- `M2K`: source `D:\Business\Trading\VSCODE\algo-trading-intraday-research\notebooks\orb_M2K_final_ensemble_validation.ipynb`, rule `majority_50`, OR `15m`, direction `long`, grid `[25, 26, 27, 28, 29, 30] x [25, 26, 27, 28, 29, 30] x [90, 91, 92, 93, 94, 95]`.

## Variants Tested

- `adverse_hard_skip`: skip when directional adverse semivariance percentile is above threshold.
- `chop_hard_skip`: skip when `rv`, `rs_plus`, and `rs_minus` are all jointly elevated.
- `adverse_downsize`: keep the trade but reduce size on adverse regimes.
- `three_state_modulation`: favorable `1.0x`, neutral reduced, hostile `0.0x` using semivariance only.
- `combined_overlay`: chop skip plus adverse downsizing.

## Portfolio OOS Snapshot

- Baseline OOS Sharpe: `0.844` | MaxDD: `-13649.75` | Trades: `773`.
- Best tested variant: `three_state_modulation__60m__t95__m25` | family `three_state_modulation` | horizon `60m`.
- Best variant OOS Sharpe: `0.986` | Sharpe delta vs baseline: `0.143`.
- Best variant OOS MaxDD: `-15958.25` | DD improvement vs baseline: `-16.91%`.
- Best variant OOS trade retention vs baseline: `91.07%`.
- Assets improved on best variant: `1` / 4.

## Per-Asset OOS Results For Best Portfolio Variant

- `M2K`: OOS Sharpe `-0.211` (delta `+0.251`), MaxDD `-9675.50` (improvement `+31.11%`), trade retention `96.89%`.
- `MES`: OOS Sharpe `-0.109` (delta `-0.027`), MaxDD `-13530.00` (improvement `+2.32%`), trade retention `94.23%`.
- `MGC`: OOS Sharpe `0.522` (delta `-0.181`), MaxDD `-11894.00` (improvement `-30.57%`), trade retention `93.49%`.
- `MNQ`: OOS Sharpe `1.867` (delta `-0.038`), MaxDD `-8434.00` (improvement `+13.63%`), trade retention `87.61%`.

## Final Verdict

- Credible overlay for the breakout portfolio: `no`.
- Best family: `three_state_modulation` on horizon `60m`.
- M2K helped more than the other assets: `yes`.

## Notes

- `adverse_downsize` does not retest the fully redundant `0.0x` setting because that is already covered by the hard-skip family.
- All overlays operate only on sessions already selected by the audited baseline ensemble for each asset.
