# Intraday Breakout Semivariance Filter Campaign

## Methodology

- Objective: test ex-ante realized semivariance as a meta-signal overlay on top of the existing audited breakout baselines, without changing alpha logic or execution assumptions.
- Universe: `MNQ`, `MES`, `MGC`, `M2K`.
- Data: latest processed `1m` parquet per asset from the repo.
- Fixed horizons: trailing `30m`, `60m`, `90m` semivariance inside the current continuous futures session.
- Session horizon: RTH `09:30` to signal timestamp only.
- Percentile calibration: rolling prior-trade history only, lookback `63` baseline-qualified trades, minimum history `20` trades.
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
- Best tested variant: `adverse_hard_skip__30m__t85` | family `adverse_hard_skip` | horizon `30m`.
- Best variant OOS Sharpe: `0.915` | Sharpe delta vs baseline: `0.072`.
- Best variant OOS MaxDD: `-14895.50` | DD improvement vs baseline: `-9.13%`.
- Best variant OOS trade retention vs baseline: `86.68%`.
- Assets improved on best variant: `2` / 4.

## Per-Asset OOS Results For Best Portfolio Variant

- `M2K`: OOS Sharpe `-0.085` (delta `+0.377`), MaxDD `-8608.00` (improvement `+38.71%`), trade retention `85.60%`.
- `MES`: OOS Sharpe `-0.047` (delta `+0.035`), MaxDD `-12062.50` (improvement `+12.91%`), trade retention `82.05%`.
- `MGC`: OOS Sharpe `0.395` (delta `-0.307`), MaxDD `-11120.00` (improvement `-22.07%`), trade retention `83.85%`.
- `MNQ`: OOS Sharpe `1.954` (delta `+0.049`), MaxDD `-9765.00` (improvement `+0.00%`), trade retention `98.53%`.

## Final Verdict

- Credible overlay for the breakout portfolio: `no`.
- Best family: `adverse_hard_skip` on horizon `30m`.
- M2K helped more than the other assets: `yes`.

## Notes

- `adverse_downsize` does not retest the fully redundant `0.0x` setting because that is already covered by the hard-skip family.
- All overlays operate only on sessions already selected by the audited baseline ensemble for each asset.
