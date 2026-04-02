# Volume Climax Pullback V2 - Final Report

## Verdict
- Final verdict: `exploitable_sous_conditions`.
- Reason: A few V2 variants improve the V1 benchmark with positive OOS behavior across more than one asset.
- Total tested rows (including V1 refs): `468`.
- V2 live variants: `162`.
- V2 dead variants removed from OOS rankings: `302`.
- True V2 improvements vs V1: `52`.

## V1 vs V2
- Best V2 variant: `dynamic_exit_atr_target_1p0_ts3_vq0p95_bf0p5_ra1p2` on `MNQ`.
- Family / entry / exit: `dynamic_exit` / `next_open` / `atr_fraction`.
- OOS Sharpe `1.298` vs V1 `-1.085`.
- OOS net PnL `2744.00` vs V1 `-1290.50`.
- OOS expectancy delta vs V1: `39.58`.
- Stability IS/OOS ratio: `1.106`.

## Research Answers
1. Regime filters improve Sharpe: `yes`.
2. Delayed entries improve expectancy: `no`.
3. Standalone defendable strategy: `yes, under conditions`.
4. Most robust asset: `MGC`.
5. IS/OOS stability improved: `yes`.

## Artifacts
- `summary_variants.csv`
- `ranking_oos.csv`
- `ranking_oos_by_asset.csv`
- `comparison_vs_v1.csv`
- `breakdown_by_asset.csv`
- `family_research_summary.csv`