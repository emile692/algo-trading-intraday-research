# Cross-Asset Summary

## Comparison Table

       label symbol            selection    sharpe  profit_factor  expectancy  max_drawdown_abs  return_over_drawdown  nb_trades  pct_days_traded  composite_score
MNQ_baseline    MNQ baseline_majority_50  1.017558       1.226577   68.654982           6508.00              2.858866        271         0.516190         1.243853
    MES_best    MES          majority_50 -0.081291       0.985454   -5.536859          13851.25             -0.124718        312         0.594286        -2.778448
    M2K_best    M2K        unanimity_100 -0.154619       0.966861  -11.272727           9003.00             -0.289237        231         0.440000        -3.186212

## Direct Answers

1. MES transferability: no.
2. M2K transferability: no.
3. MNQ parameters transfer partially, but ticker-specific recalibration improves results.
4. Robust MES cluster: No robust cluster identified.
5. Robust M2K cluster: No robust cluster identified.
6. Most promising complement to MNQ: MES.
7. M2K looks sufficiently differentiated from MNQ.
8. MES is not just a smoother clone; it changes the payoff profile materially.

## Charts

- `charts/cross_asset_equity_compare.png`
- `charts/cross_asset_metrics_compare.png`
