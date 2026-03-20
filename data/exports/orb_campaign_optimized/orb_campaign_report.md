# ORB Research Campaign

## Repo Inspection Findings

- Global market assumptions and default capital/costs live in `src/config/settings.py`.
- Session filtering is handled in `src/data/session.py` and the existing notebook uses `extract_rth(...)`.
- The current OR range is built in `src/features/opening_range.py`.
- The current breakout-after-OR signal logic lives in `src/strategy/orb.py`.
- Stop, target, entry timing, sizing, and exit handling live in `src/engine/backtester.py`.
- Execution costs are modeled in `src/engine/execution_model.py`.
- Metrics are computed in `src/analytics/metrics.py`.
- The repo already had a sweep helper in `src/analytics/heatmaps.py`, and outputs conventionally belong under `data/exports`.
- The central notebook loads the main research dataset from `MNQ_1mim.parquet` and converts it to `America/New_York`.

## Current ORB vs Paper-Exact

- Current repo logic: 15-minute opening range on intraday futures bars, then enter only after a later breakout through OR high/low.
- Paper-exact variant here: 5-minute resampled bars, first candle defines bias, and the backtester enters at the next bar open, which is the second 5-minute candle open.
- Current repo stop logic: OR boundary stop plus optional buffer.
- Paper-exact stop logic here: first 5-minute candle high/low, which matches the 5-minute OR boundaries on the resampled dataset.
- Current repo default execution assumptions: futures-style slippage and per-side commissions.
- Paper-exact reference profile here: zero slippage and `0.0005` per unit commission in the engine. On the default MNQ futures dataset this is only an approximation of the paper's ETF/share assumptions.

## ATR Regimes

The filter campaign resolves ATR(14) regimes from the current RTH dataset using tertiles:

```text
atr_regime   atr_min    atr_max
      none       NaN        NaN
    band_1  0.392857   7.196429
    band_2  7.196429  12.000000
    band_3 12.000000 224.017857
```


## Ranking Framework

robustness_score = 35.0 * clip(avg_R, -1, 2) + 18.0 * clip((profit_factor - 1) / 1.5, -1, 2) + 12.0 * clip(expectancy / max(abs(avg_loss), 1), -1, 2) + 12.0 * clip(percent_of_days_traded / 0.18, 0, 1.25) - 30.0 * clip(max_drawdown_pct / 0.20, 0, 2) - 4.0 * longest_loss_streak - 30.0 * clip((min_trades - n_trades) / min_trades, 0, 1), with min_trades = 40.

The leaderboard applies the `passes_min_trades` guardrail so a low-activity configuration does not rank first just because it avoids drawdown.

## Axis A Top Results

```text
                                       name  robustness_score  n_trades  profit_factor  expectancy    avg_R  cumulative_pnl  max_drawdown  longest_loss_streak  percent_of_days_traded
           paper_exact_reference_costs_cap4        -99.246361      1736       1.158102   18.827107 0.123313       32683.857     -6316.296                   21                0.985804
                paper_exact_repo_costs_cap4       -108.679400      1735       1.072168    8.932565 0.054777       15498.000     -7186.500                   21                0.985236
paper_exact_reference_costs_no_leverage_cap       -117.912625      1736       1.159070   25.646325 0.123313       44522.021     -9430.741                   21                0.985804
     paper_exact_repo_costs_no_leverage_cap       -125.777664      1735       1.061869    9.969452 0.054777       17297.000    -13148.500                   21                0.985236
```


## Axis B Top Results

```text
                            name  robustness_score  n_trades  profit_factor  expectancy     avg_R  cumulative_pnl  max_drawdown  longest_loss_streak  percent_of_days_traded
       current_15m_long_only_rr5        -14.541290       576       1.169939    8.858507  0.101425          5102.5       -1440.0                    8                0.327087
     prop_baseline_long_only_rr5        -14.541290       576       1.169939    8.858507  0.101425          5102.5       -1440.0                    8                0.327087
      current_15m_long_only_rr10        -15.814425       576       1.134836    7.054688  0.084025          4063.5       -1440.0                    8                0.327087
       current_15m_long_only_rr3        -16.587097       576       1.120710    6.274306  0.069941          3614.0       -1442.0                    8                0.327087
     prop_baseline_long_only_rr3        -16.587097       576       1.120710    6.274306  0.069941          3614.0       -1442.0                    8                0.327087
prop_baseline_both_rr3_risk_0p15        -23.644202       223       1.074970    2.515695  0.022677           561.0        -781.5                    8                0.126633
 prop_baseline_both_rr3_risk_0p1        -25.796772        80       1.137101    2.956250  0.058480           236.5        -501.5                    8                0.045429
prop_baseline_both_rr5_risk_0p15        -25.906862       223       1.038434    1.331839 -0.003259           297.0        -998.0                    8                0.126633
```


## Axis C Top Results

```text
                                              name  robustness_score  n_trades  profit_factor  expectancy    avg_R  cumulative_pnl  max_drawdown  longest_loss_streak  percent_of_days_traded
      filter_current_long_only_none_ema_only_ema50        -11.355918       547       1.164433    8.444241 0.083385          4619.0       -1462.5                    7                0.310619
  filter_current_long_only_none_vwap_and_ema_ema50        -11.355918       547       1.164433    8.444241 0.083385          4619.0       -1462.5                    7                0.310619
    filter_current_long_only_band_3_ema_only_ema50        -11.511219       195       1.208769   12.502564 0.115539          2438.0        -999.0                    6                0.110733
filter_current_long_only_band_3_vwap_and_ema_ema50        -11.511219       195       1.208769   12.502564 0.115539          2438.0        -999.0                    6                0.110733
              filter_current_long_only_band_2_none        -12.623853       333       1.164539    8.630631 0.102475          2874.0       -1306.0                    7                0.189097
    filter_current_long_only_band_2_ema_only_ema20        -12.845957       324       1.157268    8.237654 0.095642          2669.0       -1135.5                    7                0.183986
         filter_current_long_only_band_2_vwap_only        -12.963233       330       1.157723    8.265152 0.099741          2727.5       -1306.0                    7                0.187394
    filter_current_long_only_band_2_ema_only_ema50        -13.695385       319       1.142696    7.529781 0.082720          2402.0       -1117.0                    7                0.181147
```


## Recommendation

filter_current_long_only_none_ema_only_ema50 ranked highest on the robustness score with 547 trades, profit factor 1.16, expectancy 8.44, max drawdown -1462.50, and longest loss streak 7.

## Caveats

- The default campaign dataset is `MNQ_1mim.parquet`. If you want a closer paper replication on QQQ/TQQQ, you still need matching instrument data.
- The paper reference cost profile is only exact when quantity represents units/shares for a compatible instrument model.
- On futures-style contract data, lower risk-per-trade settings can reduce trade count because integer position sizing may skip signals that would require fractional contracts.
- Daily Sharpe is computed from daily PnL divided by a static initial capital base, then annualized with `sqrt(252)`.
- End-of-day exits use the configured `time_exit` or the last available bar in the session when no stop/target hit occurs first.
- The current filter campaign applies VWAP and EMA checks on the signal bar close to avoid lookahead.

## Outputs

- Full results CSV: `data/exports/orb_campaign_optimized/orb_campaign_results.csv`
- Leaderboard CSV: `data/exports/orb_campaign_optimized/orb_campaign_leaderboard.csv`
- Plot directory: `data/exports/orb_campaign_optimized/plots`
