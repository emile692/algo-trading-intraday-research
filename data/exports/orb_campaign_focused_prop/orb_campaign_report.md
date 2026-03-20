# Focused ORB Prop Campaign

## Objective

This campaign deepens the strongest practical branch from the previous sweep:

- current repo ORB logic
- long-only only
- EMA directional filter only
- optional ATR regime filter
- prop-style constraints centered on a 50K evaluation reference

## Research Constraints

- Account size reference: `$50,000`
- Max loss limit reference: `$2,000`
- Daily loss limit reference: `1000`
- Profit target reference: `$3,000`
- Monthly subscription cost reference: `$150`
- Daily loss limit basis: `realized_daily_pnl`

These are research settings, not engine hard-codes.

## Focused Grid

- Reward ratio: `3`, `4`, `5`
- EMA length: `30`, `50`, `70`, `100`
- ATR filter mode: `none`, `moderate_band`, `restrictive_band`
- Risk per trade: `0.10%`, `0.15%`, `0.20%`, `0.25%`
- Dataset: `MNQ_1mim.parquet`

## Resolved ATR Bands

```text
      atr_regime   atr_min    atr_max
            none       NaN        NaN
   moderate_band  9.267857 224.017857
restrictive_band 12.000000 224.017857
```


## Ranking Formula

robustness_score = 18.0 * clip((profit_factor - 1) / 0.30, -1, 2) + 16.0 * clip(expectancy / max(abs(avg_loss), 1), -1, 2) + 10.0 * clip(n_trades / 120, 0, 1.25) + 8.0 * clip(percent_of_days_traded / 0.10, 0, 1.25) + 20.0 * int(target_reached_before_max_loss) + 12.0 * clip(3.0 / months_to_target, 0, 2) - 24.0 * clip(abs(max_drawdown) / 2000, 0, 2) - 10.0 * clip(longest_loss_streak / 7.0, 0, 2) - 18.0 * int(breaches_max_loss_limit) - 6.0 * clip(number_of_daily_loss_limit_breaches, 0, 2) - 8.0 * clip(subscription_drag_estimate / 3000, 0, 2) - 12.0 * clip((min_trades - n_trades) / min_trades, 0, 1).

The score is intentionally practical: it rewards configurations that reach the target without violating the max loss reference, keeps trade count meaningful, and penalizes slow or sparse variants even if their raw PnL looks attractive.

## Top Ranked Configurations

```text
                                                name  robustness_score  n_trades  profit_factor  expectancy  cumulative_pnl  max_drawdown  longest_loss_streak  days_to_profit_target  estimated_months_to_profit_target  subscription_drag_estimate  profit_target_reached_before_max_loss
 focused_long_only_rr5_ema30_moderate_band_risk_0p25         32.824848       332       1.357751   20.006024          6642.0       -1110.5                    7                  648.0                          30.857143                 4628.571429                                   True
focused_long_only_rr5_ema100_moderate_band_risk_0p25         32.500213       294       1.363678   20.477891          6020.5       -1062.0                    8                  643.0                          30.619048                 4592.857143                                   True
 focused_long_only_rr4_ema30_moderate_band_risk_0p25         30.484776       332       1.336394   18.811747          6245.5       -1182.0                    7                  648.0                          30.857143                 4628.571429                                   True
focused_long_only_rr4_ema100_moderate_band_risk_0p25         30.052899       294       1.318947   17.959184          5280.0        -998.5                    8                  648.0                          30.857143                 4628.571429                                   True
focused_long_only_rr3_ema100_moderate_band_risk_0p25         29.869329       294       1.281871   15.789116          4642.0        -934.5                    7                  638.0                          30.380952                 4557.142857                                   True
 focused_long_only_rr5_ema70_moderate_band_risk_0p25         28.171475       304       1.349784   19.723684          5996.0       -1122.5                    7                  880.0                          41.904762                 6285.714286                                   True
 focused_long_only_rr3_ema70_moderate_band_risk_0p25         27.721583       304       1.286409   16.069079          4885.0       -1122.5                    7                  648.0                          30.857143                 4628.571429                                   True
 focused_long_only_rr3_ema30_moderate_band_risk_0p25         27.219788       332       1.298999   16.643072          5525.5       -1257.5                    7                  636.0                          30.285714                 4542.857143                                   True
 focused_long_only_rr5_ema50_moderate_band_risk_0p25         27.215711       311       1.354977   20.040193          6232.5       -1232.0                    7                  880.0                          41.904762                 6285.714286                                   True
 focused_long_only_rr3_ema50_moderate_band_risk_0p25         27.151595       311       1.295653   16.609325          5165.5       -1232.0                    7                  643.0                          30.619048                 4592.857143                                   True
 focused_long_only_rr4_ema70_moderate_band_risk_0p25         26.282260       304       1.322570   18.189145          5529.5       -1122.5                    7                  880.0                          41.904762                 6285.714286                                   True
 focused_long_only_rr4_ema50_moderate_band_risk_0p25         25.833938       311       1.335070   18.916399          5883.0       -1232.0                    7                  880.0                          41.904762                 6285.714286                                   True
```


## Direct Answers

- Which EMA length is most robust? `EMA30` led on median robustness score at `-20.45` with median drawdown `-700.25` and target-before-loss hit rate `25%`.
- Does ATR filtering genuinely help, or only reduce frequency? `moderate_band` ranked best on median score. The ATR filters can improve control in some pockets, but they materially change trade frequency rather than delivering a free improvement. Compared with no ATR filter, the restrictive regime changed median trades from 240 to 42 and median drawdown from -1076 to -656.50.
- Which RR is the best practical compromise? `RR 5` had the strongest median score, with median trades `108` and median PF `1.03`.
- Which risk-per-trade level is most compatible with Topstep-style constraints? `0.25%` ranked best on the focused score, balancing target reach, drawdown, and trade count.
- Which final config is recommended? `focused_long_only_rr5_ema30_moderate_band_risk_0p25` is the strongest robust candidate for visual validation in this branch, but it is still too slow to look subscription-efficient for a typical prop evaluation.
- How quickly does it typically reach the `3,000` target, if at all? No configuration reached the `3,000` target before estimated subscription drag exceeded the target itself. The fastest target hit still took `636` trading days, or about `30.29` trading months.
- What are the main caveats? Daily loss is approximated from realized daily PnL, max-loss feasibility is path-based rather than probabilistic, and the backtester keeps static account-size sizing instead of compounding.

## Recommendation

- Name: `focused_long_only_rr5_ema30_moderate_band_risk_0p25`
- RR: `5`
- EMA length: `EMA30`
- ATR mode: `moderate_band`
- Risk per trade: `0.25%`
- Trades: `332`
- Profit factor: `1.36`
- Expectancy: `$20.01`
- Max drawdown: `$-1110.50`
- Longest loss streak: `7`
- Days to target: `648`
- Estimated months to target: `30.86`
- Target reached before max loss: `yes`

This candidate gave the best overall balance between prop-style survivability and robustness inside the tested branch. The main practical limitation is speed: the entire branch remains too slow to reach the evaluation target efficiently once subscription drag is considered.

## Caveats

- The daily loss rule is modeled as an optional research constraint on realized daily PnL. It does not attempt intraday trailing enforcement.
- Max loss feasibility is evaluated on the realized cumulative PnL path, not on a Monte Carlo distribution.
- Position sizing still uses the repo's fixed account-size reference per trade. This keeps the extension localized and comparable to prior runs.
- Restrictive ATR filters can improve some risk statistics while sharply reducing participation.

## Outputs

- Full results CSV: `data/exports/orb_campaign_focused_prop/orb_campaign_results.csv`
- Leaderboard CSV: `data/exports/orb_campaign_focused_prop/orb_campaign_leaderboard.csv`
- Markdown report: `data/exports/orb_campaign_focused_prop/orb_campaign_report.md`
- Plot directory: `data/exports/orb_campaign_focused_prop/plots`
- Validation notebook: `notebooks/orb_topstep_validation.ipynb`
