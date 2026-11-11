# Treasury auction pressure — final investigation

## Final verdict: REJECTED

The final replication adds ZF on the same volume-ranked `v.0` construction as ZN. A historical 5Y→ZF pre-auction decline is visible, but the complete canonical mechanism does not pass the decision gate: the post-auction recovery is insignificant, the recent sample reverses, the effect is not uniquely centered at the deadline, and the signed result is concentrated in the strongest events. No strategy backtest was run.

## ZF data

- 4,069,036 one-minute rows from 2013-05-17 00:00 UTC through 2026-09-18 20:59 UTC.
- Databento `ZF.v.0`: volume-ranked front month, no back-adjustment, matching ZN.
- 3,439 audited sessions; median 399 of 400 expected bars between 08:20 and 15:00 ET.
- Zero duplicate timestamps, invalid OHLC rows, missing required values, or zero-volume rows.
- 55 instrument IDs and 56 transitions; median absolute roll discontinuity 0.326%, maximum 0.954%.
- Databento marks 32 dataset dates degraded. Nine mapping rows fall on those dates; excluding them does not change the conclusions.

## Canonical deadline study

Price direction is explicit: futures should decline before supply (`pre < 0`) and recover afterward (`post > 0`). The combined V-effect is `-pre + post`.

| Mapping | Events | Pre mean, bps (95% bootstrap) | Post mean, bps (95% bootstrap) | Combined mean, bps (block-bootstrap 95%) |
|---|---:|---:|---:|---:|
| 5Y → ZF | 159 | **−1.88** [−3.17, −0.58] | +0.40 [−0.86, +1.63] | **+2.24** [+0.55, +4.21] |
| 7Y → ZF | 171 | −0.38 [−1.68, +0.83] | +0.72 [−0.24, +1.66] | +1.00 [−0.43, +2.50] |
| 7Y → ZN | 171 | −0.12 [−1.93, +1.67] | +0.63 [−0.94, +2.11] | +0.62 [−1.42, +2.83] |
| 10Y → ZN | 162 | −0.67 [−2.73, +1.29] | +1.50 [−0.61, +3.55] | +2.17 [−0.70, +5.10] |

Only the 5Y→ZF pre-leg is statistically visible. The recovery leg required for a complete V-shape is not.

## Recency and falsification

- 5Y→ZF, 2021+: pre −2.81 bps, post −0.92 bps, combined +1.82 bps; block-bootstrap interval crosses zero.
- 5Y→ZF, 2024+: pre **+0.35** bps, post **−0.41** bps, combined **−1.02** bps; both canonical signs fail.
- The 2025 and 2026 yearly pre-legs are positive (+4.18 and +2.57 bps), contrary to the hypothesis.
- Matched controls for 5Y→ZF are flat: combined −0.08 bps [−1.84, +1.61].
- The 5Y combined effect is +2.27 bps at T−60 and +2.24 bps at T=0; it is therefore not uniquely centered at the bid deadline.
- For 7Y, matched controls are stronger than auction days on both ZF and ZN, invalidating auction specificity.
- The best 10 of 157 complete 5Y events contribute 79% of signed total effect; the largest 10 account for 21% of absolute movement.
- Leave-one-year-out means remain positive for 5Y, but several block-bootstrap intervals touch or cross zero; the failure is regime/recency and mechanism completeness, not one isolated year.

## Timing limitation

FiscalData contains the competitive bid deadline but not an exact historical public-results timestamp. Treasury's historical documentation gives target release lags, while the paper uses timestamps of the first Bloomberg result story. Since no consistent exact public series was found, all final results are deadline-based and no result-based series was fabricated.

## Gate decision

The event study fails the mandatory gate because recent data is contradictory, the post-leg is absent, timing is not uniquely centered, and the edge is concentrated. Per the frozen protocol, commissions, slippage and strategy metrics are `N/R`: constructing a backtest after this failure would be an unauthorized rescue attempt.

Primary outputs: `event_level_final.csv`, `event_path_summary_final.csv`, `temporal_stability.csv`, `yearly_metrics_final.csv`, `event_time_shifts.csv`, `placebo_event_times.csv`, `matched_control_results.csv`, `leave_one_year_out.csv`, `effect_concentration.csv`, `data_quality_sensitivity.csv`, and `final_event_study_summary.json`.
