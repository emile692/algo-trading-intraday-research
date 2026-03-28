# Local Robustness Summary

- Overall local-read verdict: `moderetement stable`
- Daily-stop vs consecutive-loss heatmap uses `max_losses_per_day` as the closest live proxy for the requested daily consecutive-loss kill-switch.
- Daily-control overlays are conservative path-preserving filters applied on the corrected trade path.

```text
              verdict  stable_neighbor_share  reference_sharpe_rank_pct                                                                                     comment                                       pair_name
    stable localement               0.666667                   0.333333                   La case de reference reste entouree d'un plateau OOS globalement positif.                    slope_threshold_x_atr_buffer
instable / pic etroit               0.222222                   0.111111 La reference n'est soutenue que par peu de voisins OOS credibles. L'optimum apparait isole.        pullback_length_x_confirmation_threshold
    stable localement               1.000000                   1.000000                   La case de reference reste entouree d'un plateau OOS globalement positif.                    open_window_end_x_max_trades
  moderetement stable               0.500000                   0.111111                       Le voisinage n'est pas vide, mais la robustesse locale reste inegale. daily_stop_threshold_x_consecutive_losses_proxy
```
