# MNQ ORB Intraday PnL Overlay Report

## Baseline

- Baseline officielle conservee: OR30 / direction `both` / RR `2.00` / one_trade_per_day `True`.
- Filtre ATR / selection ensemble conservee via `majority_50` et calibree strictement sur IS.
- Sizing de phase 1 force en `fixed_contracts=1` pour isoler l'effet overlay PnL intraday.
- Hypotheses d'execution appliquees dans cette campagne: commission side `0.62` USD, slippage `1.00` tick(s).
- Regle overlay importante: evaluation **bar-close mark-to-market** pour les hard caps / locks / giveback, alors que le stop/target baseline reste intrabar. Cette convention est leak-free mais doit etre lue comme un overlay de gestion, pas comme un changement du signal.
- Dataset: `MNQ_c_0_1m_20260321_094501.parquet` | sessions IS/OOS `1222` / `525` | sessions selectionnees par l'ensemble `1134`.

## Baseline Readout

- Baseline OOS: net pnl `9524.64` | Sharpe `1.627` | Sortino `1.468` | PF `1.356` | expectancy `28.10` | maxDD `-3410.50`.
- Baseline OOS days/trades: `339` jours trades | `339` trades | avg trades/day `0.646`.
- Baseline OOS prop daily-loss breach freq `0.0%` | cut-day freq `0.0%`.

## Main Findings

- Meilleur overlay: `combo_loss_0p5__profit_1p5` | verdict `defensive_but_costly` | OOS pnl retention `70.6%` | PF delta `-0.062` | expectancy delta `-8.26` | maxDD improvement `48.4%` | worst-day improvement `30.5%`.
- Driver principal observe: cut-day freq `44.0%`, profit-lock freq `10.9%`, giveback freq `0.0%`, daily-loss-breach freq `0.0%`.
- Candidat phase 2 le plus simple: `combo_loss_0p5__profit_1p5` (prop_defensive).
- Variants robustes: `0` | defensifs mais couteux: `6` | inactifs structurellement: `4`.

## Reponses Directes

- Est-ce qu'un overlay PnL intraday ameliore reellement la baseline ? Pas de facon assez robuste pour revendiquer un vrai upgrade de baseline.
- Les gains viennent-ils surtout du drawdown / giveback / baisse du nombre de trades ou d'une vraie amelioration du PF ? Lecture principale: surtout defensif / prop-compatible.
- Quelles regles sont les plus robustes OOS ? Meilleur overlay: `combo_loss_0p5__profit_1p5` | verdict `defensive_but_costly` | OOS pnl retention `70.6%` | PF delta `-0.062` | expectancy delta `-8.26` | maxDD improvement `48.4%` | worst-day improvement `30.5%`.
- Y a-t-il un bon candidat simple et interpretable pour une phase 2 avec sizing 3-state ? Pas encore de facon assez nette.
- L'interet est-il surtout prop defensif ou y a-t-il aussi un gain alpha net ? Plutot prop defensif.

## Structural Notes

- La baseline officielle est `one_trade_per_day=True`. Cela rend les blocs `trade cap`, `stop apres 2 pertes`, `continuer seulement si le premier trade gagne` en grande partie structurellement inactifs dans ce setup precis.
- Les overlays vraiment informatifs dans ce run sont donc surtout ceux qui coupent plus tot le trade en cours: hard loss cap, hard profit lock, giveback, state machine simple.
- Si vous souhaitez tester la pleine richesse des blocs sequence / trade-count sur une baseline multi-trades, il faudra une campagne separee ou une baseline differente.

## Exports

- `screening_summary.csv`
- `validation_summary.csv`
- `full_variant_results.csv`
- `daily_path_summary.csv`
- `intraday_state_transition_summary.csv`
- `final_report.md`
- `final_verdict.json`
- `variants/<variant>/trades.csv`
- `variants/<variant>/daily_results.csv`
- `variants/<variant>/controls.csv`
- `variants/<variant>/state_transitions.csv`
