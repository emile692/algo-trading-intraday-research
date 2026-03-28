# MNQ ORB Prop Survivability Campaign

## Baseline

- Baseline rehydratee: `MNQ` / OR30 / direction `both` / RR `2.0` / ensemble `majority_50`.
- Dataset: `MNQ_c_0_1m_20260321_094501.parquet`
- IS/OOS: `1222` / `525` sessions
- Grid gele apres calibration IS: ATR `[25, 26, 27, 28, 29, 30]` | qlow `[25, 26, 27, 28, 29, 30]` | qhigh `[90, 91, 92, 93, 94, 95]` | rule `majority_50`.

## Readout

- Nominal OOS: net pnl `35575.00` | Sharpe `1.905` | Sortino `1.863` | PF `1.428` | maxDD `-9765.00`.
- Stabilite temporelle nominale: annees positives `87.5%` | semestres positifs `73.3%` | rolling 63j positives `60.3%`.
- Aucun overlay simple ne passe le filtre utile = drawdown mieux controle sans couper trop d'exposition.
- Overlay protecteur mais couteux: `deleveraging_drawdown_3pct_5pct`.
- Overlay a eviter ou cosmetique: `cooldown_after_large_loss_1r`.

## Daily Loss Limiter

- Diagnostic honnete: avec deja `1 trade par jour`, un daily loss limiter realise ex-post ne change pas le comportement du systeme dans cette architecture.
- Seuils verifies: `500, 750, 1000` USD.
- Breach frequencies observees: `500 -> 24.0%, 750 -> 0.6%, 1000 -> 0.0%`.

## Verdict

L'edge MNQ valide tient encore sous stress raisonnable, mais aucun overlay simple teste ici ne justifie clairement une adoption systematique en plus de la baseline. Dans ce setup 1 trade/jour, la compatibilite prop vient surtout du sizing de depart et de l'acceptation du path risk, pas d'un kill switch magique.

## Exports

- `summary_variants.csv`
- `daily_loss_limiter_diagnostic.csv`
- `temporal_stability_yearly.csv`
- `temporal_stability_semester.csv`
- `temporal_stability_rolling_63d.csv`
- `variants/<variant>/...`
