# R1 — Conditional VWAP Mean Reversion : audit des données et du repository

Date : 2026-11-11. Statut : **audit terminé, aucun backtest lancé**.

Scripts : `audit_data.py`, `audit_thresholds.py`, contrôle de roll (`audit/roll_checks.txt`). Ils calculent uniquement des comptages, des profils d'activité et des amplitudes non signées. Aucun rendement postérieur à un signal n'a été calculé.

## 1. Ce que le repository a déjà testé (a priori décisif)

| Campagne | Ce qui recoupe R1 | Résultat |
|---|---|---|
| `research/mean_reversion_cross_asset_20261006` | `vwap_extension` = \|close − VWAP\| / ATR14 sur barres 5 min, seuils 0,75–2,0, sorties VWAP / 50 % / 15, 30, 120 min | **REJECTED sur 6/6 actifs** (MNQ, MGC, MES, M2K, NQ, MBT). MNQ : OOS Sharpe −1,36 sur 4 096 trades. MGC : −2,05. Long et short tous deux négatifs en IS. |
| idem | event study après extension VWAP | **continuation**, pas réversion : MNQ −1,1 % de l'écart à 15 min, −2,0 % à 30 min, −3,4 % à 60 min. MGC −0,7 % à 30 min. |
| idem | `range_extreme` (écart au milieu d'un range glissant) | REJECTED sur 6/6 |
| idem | `consecutive_exhaustion` (série de barres + déplacement) | REJECTED, sauf MNQ en `WEAK_WATCHLIST` (OOS Sharpe 0,10) |
| idem | `gap_reversion` | seul mécanisme survivant ; déjà en shadow sur NQ/MNQ, hors du champ de R1 |
| Fast-alpha execution overlay | réversion 5 minutes MNQ / MGC | edge disparaît en seconde moitié et avec un tick adverse |
| TSI mean reversion | oscillateur contrarien MNQ | REJECTED (−6,59 USD par trade) |
| `src/strategy/mean_reversion.py` | `generate_vwap_extension_reversion_signals` et 5 autres familles | framework exploratoire, aucun résultat retenu |

**Conséquence sur les périodes.** La campagne du 2026-10-06 a coupé IS/OOS au 2023-03-17 (MGC) et au 2024-02-23 (MNQ). Sa fenêtre OOS a été lue pour la famille VWAP extension. **Pour H1, l'OOS 2024–2026 n'est donc pas vierge sur MNQ et MGC.** La règle de R1 est nouvelle (normalisation, condition de tendance), mais elle est corrélée à ce qui a été vu. Fenêtres réellement vierges pour ce type de règle :
- EXT 2026-03-20 → 2026-10-02 pour MNQ et MGC, environ 130 séances ;
- tout l'historique MCL, jamais testé en mean reversion (données absentes en octobre) ;
- le holdout antérieur NQ 2010–2019, non utilisé par cette campagne.

## 2. Données

Barres d'une minute Databento OHLCV, horodatées au début de la barre, converties en `America/New_York`. **Pas de ticks, pas de carnet, pas de bid/ask.**

| Instrument | Fichier | Contrat continu | Historique | Séances complètes IS |
|---|---|---|---|---|
| MNQ | `data/processed/parquet/MNQ_c_0_1m_20190505_20261003.parquet` | `c.0` (roll calendaire à l'expiration) | 2019-05 → 2026-10-02 | 1 139 |
| MGC | `data/processed/parquet/MGC_v_0_1m_20150101_20261003.parquet` | `v.0` (roll au volume) | 2015 → 2026-10-02 ; exploitable depuis 2019 | 1 143 |
| MCL | `data/downloaded/MCL_v_0_1m_20210712_20261111.parquet` | `v.0` | 2021-07 → 2026-11-10 | 609 |
| MES | `data/processed/parquet/MES_c_0_1m_20260322_135702.parquet` | `c.0` | 2019-05 → 2026-03-20 | 1 143 |
| NQ (holdout) | `data/processed/parquet/NQ_c_0_1m_20260321_095036.parquet` | `c.0` | 2010-06 → 2026-03-20 | 2 150 avant 2019-05 |
| CL (holdout) | `data/downloaded/CL_v_0_1m_20150101_20210712.parquet` | `v.0` | 2015 → 2021-07 | 1 627 |

**MGC avant 2019 est inutilisable** : 29, 15 et 81 séances au moins complètes à 90 % en 2016, 2017 et 2018. MGC n'a donc pas de holdout antérieur.

Dans la séance de chaque actif, il ne manque aucune minute entre 08:30 et la fin de séance. Il n'y a ni doublon ni prix nul.

## 3. Roll des contrats continus

| | Changements de contrat à l'intérieur d'une séance | Rolls | Volume des 5 séances avant le roll (rapporté aux séances −25 à −6) |
|---|---|---|---|
| MNQ | 0 | 30 | 0,86 / 0,41 / 0,24 / 0,18 / 0,12 |
| MES | 0 | 27 | 0,72 / 0,29 / 0,19 / 0,16 / 0,11 |
| MGC | 0 | 59 | 0,94 / 0,82 / 0,64 / 0,14 / 0,01 |
| MCL | 0 | 64 | 0,95 / 0,89 / 0,83 / 0,28 / 0,05 |

- Le VWAP, l'écart au VWAP et les sorties sont intra-séance : **pas de couture de roll dans les prix.**
- En revanche, les dernières séances avant chaque roll portent sur un contrat que plus personne ne traite : 12 à 41 % du volume normal sur MNQ, 1 à 14 % sur MGC. Un trader réel serait déjà sur le contrat suivant. Les filtres de volume (H2, H3) y seraient faussés, et le VWAP y serait calculé sur un flux résiduel.
- **Exclusion proposée** : les 4 séances avant chaque roll pour MNQ et MES, les 2 séances avant pour MGC et MCL. Les dates d'expiration étant publiques à l'avance, cette exclusion n'utilise pas d'information future.

## 4. Sessions et activité

Sessions du repository : MNQ et MES 09:30–16:00, MGC 08:20–13:30, MCL 09:00–14:30 ET (horloges utilisées par tout le repository).

Part médiane du volume de séance par demi-heure, IS :

| | 09:30 | 10:30 | 11:30 | 12:00 | 12:30 | 13:00 | 13:30 | 14:00 | 15:30 |
|---|---|---|---|---|---|---|---|---|---|
| MNQ | 17,1 % | 10,0 | 6,8 | 5,8 | 4,9 | 4,8 | 4,5 | 4,9 | 7,1 |
| MGC | 14,8 | 9,9 | 6,0 | 4,7 | 3,8 | 4,4 | — | — | — |
| MCL | 13,4 | 11,4 | 8,2 | 6,4 | 5,2 | 4,7 | 4,1 | 6,5 | — |

La fenêtre 11:30–13:30 est bien le creux d'activité sur les trois actifs. Pour MGC, la séance se termine à 13:30 avec le settlement 13:29–13:30 ; pour MCL, le settlement a lieu entre 14:28 et 14:30.

## 5. Échelles non signées (IS, barres 5 min)

| | MNQ | MGC | MCL | MES |
|---|---|---|---|---|
| Coût aller-retour (1 tick + commission par côté) | 3,50 USD | 4,50 | 4,50 | 5,00 |
| Coût en bp | 1,4 | 2,5 | 5,6 | 2,5 |
| ATR14 5 min en bp | 15,4 | 10,8 | 29,8 | 11,2 |
| Écart médian \|close − VWAP\| / ATR5 | 1,77 | 1,38 | 1,50 | 1,64 |
| Écart en bp quand \|close − VWAP\| ≥ 2 ATR5 | 50 | 32 | 92 | 37 |
| Coût / écart dans ce cas | 0,03 | 0,08 | 0,06 | 0,07 |

Deux enseignements :

1. **La normalisation ATR5 de la campagne précédente est mal calibrée.** La moitié des barres sont déjà à plus de 1,8 ATR du VWAP, parce que le VWAP s'éloigne naturellement du prix au fil de la séance alors que l'ATR5 mesure le bruit d'une barre. Un seuil de 2 ATR se déclenche sur environ 30 barres par séance : ce n'est pas un événement extrême. C'est ce qui explique les 4 000 trades et plus de cette campagne.
2. **Les coûts ne sont pas le problème** (3 à 8 % de l'écart à capturer), contrairement aux campagnes momentum. La question est le signe : le prix revient-il vers le VWAP plus souvent qu'il ne s'en éloigne ?

## 6. Fréquence des normalisations candidates (IS, croisements par séance)

- N1 = (close − VWAP) / σ_VWAP, où σ_VWAP est l'écart-type pondéré par le volume du prix typique autour du VWAP depuis l'ouverture (bandes VWAP classiques).
- N2 = (close/VWAP − 1) / (σ_jour · √τ), où σ_jour est l'écart-type des rendements ouverture → clôture des 20 séances précédentes et τ la fraction de séance écoulée.

Un croisement est une barre qui dépasse le seuil alors que la barre précédente ne le dépassait pas. On compte les croisements entre l'ouverture + 30 min et la fin − 30 min.

| Seuil | MNQ | MGC | MCL | dont 11:30–13:30, MNQ |
|---|---|---|---|---|
| N1 ≥ 2,0 | 2,71 | 1,88 | 2,47 | 0,78 |
| N1 ≥ 2,5 | 0,94 | 0,42 | 0,69 | 0,28 |
| N1 ≥ 3,0 | 0,19 | 0,05 | 0,10 | 0,06 |
| N2 ≥ 0,75 | 2,57 | 2,00 | 2,46 | 0,61 |
| N2 ≥ 1,0 | 1,69 | 1,15 | 1,45 | 0,38 |
| N2 ≥ 1,5 | 0,61 | 0,41 | 0,43 | 0,12 |

L'efficiency ratio de séance (|close − ouverture| / chemin parcouru) vaut en médiane 0,13–0,14 sur toutes les barres et 0,17–0,36 au moment des croisements. Les grands écarts surviennent plus souvent les jours de tendance, ce qui justifie la condition de tendance de H1.

## 7. Book de référence

- Les 6 sleeves citées (MNQ ORB VWAP/ATR, Volatility Breakout, Compression Breakout, ATR Trend Flip, MGC Volatility Breakout, Gap Continuation) existent en PnL quotidien :
  - sur OOS, 2024-01-02 → 2026-03-19, dans `research/locked_nine_sleeve_portfolio_20261101/daily_pnl.csv` ;
  - sur le forward, 2026-03-20 → 2026-10-02, dans `research/frozen_vote_forward_replay_20261101/forward_daily_six_new.csv`.
- **Il n'existe pas de PnL du book sur IS.** La diversification ne peut se mesurer qu'après lecture de l'OOS.
- **Écart avec le book actif** : le lock actif `eight_sleeve_equal_contract_reference_20261103` met ATR Trend Flip en SHADOW et contient 3 sleeves MBT absentes de ta liste.

## 8. Framework réutilisable

| Composant | Où | Usage proposé |
|---|---|---|
| Specs instruments (tick, valeur, commission 1,25, slippage 1 tick) | `src/config/settings.py` | repris tel quel ; MCL ajouté localement avec la même convention |
| Modèle d'exécution (1 tick adverse par côté + commission) | `src/engine/execution_model.py` | repris |
| VWAP de session | `src/features/intraday.py::add_session_vwap` | repris (et contrôlé contre une implémentation indépendante) |
| Sizing fixe / risque | `src/risk/position_sizing.py::resolve_position_size` | repris pour le sizing en USD |
| Gabarit de campagne gelée (spec hashée, self-test, porte IS, OOS sur accord) | campagnes du 2026-11-11 | repris |
| Backtester mean reversion barre à barre | `src/engine/mean_reversion_backtester.py` | non repris : il exécute au close de la barre de signal et sur 5 min ; il faut des sorties au niveau de la minute pour les stops, les MAE/MFE et le remplissage des objectifs |

## 9. Limites d'exécution

- **Données en barres d'une minute** : un objectif ou un stop touchés dans la même minute ne peuvent pas être ordonnés. Convention conservatrice : le stop est réputé touché en premier.
- **Pas de file d'attente** : un objectif en ordre limite (VWAP) n'est réputé exécuté que si la minute traite au moins 1 tick au-delà du niveau.
- **Stops** : exécutés au pire du niveau de stop et de l'open de la minute, plus 1 tick.
- **Pas de calendrier macro historique** : pas de filtre d'annonces. Sur MGC, les publications de 08:30 tombent dans la première demi-heure, qui est exclue des entrées.
- **Pas de GVZ ni d'OVX** : le régime de volatilité de MGC et MCL se mesure par la volatilité réalisée, celui des indices par le VIX.
