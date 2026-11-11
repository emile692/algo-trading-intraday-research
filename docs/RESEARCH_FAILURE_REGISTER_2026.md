# Registre des recherches non promues — 2026

Date de consolidation : **4 octobre 2026**.

Ce registre répond à une règle simple : un résultat négatif doit rester aussi facile à retrouver qu'un signal retenu. Il est volontairement autonome : les conclusions, chiffres décisifs et règles de réouverture restent lisibles même lorsqu'un rapport généré a été nettoyé ou n'a pas pu être récupéré. « Non promu » couvre trois cas distincts qu'il ne faut pas confondre : hypothèse rejetée, edge trop faible/non actionnable, ou étude arrêtée faute de données causales.

## Légende de décision

| Statut | Sens | Réouverture autorisée seulement si… |
|---|---|---|
| `REJECTED` | test exécuté, gates économiques/statistiques manqués | nouvelle donnée réellement indépendante ou hypothèse nouvelle pré-enregistrée |
| `WEAK_EDGE` / `NOT_ACTIONABLE` | phénomène partiel, insuffisant pour une règle tradable | meilleure microstructure mesurée ou validation indépendante sans retuning |
| `STOPPED_DATA_GATE` | hypothèse non testée proprement | donnée point-in-time causale devenue disponible |
| `FORWARD_TEST_ONLY` | candidat intéressant mais pas promu dans les 9 sleeves | observation prospective conforme au protocole gelé |

## Registre synthétique

| Campagne | Décision | Cause principale | Preuve décisive | Action |
|---|---|---|---|---|
| Close-of-day momentum | `CLOSED_COST_CONSTRAINED` | effet brut trop petit face aux coûts | 8/8 instruments négatifs nets ; 6E Sharpe −1,11, ZN −5,97 | fermer ; revoir seulement sous coûts observés inférieurs au break-even |
| Conditional shock / overnight transfer | `REJECTED` | absence d'edge ou instabilité | 0/6 représentants standalone ; 2/6 seulement résistent au stress coût | ne pas chercher de variante de sauvetage |
| Cross-asset lead/lag | `REJECTED_NO_LEAD_LAG_EDGE` | instabilité et coûts/délai | 0/8 paire×sens passe tous les gates | aucun backtest portefeuille/Topstep |
| Dealer gamma NQ/MNQ | `STOPPED_DATA_GATE` | pas de GEX point-in-time causal | aucune colonne/fichier GEX et `available_at` historique absent | reprendre uniquement avec source causale documentée |
| Dealer gamma MES, régime GEX (2026-11-11) | `REJECT_MECHANISM` | GEX prédit la forme de la journée, pas sa direction | pente LOW−HIGH −0,08 (HAC t −0,63) ; règle canonique −4,84 USD/trade IS ; vol LOW +17 % (t 3,1) | pas de variante ; usage sizing seulement via nouvelle hypothèse pré-enregistrée ([fiche](../research/mes_gex_regime_phase1_20261111/HYPOTHESIS_CLOSURE.md)) |
| EIA natural-gas storage | `WEAK_EDGE` | edge concentré dans la première moitié | continuation positive mais HAC faible ; post-short faible | ne pas promouvoir ; collecter consensus/fills causaux |
| Fast-alpha execution overlay | `REJECTED_MICRO_ALPHA` | décroissance et fragilité au tick | MNQ disparaît en seconde moitié ; +1 tick rend le mean edge négatif | arrêt avant replay des sleeves |
| Intraday periodicity | `REJECTED` | signal brut déjà négatif et coûts destructeurs | N=20 brut −4 247 USD ; +1 tick −172 688 USD ; 0/5 contributeur positif | fermer sans sélection de créneau |
| Options-implied state conditioning | `STATE_DEPENDENCE_VALIDATED_NOT_ACTIONABLE` | dépendance de queue, pas overlay robuste | overlay commun échoue au moins un gate économique | ne pas promouvoir en filtre/sizing |
| Rebalancing 60/40 | `NO_PHASE2` | mécanisme central instable | pente month-end HAC −0,31 ; règle MES négative dans 2/3 sous-périodes | pas de Phase 2 |
| Treasury auction pressure | `REJECTED` | signe récent inverse, post-leg absent, concentration | 5Y→ZF 2024+ contredit les deux signes ; IC bootstrap croise zéro | aucun backtest après gate |
| TSI mean reversion | `REJECTED` | expectancy et stabilité négatives | MNQ −6,59 USD/trade, PF 0,847 ; 3/3 sous-périodes négatives | pas de validation aval/Topstep |
| Weekly-open reversion | `REJECTED` | touch-rate non tradable, pas de zone robuste | aucune zone 2×2 passe net, stabilité, significativité et contrôles | gate Topstep fermé |
| BTC Monday trend | `REJECTED` | lundi insuffisamment distinct / coûts | gate primaire non franchi | arrêt sans retuning |
| MBT momentum-confirmed breakout | `NOT_FROZEN_NOT_INCLUDED` | région vue, validation indépendante absente | région candidate ADX 15–20 / lookback 20–40 seulement | exclu des 9 sleeves |
| Momentum intraday cross-asset, open→open+h puis tenue jusqu'au close (2026-11-11) | `REJECTED` | effet brut positif mais plus petit que les coûts ; non répliqué sur NQ 2010–2019 | cellule IS retenue h=90 min, k=1σ : +0,008 R/jour, HAC t 0,96 ; holdout PRE t 0,29 ; indices actions t −0,16 ; seul MBT positif (t 2,29, 96 trades) | fermé ; OOS non lu ; MBT seul = nouvelle campagne pré-enregistrée, à juger en marginal vs les 3 sleeves MBT |
| Momentum glissant cross-asset : signe de `close.diff(n)` et croisement EMA pur, 5–60 min (2026-11-11) | `REJECTED` | brut légèrement positif, toujours sous les coûts ; plus ça trade, pire c'est | 32/32 cellules négatives nettes ; meilleure (EMA 5/20, 30 min) −0,030 R/jour, t −1,79 ; permutation p 0,54 ; 0/5 années ; PRE t −1,84 | fermé ; OOS non lu ; seul MNQ positif sur ~la moitié des cellules (post-hoc, non répliqué sur NQ 2010–2019) |
| Prime de session nuit cross-asset, long 18:00 → 08:00 New York (2026-11-11) | `REJECTED` | prime brute inférieure aux coûts ; sur indices actions c'est du bêta, pas un effet de nuit | 12/12 cellules négatives nettes ; meilleure (18:00 → 08:00, chaque nuit) −0,017 R/nuit, t −0,85 ; permutation p 0,19 ; 0/4 clusters ; PRE t 0,11 ; coût ≈ 0,054 R par trade | fermé ; OOS non lu ; pas de jambe short or/pétrole sans nouvelle campagne pré-enregistrée |
| Relative value intraday entre indices actions, spreads MNQ/MES/M2K (2026-11-11) | `REJECTED` | pas de retour à la moyenne ; continuation brute du même ordre que le coût de deux jambes | 30/30 cellules négatives nettes ; meilleure (continuation, h=30 min, k=1,5σ) −0,001 R/jour, t −0,16 ; permutation p 0,14 ; 1/3 paires ; coût 0,056–0,088 R par trade | fermé ; OOS non lu ; version minis = nouvelle campagne, données ES/RTY absentes |
| MCL, cinq stratégies génériques pétrole de l'article Mudrex : repli SMA 50/200, VWAP + RSI, cassure + retest, range RSI/Bollinger, tendance post-EIA (2026-11-11) | `REJECTED` | aucune règle ne passe le seuil IS ; brut inférieur aux coûts ; non répliqué sur CL 2015–2021 | 5/5 échouent (meilleur t IS 1,47, EIA : +1 392 USD sur 107 trades) ; CL 2015–2021 : 5/5 négatives nettes, EIA −610 USD (t −0,76), VWAP t −5,51, cassure t −5,05 | fermé ; OOS non lu ; idée EIA seulement via nouvelle campagne avec surprise consensus/réel ([fiche](../research/mcl_mudrex_strategies_20261111/HYPOTHESIS_CLOSURE.md)) |
| MCL, plan intraday EMA 20/50/200 + RSI 60/40 + MACD + Supertrend (2026-11-11) | `REJECTED` | indiscernable de zéro ; texte source tronqué, règle reconstruite | IS +198 USD sur 137 trades, t 0,19 ; CL 2015–2021 −548 USD, t −0,60 | fermé ; OOS non lu ; la suite du plan, si fournie, se teste une fois comme nouvelle spec gelée ([fiche](../research/mcl_ema_rsi_macd_supertrend_20261111/HYPOTHESIS_CLOSURE.md)) |
| MCL, deux stratégies trend following Unger Academy : croisement EMA multi-jours, Donchian intraday (2026-11-11) | `REJECTED` | positives en IS mais non significatives ; non répliquées sur CL 2015–2021 ; paramètres et filtres absents de la source | EMA 50/200 : IS +4 001 USD, t 0,77, CL −428 USD ; Donchian 20 : IS +3 604 USD, t 1,11, CL −6 516 USD (t −2,40) | fermé ; OOS non lu ; pas de filtre ajouté après coup ; l'EMA multi-jours est de toute façon incompatible Topstep ([fiche](../research/mcl_unger_trend_following_20261111/HYPOTHESIS_CLOSURE.md)) |
| MCL, paysage de paramètres : signe EMA intraday (09:00–14:30 et 18:00 → 16:00) et Donchian intraday (2026-11-11) | `REJECTED` | zone positive limitée à MCL 2021–2023 ; aucune cellule positive à la fois en IS et sur CL 2015–2021 | 113 cellules : CL 0/113 > 0 ; IS 52/113 ; OOS 2/113 (max +0,01) ; corrélation de rang IS↔CL 0,02 / 0,13 / −0,40 | fermé ; **OOS 2024-01 → 2026-03 lu sur toute la grille** ; seule l'extension 2026-03 → 2026-11 reste vierge ([fiche](../research/mcl_trend_landscape_20261111/HYPOTHESIS_CLOSURE.md)) |
| Market intraday momentum (Gao et al. 2018) : signe clôture précédente → 10:00, position 15:30 → 16:00 ET, MNQ/MES/MGC/MCL (2026-11-11) | `REJECTED` | anomalie non répliquée : brut nul, pas seulement inférieur aux coûts ; pente portée par 2020 seule | 0/7 portes sur 4/4 instruments ; MNQ IS brut −0,08 bp, net −1,59 bp (t −1,15), permutation p 0,50 ; pente IS t 1,40, hors 2020 t −0,90 ; NQ 2010–2019 t 0,66 ; OOS MNQ −4 099 USD ; Sharpe du book 1,75 → 1,50 | fermé ; OOS de A lu (sans paramètre), OOS de B/C non lu ; réouverture seulement sur ES/SPY 1993–2019 ([fiche](../research/market_intraday_momentum_gao_r2_20261111/HYPOTHESIS_CLOSURE.md)) |
| Mean reversion intraday conditionnelle autour du VWAP (R1) : écart au VWAP normalisé + filtre de tendance, épuisement de volatilité, réversion à l'heure du déjeuner ; MNQ, MGC, MCL (2026-11-11) | `REJECTED` | pas de réversion exploitable : continuation sur MNQ, réversion brute < coût sur MGC, négatif avant coûts sur MCL | 0/18 cellules centrales passent G1 ; 11/126 cellules IS positives (meilleur t 1,07) ; holdouts NQ 2010–2019 et CL 2015–2021 : 12/12 centrales négatives, t −2,5 à −6,9, négatives aussi en brut | fermé ; OOS non lu ; MES non lancé ; MNQ H2-D1 (épuisement + volume, +910 USD, t 0,67) vu après coup, à ne pas reprendre ([fiche](../research/conditional_vwap_mean_reversion_r1_20261111/HYPOTHESIS_CLOSURE.md)) |

## Fiches d'échec

### 1. Close-of-day intraday momentum

Le mécanisme publié — direction du close précédent vers T−30, puis position sur les trente dernières minutes — se retrouve statistiquement sur 6E et ZN, mais pas économiquement. Les huit contrats testés perdent après coûts. Le coût brut break-even est `7,10 USD` pour 6E contre `15 USD` configurés et `5,47 USD` pour ZN contre `33,75 USD`. Le bon enseignement n'est pas « l'effet n'existe pas », mais « son amplitude n'est pas capturable avec ce modèle de coûts ». Le rapport généré d'origine n'est plus disponible ; cette fiche en conserve la décision.

### 2. Conditional shock recovery et transfert overnight/RTH

Les trois familles ont été testées sur MNQ et MGC : shock recovery (`REJECTED_NO_EDGE`), acceptance continuation (`REJECTED_UNSTABLE`) et rejection fade (`REJECTED_NO_EDGE`). Les échantillons shock/fade sont parfois minuscules en OOS ; la continuation change de signe et échoue coût/délai. Aucun survivant n'autorise une analyse marginale portefeuille. Le rapport généré d'origine n'est plus disponible ; cette fiche en conserve la décision.

### 3. Lead/lag cross-asset

MES→MNQ, M2K→MNQ, 6E→MGC et ZN→MGC, dans les deux directions, ont produit `109 236` événements de grille mais zéro règle. Certaines lignes passent la monotonie, jamais simultanément le bootstrap, la stabilité et l'économie avec délai. L'arrêt avant construction de trades évite de transformer un diagnostic faible en stratégie opportuniste. Le rapport généré d'origine n'est plus disponible ; cette fiche en conserve la décision.

### 4. Dealer gamma

Ce n'est pas un rejet du mécanisme. Le dépôt ne contient que prix/volume futures et variables de volatilité ; ni GEX ni timestamp historique de disponibilité. Construire un proxy gamma à partir de NQ aurait violé le mandat causal. Les sorties `NOT_RUN` sont donc le résultat correct. Reprendre seulement avec `reference_session`, `gex_value`, unités, méthode, source et `available_at<=decision_time`. Le rapport généré d'origine n'est plus disponible ; cette fiche conserve le data gate.

### 5. EIA natural gas

Sur 289 releases officielles complètes, la continuation agrégée reste positive après un tick mais son avantage est concentré dans la première moitié. Le short post-annonce schedule-only est positif dans les deux moitiés mais faible en HAC/Sharpe. Sans historique de consensus point-in-time, aucune surprise causale ne peut être calculée. Verdict conservé : phénomène digne de suivi, pas sleeve. Le rapport généré d'origine n'est plus disponible.

### 6. Fast alpha d'exécution

L'effet reversal 5 minutes MNQ est significatif sur l'ensemble mais tombe à zéro en deuxième moitié ; MGC est faible. Une perturbation adverse d'un tick à chaque extrémité fait passer l'edge MNQ de `+0,311 bp` à `−0,051 bp` et MGC de `+0,131 bp` à `−0,982 bp`. Cette sensibilité invalide un overlay live. Le rapport généré d'origine n'est plus disponible.

### 7. Périodicité intraday

Le portefeuille cross-section canonique est négatif avant coûts (`−4 247 USD`, Sharpe `−0,136`) puis s'effondre avec un tick par côté (`−172 688 USD`, Sharpe `−5,49`). Aucun des cinq instruments ne contribue positivement au net agrégé et aucun bloc chronologique N=20 n'a un Sharpe net positif. Le rapport généré d'origine n'est plus disponible.

### 8. Conditioning options-implied

VIX/term structure apporte une information statistique additionnelle sur la queue gauche de `volatility_breakout`, mais pas un changement fiable de PnL moyen. L'unique overlay ex-ante commun autorisé n'est pas robuste économiquement ; plusieurs variables sont des transformations redondantes de la même information. L'étude valide une dépendance d'état, pas une règle d'allocation. Le rapport généré d'origine n'est plus disponible.

### 9. Rééquilibrage 60/40

La pente month-end attendue est négative mais insignifiante (`HAC t=−0,313`). La règle MES month-end perd sur 2019–2021 et 2022–2023, puis devient légèrement positive en 2024–2026. Cette inversion et le faible nombre d'événements ne justifient pas la Phase 2 malgré une corrélation faible aux sleeves actives. Le rapport généré d'origine n'est plus disponible.

### 10. Treasury auction pressure

Le pattern historique pré-auction sur ZF existe par endroits, mais la jambe post-auction est absente, les années récentes inversent le signe et les événements les plus forts concentrent le résultat. En 2024+, le pré-leg 5Y→ZF devient `+0,35 bp` et le post-leg `−0,41 bp`, contraires à l'hypothèse. Le protocole interdit à juste titre le backtest après échec du gate. Source : [rapport](../research/treasury_auction_pressure/README.md).

### 11. TSI mean reversion

La règle MNQ longue TSI(25/13), signal EMA7, seuil −25, stop 1,5 ATR et horizon 12 barres produit 1 455 trades à `−6,59 USD` d'espérance, PF `0,847`. Les trois sous-périodes sont négatives et la monotonie des seuils manque. Tous les étages aval restent `SKIPPED_BY_PRIMARY_GATE`. Le rapport généré d'origine n'est plus disponible.

### 12. Weekly open reversion

Le retour visuel vers le weekly open est réel, mais le simple touch-rate masque l'excursion adverse. La confirmation bornée sur 783 semaines uniques ne trouve aucune zone 2×2 contiguë passant simultanément net, stabilité, significativité et deux contrôles. Aucune simulation Topstep ne doit être lancée. Le [screening](../research/weekly_open_reversion_20261001/REPORT.md) est conservé ; le rapport de confirmation généré ne l'est plus.

### 13. Bitcoin Monday trend

Le lundi ne se distingue pas assez des jours de contrôle et l'avantage ne survit pas aux coûts sur le protocole gelé. La période 2018–2025 est une réplication historique, non un OOS post-publication. Décision : arrêt sans recherche de sous-horaire gagnant. Le rapport généré d'origine n'est plus disponible.

### 14. MBT momentum-confirmed breakout

La région ADX 15–20 / lookback 20–40 était candidate dans les heatmaps MBT, mais n'a pas été gelée faute de validation indépendante. Elle ne fait pas partie des trois ensembles MBT ni du portefeuille actif. Le [rapport de freeze](../research/mbt_three_sleeve_freeze_20261004/REPORT.md) conserve cette décision ; son manifeste généré reste à régénérer.

### 15. Momentum intraday cross-asset (open → open+h, tenue jusqu'au close)

Règle commune sur MNQ, MES, M2K, MGC et MBT : suivre le mouvement open→open+h s'il dépasse k fois sa taille habituelle, sortie au close, sans stop. Grille de 15 cellules identique partout (h 30–180 min, k 0–1σ), sélection sur IS 2019-05 → 2023-12 par Sharpe lissé sur le voisinage, série poolée équilibrée par cluster (indices actions / or / bitcoin). La cellule retenue (h=90 min, k=1σ) donne +0,008 R par jour (HAC t 0,96) et devient négative à coûts doublés. Le holdout antérieur (NQ 2010-06 → 2019-05, MGC 2016-02 → 2019-05) ne réplique pas : t 0,29. La permutation max-stat (p 0,046) et le PBO (0,17) passent, mais ils valident le classement de la grille, pas le niveau. L'effet brut est positif dans les 15 cellules et inférieur aux coûts dans 11 ; il est nul sur les indices actions et l'or, et porté par MBT (t 2,29 sur 96 trades, observation post-hoc). OOS 2024-01 → 2026-03 non lu. 6E, ZN et MCL non testés : données 1 minute absentes du disque. Détail : [fiche de clôture](../research/intraday_momentum_cross_asset_phase1_20261111/HYPOTHESIS_CLOSURE.md).

### 16. Momentum glissant cross-asset (signe de diff, croisement EMA pur)

Deux règles sans filtre, tenues et retournées barre par barre pendant la séance RTH, flat la nuit : signe de `close.diff(n)` sur barres de 5, 15, 30 ou 60 minutes (n = 1 à 16 barres) et signe EMA rapide − EMA lente (5/20, 10/40, 20/80). Exécution à l'open de la barre suivante, 1 tick + commission par côté. Mêmes 32 cellules sur MNQ, MES, M2K, MGC et MBT, poolées par cluster, sélection sur IS 2019-05 → 2023-12. Les 32 cellules sont négatives après coûts ; la moins mauvaise (EMA 5/20 sur 30 minutes) perd 0,030 R par jour (t −1,79), aucune année n'est positive, la permutation donne p 0,54 et le holdout NQ/MGC antérieur est négatif (t −1,84). Le brut est faiblement positif (≈ +0,02 à +0,06 R/jour), du même ordre que la campagne open→open+h : le momentum existe mais est plus petit que le coût de le suivre. Seul MNQ est positif sur environ la moitié des cellules, observation post-hoc proche de la sleeve `atr_trend_flip` et non répliquée sur NQ 2010–2019. OOS non lu. Détail : [fiche de clôture](../research/rolling_momentum_cross_asset_phase1_20261111/HYPOTHESIS_CLOSURE.md).

### 17. Prime de session nuit cross-asset (long Globex 18:00 → 08:00)

Position longue pendant la nuit Globex, sortie à 08:00 New York avant toutes les sleeves RTH. Six fenêtres entre 18:00, 02:00, 03:00 et 08:00, tradées chaque nuit ou seulement après une séance RTH baissière : 12 cellules identiques sur MNQ, MES, M2K, MGC, MCL (depuis 2021-07) et MBT, poolées par cluster, sélection sur IS 2019-05 → 2023-12. Les 12 cellules sont négatives après coûts. La moins mauvaise (18:00 → 08:00, chaque nuit) perd 0,017 R par nuit (t −0,85), avec 2 années positives sur 5, aucun cluster positif et une permutation à p 0,19. Le coût d'un aller-retour vaut environ 0,054 R, plus que la prime brute de la meilleure fenêtre (+0,037 R) : la nuit est peu volatile, donc tenir longtemps ne dilue pas le coût. L'heure de l'ouverture européenne (02:00 → 03:00) est nulle en brut (+0,004 R). L'or n'a pas de prime de nuit (négatif sur IS et sur 2015–2019) et le pétrole est significativement négatif sur CL 2015–2021 (t −2,75). Seuls les indices actions ont une prime brute, inférieure à celle d'une position longue RTH sans coûts sur IS comme sur NQ 2010–2019 : c'est du bêta. NQ 2010–2019 est positif isolément (t 2,12), observation post-hoc non prolongée sur MNQ 2019–2023 (t 0,73). OOS non lu. Détail : [fiche de clôture](../research/night_session_premium_phase1_20261111/HYPOTHESIS_CLOSURE.md).

### 18. Relative value intraday entre indices actions (spreads MNQ/MES/M2K)

Position market-neutral, longue un indice et courte l'autre à notionnel égal, ouverte à open+h quand l'écart relatif depuis l'ouverture RTH dépasse k fois sa taille habituelle, fermée au close. Trois paires (MNQ−MES, MNQ−M2K, MES−M2K) poolées comme un seul pari, 30 cellules : h 30–180 min, k 0,5–1,5σ, dans les deux sens (retour à la moyenne et continuation). Sélection sur IS 2019-05 → 2023-12 ; pas de holdout antérieur faute de données ES/RTY avant 2019. Les 30 cellules sont négatives après coûts. Le retour à la moyenne est négatif avant coûts dans 14 cellules sur 15. La continuation est positive en brut à h=30 et 60 minutes seulement (+0,013 à +0,028 R par jour), nulle au-delà ; la cellule retenue (continuation, h=30 min, k=1,5σ) gagne 5 à 11 bp bruts par trade pour 4 à 7 bp de coût sur deux jambes, soit −0,001 R net par jour (t −0,16), permutation p 0,14, une paire positive sur trois (MNQ−M2K, post-hoc). Même schéma que les momentum mono-instrument : un petit effet de continuation que le coût efface ; retirer le facteur marché ne l'agrandit pas et la seconde jambe double le coût. OOS non lu. Détail : [fiche de clôture](../research/index_relative_value_phase1_20261111/HYPOTHESIS_CLOSURE.md).

### 19. MCL — cinq stratégies génériques de l'article Mudrex

Transposition intraday des cinq stratégies de l'article Mudrex sur le pétrole, une seule version figée par stratégie, signaux sur barres 15 minutes, séance 09:00–14:30 New York, 1 contrat, 4,50 USD par aller-retour : repli sur la SMA 50 en régime SMA 50/200 (S1), cassure tenue du VWAP avec RSI > 50 puis entrée au repli (S2), cassure du plus haut/bas de la veille sur volume puis entrée au retest (S3), retour depuis les bandes de Bollinger avec RSI extrême hors tendance (S4), entrée dans le sens de la tendance 15 minutes après le rapport EIA (S5). Seuil IS fixé avant le run : au moins 100 trades, net > 0, t journalier ≥ 2. Sur MCL 2021-07 → 2023-12, aucune ne passe : S1 +334 USD (t 0,16), S2 −49 USD (t −0,05), S3 +526 USD (t 0,56), S4 −1 074 USD (t −1,20), S5 +1 392 USD sur 107 trades (t 1,47, Sharpe 0,94). Sur CL 2015 → 2021-07 simulé aux coûts MCL, les cinq sont négatives : S1 −4 138 USD (t −2,25), S2 −4 773 USD (t −5,51), S3 −3 487 USD (t −5,05), S4 −1 635 USD (t −1,97), S5 −610 USD (t −0,76). S1 a un brut positif sur les deux périodes (+3 443 et +3 625 USD), entièrement absorbé par les coûts. Le résultat IS de S5 est le meilleur de cinq essais sur 2,4 ans et son signe s'inverse sur CL. OOS 2024-01 → 2026-03 et extension 2026 non lus. Données : `MCL.v.0` et `CL.v.0` 1 minute, ajustées aux rolls, 26 jours dégradés exclus. Détail : [fiche de clôture](../research/mcl_mudrex_strategies_20261111/HYPOTHESIS_CLOSURE.md).

### 20. MCL — plan intraday EMA / RSI / MACD / Supertrend

Plan discrétionnaire écrit pour le MCX indien, dont le texte source s'arrête avant les règles d'entrée et de sortie ; la règle testée est une reconstruction à partir de sa liste d'indicateurs. Tendance 15 minutes par alignement clôture > EMA 20 > EMA 50 > EMA 200 ; entrée sur barre 5 minutes avec RSI(14) > 60 (ou < 40), croisement MACD(12,26,9) dans le sens du trade, volume supérieur à sa moyenne 20 barres et Supertrend(10,3) du bon côté ; stop suiveur sur le Supertrend, sortie à 14:30 New York, pas d'objectif. Mêmes données, coûts et seuil IS que la fiche 19. MCL IS : +198 USD sur 137 trades, PF 1,05, t 0,19. CL 2015 → 2021-07 : −548 USD sur 363 trades, t −0,60. Le brut est légèrement positif sur les deux périodes (3 à 6 USD par trade) et ne couvre pas 4,50 USD de coûts ; quatre conditions simultanées réduisent le nombre de trades sans améliorer le trade moyen. OOS non lu. Détail : [fiche de clôture](../research/mcl_ema_rsi_macd_supertrend_20261111/HYPOTHESIS_CLOSURE.md).

### 21. MCL — deux stratégies trend following Unger Academy

L'article ne donne ni longueurs d'EMA, ni longueur de Donchian, ni stop, ni coûts, et ne définit pas ses filtres ; ses résultats publiés (333 900 et environ 170 000 USD sur CL depuis 2010) ne sont donc pas reproductibles. Le test porte sur le cœur non filtré de chaque idée avec des paramètres conventionnels figés avant le run. S7 : croisement EMA 50/200 sur barres 15 minutes, toujours en position, tenue sur plusieurs séances, PnL valorisé chaque jour, un aller-retour facturé à chaque roll. S8 : cassure du canal de Donchian 20 barres de 15 minutes par ordre stop, première cassure du jour entre 09:00 et 14:00 New York, stop suiveur sur le bord opposé, sortie à 14:30. Mêmes données, coûts et seuil IS que la fiche 19. MCL IS : S7 +4 001 USD sur 309 trades (t 0,77), S8 +3 604 USD sur 593 trades (t 1,11). CL 2015 → 2021-07 : S7 −428 USD sur 883 trades (t −0,09), S8 −6 516 USD sur 1 591 trades (t −2,40, brut +644 USD). L'IS contient 2022, année de forte tendance sur le pétrole. S8 trade presque chaque séance (593 trades sur 609 jours, dont 101 exécutés dès la barre de 09:00) : c'est un pari quotidien sur le sens de la nuit plus qu'une cassure sélective. S7 tient des positions la nuit et le week-end, ce que Topstep n'autorise pas. OOS non lu. Détail : [fiche de clôture](../research/mcl_unger_trend_following_20261111/HYPOTHESIS_CLOSURE.md).

### 22. MCL — paysage de paramètres, EMA intraday et Donchian intraday

Exploration demandée après la clôture de la campagne Unger, sans seuil ni pré-enregistrement : Sharpe annualisé du PnL journalier net (1 contrat, 4,50 USD par aller-retour) sur CL 2015-01 → 2021-07, MCL IS 2021-07 → 2023-12 et MCL OOS 2024-01 → 2026-03. **L'OOS a été lu sur toute la grille** ; il n'est plus un verdict vierge pour ces deux familles, seule l'extension 2026-03-20 → 2026-11-10 n'a pas été calculée. S7 adapté à l'intraday : position = signe (EMA rapide − EMA lente) sur barres 15 minutes, tenue soit 09:00–14:30 New York, soit 18:00 → 16:00 (plat avant la coupure Topstep), 34 cellules chacune (rapide 5 à 80, lente 50 à 400). S8 : cassure Donchian, 45 cellules (canal 4 à 90 barres, première entrée de 09:00 à 10:30). Cellules positives : CL 0/34, 0/34 et 0/45 ; IS 7/34, 15/34 et 30/45 ; OOS 0/34, 0/34 et 2/45 (maximum +0,01). Meilleur Sharpe IS +0,28, +0,71 et +0,77 ; médianes OOS −0,92, −1,17 et −0,66. Aucune cellule n'est positive à la fois en IS et sur CL ; la corrélation de rang des cellules entre IS et CL vaut 0,02, 0,13 et −0,40. La zone positive est donc propre à 2021–2023, qui contient 2022. Forcer S7 à plat chaque jour ajoute un aller-retour quotidien : EMA 50/200 passe de +4 001 USD (multi-jours, 309 trades) à +861 USD (18:00 → 16:00, 890 trades) en IS, puis −9 351 USD sur CL et −6 613 USD en OOS. Le brut est proche de zéro hors IS. Sur le seuil : t ≥ 2 sur 2,4 ans exige un Sharpe d'environ 1,3 ; sur CL + MCL avant 2024 (9 ans) il suffit de 0,67, mais aucune cellule n'y est même positive. Le blocage vient du signe, pas du seuil. Détail et heatmaps : [fiche de clôture](../research/mcl_trend_landscape_20261111/HYPOTHESIS_CLOSURE.md).

### 23. Market intraday momentum (Gao, Han, Li, Zhou, JFE 2018) sur futures

Réplication stricte, sans paramètre, gelée avant calcul : signe du rendement entre la clôture précédente (barre 15:59 ET) et 10:00 ET, position d'un contrat de l'open de 15:30 au close de 15:59, 1 tick + commission par côté. MNQ, MES, MGC, MCL testés séparément ; benchmarks NQ 2010–2019 et CL 2015–2021. Le brut est nul : MNQ IS −0,08 bp par séance pour 1,5 bp de coût, net −1,59 bp (t −1,15), taux de réussite brut 49,5 %, permutation p 0,50, exactement le résultat d'un signal tiré au hasard. MES −2,90 bp (t −2,58), MGC −2,20, MCL −5,23 ; OOS négatif sur les quatre (MNQ −4 099 USD). La pente de réplication IS de MNQ (b 0,056, t 1,40, R² 1,9 %, du même ordre que le papier) vient de 2020 seule (t 2,29) ; hors 2020 elle vaut −0,013 (t −0,90), et elle est nulle sur NQ 2010–2019 (t 0,66), qui recouvre la fin de l'échantillon du papier. Seules 2012 et 2020 ont une pente significative sur NQ 2010–2026. Le placebo 15:00 → 15:30 fait mieux que la vraie fenêtre. Variantes (normalisation par la vol, sans gap, gap seul, avant-dernière demi-heure, horloge native or/pétrole) : 0/18 après Holm ; le peu de pente vient du gap nocturne, pas de 09:30–10:00. Régimes : pente plus forte en VIX haut (t 1,75, sens du papier) mais net non significatif ; le volume va à l'inverse du papier ; aucun candidat conditionnel. Corrélation au book faible (−0,04 à −0,12) mais Sharpe du book 1,75 → 1,50 avec MNQ. OOS de A lu (règle sans paramètre), OOS des variantes et régimes non lu. Détail : [rapport final](../research/market_intraday_momentum_gao_r2_20261111/final_report.md).

### 24. Mean reversion intraday conditionnelle autour du VWAP (R1)

Trois hypothèses, deux définitions chacune, une cellule centrale pré-enregistrée par définition, testées séparément sur MNQ, MGC et MCL. Exécution : barres d'une minute, entrée au marché + 1 tick, stop + 1 tick, objectif en limite exigeant 1 tick au-delà, stop prioritaire, 1,25 USD par côté. H1 : écart au VWAP normalisé par les bandes σ du VWAP ou par σ_jour·√τ, avec un filtre anti-tendance (efficiency ratio), sortie au VWAP. H2 : mouvement de 30 min ≥ 2,5 σ suivi d'une contraction de volatilité ou d'une barre d'arrêt, sortie à 50 % du mouvement. H3 : 11:30–13:30, extension vers le VWAP ou le bord du range sur faible volume. Les 18 cellules centrales échouent G1 ; 11 cellules IS sur 126 sont positives (meilleur t 1,07), Holm 1,0 partout. Sur MNQ, le prix éloigné du VWAP continue plus souvent qu'il ne revient (H1 brut −0,03 à −0,05 R, MNQ H1-N1 −6 690 USD, t −2,24), filtre de tendance ou non ; la fenêtre du déjeuner ne change rien (H3-RANGE −5 230 USD, t −2,71). Sur MGC, une réversion brute existe (+0,02 à +0,07 R) mais reste inférieure à un aller-retour (≈ 0,08 R). Sur MCL, tout est négatif avant coûts. Holdouts NQ 2010–2019 (économie MNQ) et CL 2015–2021 (économie MCL) : les 12 cellules centrales sont négatives, t −2,5 à −6,9, et négatives aussi en brut. Seul effet de timing : épuisement MNQ avec contraction (H2-D1, placebo p 0,018, brut +0,15 R), net nul (+42 USD sur 299 trades), négatif à coût ×2 et sur NQ 2010–2019 (t −4,4). Troisième campagne, après `mean_reversion_cross_asset_20261006` et le fast-alpha, à trouver continuation ou neutralité plutôt que réversion intraday sur les indices. OOS non lu, MES non lancé, diversification sans objet. Détail : [rapport final](../research/conditional_vwap_mean_reversion_r1_20261111/final_report.md).

## Candidats positifs non actifs — à ne pas classer comme échecs

- Le Gotobi 6J month-end est `VALIDATED_CANDIDATE`, puis `KEEP_FORWARD_TEST_ONLY` à l'intégration. Il reste volontairement hors des neuf sleeves, sans être rejeté.
- Les quatre séries de Manuel montrent une forte orthogonalité statistique, mais le mécanisme et les trades ne sont pas assez documentés pour une promotion.
- L'étude 6→9 MBT a eu un verdict de gate global `NEUTRAL`; l'utilisateur a néanmoins promu explicitement le mapping equal-contract comme référence. Cette décision de gouvernance est enregistrée, sans réécrire le verdict statistique.

## Historique antérieur déjà consolidé

Les campagnes MES/M2K, 6E, ZN, MCL et autres pistes supprimées lors du nettoyage du 26 septembre restent documentées dans [RESEARCH_CLOSURE_2026.md](RESEARCH_CLOSURE_2026.md). Les anciens inventaires CSV comptaient 180 campagnes et 3 657 fichiers pour 2 795 114 545 octets ; leurs lignes détaillées n'ont pas été récupérées et ne sont donc pas présentées comme disponibles.

## Règle de gouvernance

Une campagne fermée ne peut pas être « sauvée » en choisissant après coup un instrument, une fenêtre ou un paramètre gagnant. Toute réouverture doit créer un nouvel identifiant de campagne, pré-enregistrer hypothèse/gates, utiliser une information nouvelle et laisser ce registre inchangé.

