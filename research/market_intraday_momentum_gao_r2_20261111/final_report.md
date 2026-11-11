# R2 — Market Intraday Momentum sur futures : rapport final

Date : 2026-11-11. Référence : Gao, Han, Li, Zhou (2018), *Market Intraday Momentum*, JFE 129, 394–414.

## Verdict : **FAIL**

L'anomalie n'est pas répliquée sur les futures testés. Ce n'est pas un problème de coûts : **le rendement brut de la stratégie est nul**. Sur MNQ en IS, il vaut −0,08 bp par séance, avant un coût de 1,5 bp.

- **Pente IS** : la relation première demi-heure → dernière demi-heure n'est significative sur aucun instrument (meilleur t 1,40 sur MNQ).
- **Concentration** : cette pente vient entièrement de 2020 (t 2,29). Hors 2020, elle est nulle ou négative.
- **Holdout** : la pente ne se retrouve pas sur NQ 2010–2019 (t 0,66), période qui recouvre la fin de l'échantillon du papier.
- **Portes** : aucun instrument ne franchit une seule des sept portes fixées à l'avance.
- **Diversification** : sans objet. Ajoutée au book, la stratégie baisse son Sharpe sur les quatre instruments.

Protocole : spécification gelée avant tout calcul de rendement ([FROZEN_SPECIFICATION.json](FROZEN_SPECIFICATION.json), sha256 `396c6536…70f098`, [SPEC_LOCK.json](SPEC_LOCK.json)). Self-test synthétique passé avant le gel. La campagne A a été lue sur PRE, IS, OOS et EXT, comme prévu (aucun paramètre). L'OOS des campagnes B et C **n'a pas été lu** : leur règle de lecture n'était pas remplie.

## 1. Campagne A — réplication stricte

Règle :
- signal = signe du rendement clôture précédente (15:59) → 10:00 ET ;
- entrée à l'open de la barre de 15:30, sortie au close de la barre de 15:59 ;
- 1 contrat, 1 tick + commission par côté.

Unités : bp du notionnel par séance ; une séance sans position compte pour 0.

### Stratégie

| Instrument | Période | Séances | Brut (bp) | Coût (bp) | Net (bp) | t NW net | Net USD | Sharpe | Rdt ann. | Max DD USD | Réussite | PF | IC 95 % net (bp) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MNQ | IS | 1 123 | −0,08 | 1,51 | −1,59 | −1,15 | −6 668 | −0,58 | −4,0 % | −9 678 | 45,9 % | 0,82 | [−4,4 ; +1,9] |
| MNQ | OOS | 533 | −1,04 | 0,83 | −1,87 | −1,71 | −4 099 | −1,07 | −4,7 % | −4 749 | 45,8 % | 0,82 | [−3,9 ; +0,2] |
| MNQ | EXT | 130 | +4,10 | 0,61 | +3,49 | 1,72 | +2 665 | 2,24 | +8,8 % | −646 | 55,4 % | 1,53 | [−0,2 ; +7,6] |
| MES | IS | 1 123 | −0,23 | 2,67 | −2,90 | −2,58 | −6 828 | −1,17 | −7,3 % | −7 571 | 42,1 % | 0,74 | [−5,1 ; −0,4] |
| MES | OOS | 533 | −0,32 | 1,70 | −2,02 | −1,93 | −3 221 | −1,37 | −5,1 % | −3 526 | 44,7 % | 0,77 | [−4,1 ; +0,1] |
| MGC | IS | 1 108 | +0,34 | 2,54 | −2,20 | −6,34 | −4 211 | −2,96 | −5,5 % | −4 432 | 38,4 % | 0,57 | [−2,9 ; −1,5] |
| MGC | OOS | 529 | +0,17 | 1,53 | −1,36 | −2,65 | −1 885 | −1,76 | −3,4 % | −2 954 | 40,3 % | 0,77 | [−2,4 ; −0,2] |
| MGC | EXT | 131 | +3,56 | 1,02 | +2,54 | 1,85 | +1 463 | 2,21 | +6,4 % | −426 | 49,6 % | 1,50 | [0,0 ; +5,1] |
| MCL | IS | 567 | +0,25 | 5,48 | −5,23 | −6,37 | −2 440 | −4,01 | −13,2 % | −2 431 | 36,0 % | 0,52 | [−6,8 ; −3,6] |
| MCL | OOS | 506 | −1,01 | 6,50 | −7,51 | −8,40 | −2 667 | −5,86 | −18,9 % | −2 786 | 26,5 % | 0,31 | [−9,2 ; −5,9] |
| MCL | EXT | 153 | −0,97 | 5,03 | −6,00 | −1,83 | −831 | −2,17 | −15,1 % | −802 | 47,1 % | 0,62 | [−12,7 ; +0,3] |
| NQ (benchmark) | PRE 2010–19 | 2 147 | +0,58 | 1,75 | −1,17 | −1,95 | −14 980 | −0,71 | −2,9 % | −20 040 | 47,5 % | 0,90 | [−2,3 ; 0,0] |
| NQ | IS | 1 124 | +0,05 | 0,54 | −0,50 | −0,36 | −38 920 | −0,18 | −1,2 % | −75 105 | 48,5 % | 0,89 | [−3,2 ; +2,9] |
| NQ | OOS | 533 | −1,12 | 0,30 | −1,42 | −1,29 | −30 685 | −0,81 | −3,6 % | −37 808 | 46,2 % | 0,86 | [−3,6 ; +0,7] |
| CL (économie MCL) | PRE 2015–21 | 1 547 | +0,79 | 9,14 | −8,35 | −11,33 | −6 692 | −4,49 | −21,0 % | −6 757 | 27,1 % | 0,31 | [−9,8 ; −6,7] |

Le taux de réussite brut est de 49,5 % sur MNQ et de 47,1 % sur MES en IS. Le signal ne fait pas mieux qu'une pièce.

### Régression de réplication

Régression : `r_dernière = a + b · r_première`, erreurs Newey-West. C'est le chiffre comparable au papier.

| Instrument | Période | b | t NW | R² |
|---|---|---|---|---|
| MNQ | IS | 0,056 | 1,40 | 1,86 % |
| MNQ | OOS | −0,023 | −1,32 | 0,57 % |
| MES | IS | 0,040 | 1,04 | 0,89 % |
| MES | OOS | −0,024 | −1,11 | 0,47 % |
| MGC | IS | 0,007 | 1,28 | 0,25 % |
| MGC | OOS | 0,013 | 1,81 | 1,19 % |
| MCL | IS | 0,004 | 0,65 | 0,11 % |
| MCL | OOS | −0,020 | −1,56 | 3,06 % |
| NQ | PRE 2010–19 | 0,011 | 0,66 | 0,08 % |
| CL | PRE 2015–21 | 0,025 | 1,90 | 3,70 % |

Le R² IS de MNQ (1,9 %) est du même ordre que celui du papier, mais il est porté par une seule année ([results/A_diag_slope_by_year.txt](results/A_diag_slope_by_year.txt), diagnostic hors protocole) :

| | b | t NW | R² |
|---|---|---|---|
| MNQ IS, toutes années | 0,056 | 1,40 | 1,86 % |
| MNQ IS sans 2020 | −0,013 | −0,90 | 0,17 % |
| MNQ 2020 seule | 0,168 | 2,29 | 10,2 % |

Sur NQ 2010–2026, seules deux années ont une pente significativement positive : 2012 (t 3,0) et 2020 (t 2,3). 2021 est significativement négative (t −2,0). C'est cohérent avec ce que dit le papier (effet plus fort en crise), mais ce n'est pas une anomalie exploitable année après année.

### Le signal apporte-t-il quelque chose ? (IS)

| | MNQ | MES | MGC | MCL |
|---|---|---|---|---|
| Stratégie, net (bp) | −1,59 | −2,90 | −2,20 | −5,23 |
| Exposition comparable, même nombre de longs et de shorts tirés au hasard (bp) | −1,59 | −2,77 | −2,54 | −5,53 |
| p de permutation, signal permuté sur toute la période | 0,50 | 0,54 | 0,17 | 0,37 |
| p de permutation, à l'intérieur de chaque année | 0,50 | 0,53 | 0,18 | 0,39 |
| Toujours long (bp) | −2,05 | −3,36 | −2,68 | −6,24 |
| Toujours short (bp) | −0,98 | −2,00 | −2,43 | −4,75 |

Sur MNQ, la stratégie fait exactement ce que ferait un signal tiré au hasard.

Placebos, brut IS de MNQ :

| Placebo | Brut IS MNQ (bp) | t |
|---|---|---|
| Signal de la veille | −0,52 | — |
| Même signal sur 14:30 → 15:00 | +1,02 | — |
| Même signal sur 15:00 → 15:30 | +1,57 | 1,7 |
| Fenêtre réelle 15:30 → 16:00 | −0,08 | — |

La fenêtre du papier ne se distingue pas des fenêtres placebo ; elle est même la plus faible.

### Long et short, années, volatilité

- **Long et short** : les deux côtés sont négatifs en net sur IS et sur OOS pour MNQ, MES, MGC et MCL. Aucune asymétrie à exploiter.
- **Années** : MNQ n'a qu'une année IS positive sur cinq (2020, +1 736 USD), MES aucune, MGC et MCL aucune. Sur MNQ, la moyenne glissante sur 252 séances est positive dans 19 % des fenêtres.
- **Terciles de VIX (MNQ IS)** :

  | Tercile de VIX | Net (bp) | t |
  |---|---|---|
  | Bas | −2,69 | −2,50 |
  | Normal | −2,37 | — |
  | Haut | +0,82 | 0,20 |

  En OOS, les trois terciles sont négatifs.

### Sensibilités d'exécution (net, bp ; non sélectionnables)

| Cas | MNQ IS | MNQ OOS | MES IS | NQ PRE |
|---|---|---|---|---|
| Référence | −1,59 | −1,87 | −2,90 | −1,17 |
| Coût ×2 | −3,10 | −2,70 | −5,56 | −2,92 |
| Entrée à l'open de 15:31 | −1,27 | −1,67 | −2,74 | −1,13 |
| Entrée au close de la barre 15:30 | −1,26 | −1,67 | −2,75 | −1,13 |
| Sortie à l'open de 16:00 | −1,61 | −1,88 | −2,93 | −1,20 |
| Fenêtre −5 min (15:25 → 15:55) | −0,12 | −0,96 | −1,52 | −1,59 |
| Fenêtre +5 min (15:35 → 16:05) | −2,16 | −1,78 | −2,91 | −1,46 |
| Signal à 09:55 | −1,46 | −1,24 | −2,13 | −1,44 |
| Signal à 10:05 | −1,95 | −1,72 | −2,72 | −1,23 |

Aucune perturbation ne rend la stratégie positive, sauf NQ IS sur la fenêtre −5 min (+0,95 bp, non significatif, non retenu).

### Portes (fixées avant le test)

| | S1 t ≥ 2 + Holm | S2 permutation | S3 > exposition | S4 coût ×2, délai | S5 années | S6 PRE | S7 OOS |
|---|---|---|---|---|---|---|---|
| MNQ | non (t −1,15) | non (0,50) | non (0,50) | non | non (1/5) | non (NQ t 0,66) | non |
| MES | non (t −2,58) | non | non | non | non (0/5) | non | non |
| MGC | non (t −6,34) | non | non | non | non | sans objet | non |
| MCL | non (t −6,37) | non | non | non | non | non (CL t 1,90) | non |

## 2. Campagne B — variantes (PRE et IS seulement)

Net IS en bp, t NW entre parenthèses. Holm sur 18 tests : toutes les p ajustées valent 1,0.

| Variante | MNQ | MES | MGC | MCL | NQ PRE |
|---|---|---|---|---|---|
| Référence | −1,59 (−1,15) | −2,90 (−2,58) | −2,20 (−6,34) | −5,23 (−6,37) | −1,17 (−1,95) |
| B1 normalisée par la vol, \|z\| ≥ 0,5 | −0,26 (−0,22) | −1,69 (−1,70) | −0,89 (−3,29) | −3,25 (−5,15) | −0,60 (−1,23) |
| B2 sans gap (09:30 → 10:00) | −2,21 (−1,88) | −2,96 (−2,42) | −2,67 (−7,11) | −6,62 (−8,54) | −1,44 (−2,74) |
| B3 gap seul (clôture → 09:30) | −0,99 (−0,75) | −3,18 (−2,70) | −1,94 (−5,59) | −6,04 (−7,94) | −1,21 (−1,99) |
| B4 avant-dernière demi-heure (15:00 → 15:30) | +0,39 (0,30) | −1,04 (−0,97) | −2,58 (−6,82) | −5,33 (−6,13) | −1,60 (−2,92) |
| B5 horloge native | — | — | −2,18 (−5,79) | −8,08 (−3,94) | CL −6,60 (−2,86) |

Lecture :
- Le peu de pente qui existe vient du gap nocturne (B3), pas de la première demi-heure de séance (B2 : pente nulle).
- B4, le second prédicteur du papier, a la meilleure pente IS (MNQ t 1,88). Elle n'est pas confirmée sur NQ PRE (t 1,34), elle est négative sur CL (t −5,4), et son net reste nul.
- Sensibilités secondaires (signal à 09:45, 10:15 ou 10:30 ; entrée à 15:00, 15:15 ou 15:45) : aucune n'a un brut significatif ([B_secondary_sensitivities.csv](results/B_secondary_sensitivities.csv)).

## 3. Campagne C — régimes (PRE et IS seulement)

Aucun candidat conditionnel. Holm sur 36 cellules IS : toutes les p ajustées valent 1,0.

| Régime | MNQ IS | Lecture |
|---|---|---|
| VIX haut | pente 0,105 (t 1,75) ; net +0,82 bp (t 0,20) | conforme au sens du papier, différence haut − bas de pente t 1,99, mais pas significatif en net et absent en OOS (lu en A) |
| Amplitude de la première demi-heure, haute | pente 0,092 (t 1,40) | idem, plus faible |
| Volume de la première demi-heure, haut | pente −0,003 (t −0,10) | **opposé au papier** : c'est le volume *bas* qui a la pente la plus forte (MES t 2,17) |

Sur CL PRE, la pente est significative en régime de forte volatilité ou de fort volume (t 2,6 à 3,6). Mais le coût de MCL (9 bp) est trois à quatre fois plus élevé que le brut dans ces cellules. Le calendrier macro n'est pas disponible : non testé.

## 4. Diversification contre le book gelé (OOS 2024-01-02 → 2026-03-19, 1 contrat)

| | MNQ | MES | MGC | MCL |
|---|---|---|---|---|
| Corrélation avec le book | −0,04 | −0,12 | +0,04 | −0,02 |
| Corrélation les jours négatifs du book | −0,18 | −0,17 | −0,09 | −0,05 |
| PnL de la stratégie (USD) | −4 099 | −3 221 | −1 819 | −2 663 |
| Contribution au PnL total | −14,9 % | −11,3 % | −6,1 % | −9,2 % |
| Sharpe du book : 1,75 → | 1,50 | 1,58 | 1,64 | 1,60 |
| Max DD du book : −5 426 → | −5 457 | −5 011 | −5 760 | −5 751 |
| Part de variance | 3,6 % | 0,1 % | 1,1 % | 0,0 % |
| PnL les jours négatifs du book | −1 928 | −1 259 | −1 258 | −1 242 |
| Valeur marginale positive | non | non | non | non |

La corrélation est faible : contrairement à la crainte du brief, le signal ne double pas les sleeves MNQ (corrélations par sleeve entre −0,07 et +0,06). Mais une stratégie décorrélée et perdante reste perdante. Dans le leave-one-out à 9 composantes, retirer R2 donne le meilleur Sharpe. PnL de MNQ pendant les cinq pires drawdowns du book : −59, −165, +247, −806, −730 USD.

## 5. Limites

- **Pas de S&P avant 2019** (ni ES ni SPY sur disque). Le benchmark académique est NQ, pas SPY ; la période de recouvrement avec le papier se limite à 2010-06 → 2013-12.
- **Pas d'enchère de clôture** : la sortie est le dernier prix négocié avant 16:00, pas l'enchère de clôture de l'ETF.
- **Gap négociable** : un future cote la nuit. Ce qui est un saut d'ouverture pour l'ETF est ici un rendement Globex négociable, ce qui peut diluer le mécanisme.
- **Séances perdues au roll** : les jours d'expiration trimestrielle et les lundis suivants sont perdus (`c.0`), soit environ 8 séances par an.
- **EXT positif sur MNQ et MGC** (+2 665 et +1 463 USD sur 130 séances, t 1,7 à 1,9) : informatif, hors verdict, de la même taille que la variabilité annuelle observée.

## 6. Ce qui pourrait rouvrir l'hypothèse

Il faudrait une nouvelle campagne pré-enregistrée, sur une donnée nouvelle. Par exemple :
- ES/SPY 1993–2019 pour une réplication sur l'actif du papier ;
- un test limité aux régimes de crise défini ex ante (VIX haut), à juger sur l'extension future seulement.

Changer d'horaire, de filtre ou de variante sur ces données est exclu.

## Fichiers

- Design et audit : [research_design.md](research_design.md), [data_audit.md](data_audit.md), [audit_data.py](audit_data.py), [audit/](audit/).
- Runner : [run_r2.py](run_r2.py) (`selftest`, `a`, `b`, `c`, `div`).
- Résultats : [results/](results/).
  - Campagne A : `A_summary.csv`, `A_replication_regression.csv`, `A_gates.csv`, `A_years.csv`, `A_long_short.csv`, `A_vix_terciles.csv`, `A_distribution.csv`, `A_placebos.csv`, `A_execution_sensitivity.csv`, `A_rolling_252.csv`, `A_daily_*.csv`.
  - Campagnes B et C : `B_*.csv`, `C_*.csv`.
  - Diversification : `D_diversification.csv`, `D_leave_one_out.csv`, `D_sleeve_correlations.csv`, `D_book_drawdowns.csv`.
