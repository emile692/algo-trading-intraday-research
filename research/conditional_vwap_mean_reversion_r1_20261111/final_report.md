# R1 — Conditional VWAP Mean Reversion : rapport final

Date : 2026-11-11.

## Verdict : **FAIL**

Aucune des trois hypothèses n'est validée. Les 18 tests pré-enregistrés (6 définitions × MNQ, MGC, MCL) échouent tous la porte IS. Aucun ne franchit la première condition (G1 : PnL net positif avec t ≥ 2).

| Constat | Chiffres |
|---|---|
| Cellules centrales IS positives après coûts | 2/18, à +42 et +32 USD sur cinq ans, t ≈ 0 |
| Toutes cellules IS positives après coûts | 11/126 ; meilleur t 1,07 |
| Holdouts NQ 2010–2019 et CL 2015–2021 | 1 cellule positive sur 84 ; les 12 cellules centrales négatives, t de −2,5 à −6,9 |
| Brut (avant coûts) | négatif sur MNQ pour H1 et H3, légèrement positif mais inférieur aux coûts sur MGC, négatif sur MCL |

L'OOS (2024-01 → 2026-03) **n'a pas été lu**. MES n'a pas été lancé, sa condition de seconde étape n'étant pas remplie. L'analyse de diversification n'a pas lieu d'être : il n'y a aucun candidat, et elle aurait exigé de lire l'OOS.

Protocole :
- spécification gelée avant tout calcul de rendement : [FROZEN_SPECIFICATION.json](FROZEN_SPECIFICATION.json), sha256 `a0742a03…af40f`, [SPEC_LOCK.json](SPEC_LOCK.json) ;
- self-test synthétique passé : sur une marche aléatoire, R brut 0,00 ; sur un processus mean-reverting, R brut +0,72 ; causalité vérifiée sur les 6 tests ; stop prioritaire dans une minute ambiguë.

## 1. Ce qui a été testé

Le détail est dans [research_design.md](research_design.md) et la spécification gelée.

- **Exécution** :
  - données en barres d'une minute ; signaux à la clôture de barres de 5 minutes ;
  - entrée à l'open de la minute suivante + 1 tick ;
  - stop au pire du niveau et de l'open, + 1 tick ;
  - objectif en limite, rempli seulement si le prix traite 1 tick au-delà ; stop prioritaire dans une minute ambiguë ;
  - commission 1,25 USD par côté, 1 contrat, une position à la fois.
- **Séances exclues** : séances complètes à moins de 90 % et séances avant chaque roll.

| Test | Signal (cellule centrale) | Sortie |
|---|---|---|
| H1-N1 | \|close − VWAP\| / σ_VWAP ≥ 2,25 et efficiency ratio ≤ 0,30 | VWAP, stop 1× l'écart, 60 min |
| H1-N2 | écart au VWAP / (σ_jour · √τ) ≥ 1,0 et ER ≤ 0,30 | idem |
| H2-D1 | mouvement de 30 min ≥ 2,5 σ, puis contraction de la volatilité (10 min ≤ 0,5 × 20 min précédentes) | 50 % du mouvement, stop au-delà de l'extrême, 60 min |
| H2-D2 | mouvement ≥ 2,5 σ, puis barre d'arrêt (sens opposé, pas de nouvel extrême) | idem |
| H3-VWAP | 11:30–13:30, \|z N1\| ≥ 2,0 et volume des 30 min ≤ médiane des 20 séances | VWAP, sortie 14:00 |
| H3-RANGE | 11:30–13:30, prix dans le décile extrême du range de séance et volume faible | milieu du range |

## 2. Cellules centrales, IS (2019-05 → 2023-12 ; MCL depuis 2021-07)

| Instrument | Test | Trades | Net USD | t NW | R moyen net | R moyen brut | Réussite | PF | Long / short USD | Placebo p |
|---|---|---|---|---|---|---|---|---|---|---|
| MNQ | H1-N1 | 766 | −6 690 | −2,24 | −0,063 | −0,028 | 47 % | 0,80 | −4 826 / −1 864 | 0,79 |
| MNQ | H1-N2 | 460 | −5 982 | −2,37 | −0,078 | −0,052 | 47 % | 0,77 | −4 747 / −1 235 | 0,87 |
| MNQ | H2-D1 | 299 | +42 | 0,03 | +0,062 | +0,148 | 35 % | 1,00 | +79 / −37 | **0,018** |
| MNQ | H2-D2 | 180 | −115 | −0,11 | −0,035 | +0,037 | 40 % | 0,98 | −684 / +569 | 0,35 |
| MNQ | H3-VWAP | 229 | −1 069 | −0,82 | −0,069 | −0,031 | 46 % | 0,87 | +406 / −1 475 | 0,47 |
| MNQ | H3-RANGE | 480 | −5 230 | −2,71 | −0,106 | −0,069 | 40 % | 0,74 | −55 / −5 176 | 0,94 |
| MGC | H1-N1 | 522 | −1 882 | −2,10 | −0,077 | +0,024 | 52 % | 0,80 | −1 353 / −529 | 0,25 |
| MGC | H1-N2 | 325 | +32 | 0,04 | −0,001 | +0,063 | 53 % | 1,01 | +392 / −361 | **0,030** |
| MGC | H2-D1 | 269 | −860 | −1,26 | −0,114 | +0,073 | 35 % | 0,84 | −850 / −10 | 0,18 |
| MGC | H2-D2 | 193 | −1 385 | −1,92 | −0,166 | −0,027 | 38 % | 0,72 | −765 / −619 | 0,54 |
| MGC | H3-VWAP | 82 | −542 | −1,61 | −0,095 | +0,028 | 45 % | 0,58 | −279 / −263 | 0,31 |
| MGC | H3-RANGE | 318 | −1 036 | −1,86 | −0,068 | +0,024 | 46 % | 0,76 | −772 / −264 | 0,26 |
| MCL | H1-N1 | 360 | −1 902 | −1,70 | −0,116 | −0,042 | 48 % | 0,78 | −621 / −1 282 | 0,91 |
| MCL | H1-N2 | 216 | −2 794 | −3,16 | −0,169 | −0,105 | 41 % | 0,59 | −1 412 / −1 382 | 0,995 |
| MCL | H2-D1 | 119 | −989 | −1,56 | −0,396 | −0,214 | 24 % | 0,66 | −206 / −783 | 0,97 |
| MCL | H2-D2 | 68 | −62 | −0,13 | −0,133 | −0,006 | 37 % | 0,97 | +147 / −209 | 0,51 |
| MCL | H3-VWAP | 113 | −1 509 | −2,48 | −0,243 | −0,169 | 39 % | 0,56 | −847 / −663 | 0,99 |
| MCL | H3-RANGE | 274 | −2 494 | −2,77 | −0,166 | −0,097 | 41 % | 0,69 | −1 118 / −1 377 | 0,98 |

### Holdouts (cellules centrales)

| | H1-N1 | H1-N2 | H2-D1 | H2-D2 | H3-VWAP | H3-RANGE |
|---|---|---|---|---|---|---|
| NQ 2010–2019, économie MNQ : net USD (t) | −5 101 (−5,8) | −3 791 (−4,1) | −2 269 (−4,4) | −2 342 (−5,5) | −1 334 (−3,2) | −4 190 (−5,8) |
| NQ, R brut | −0,031 | −0,032 | −0,046 | −0,147 | −0,006 | −0,013 |
| CL 2015–2021, économie MCL : net USD (t) | −5 709 (−6,8) | −1 601 (−2,5) | −2 082 (−4,4) | −1 649 (−3,8) | −1 896 (−4,1) | −5 237 (−6,9) |

Les holdouts sont négatifs **avant coûts** pour les 12 cellules. Le rejet ne dépend donc pas d'un modèle de coûts trop sévère.

## 3. Lecture par hypothèse

**H1, écart au VWAP.**
- Rejet sur les trois instruments, sous les deux normalisations, avec ou sans filtre de tendance.
- Sur MNQ, le brut est négatif (−0,03 à −0,05 R). Le prix éloigné du VWAP continue plus souvent qu'il ne revient, comme le montrait déjà l'event study de la campagne du 2026-10-06. Le filtre anti-tendance (ER ≤ 0,2 ou 0,3) ne l'inverse pas : 0/9 cellules H1-N1 positives sur MNQ, 0/9 sur MGC et MCL.
- Sur MGC, le brut est légèrement positif (+0,02 à +0,06 R), mais inférieur au coût (≈ 0,08 R).
- MAE médian −0,5 R, MFE médian +0,4 à +0,5 R : la distribution est symétrique, sans bord.

**H2, épuisement de volatilité.** C'est la seule hypothèse avec un signe positif sur MNQ :
- brut +0,15 R sur H2-D1 ;
- placebo p 0,018 : le timing fait mieux qu'un instant aléatoire de la même demi-heure ;
- la cellule « avec filtre de volume » donne +910 USD (t 0,67).

Mais le net est nul (+42 USD sur 299 trades, t 0,03), coût ×2 négatif, 3 années sur 5 positives, 2 cellules positives sur 6. Et la même règle perd sur NQ 2010–2019 (−2 269 USD, t −4,4, brut −0,05 R). C'est une observation post-hoc à faible puissance, pas un candidat. Sur MGC et MCL, H2 est négative.

**H3, heure du déjeuner.**
- Rejet. H3-RANGE est le test le plus nettement négatif sur MNQ (−5 230 USD, t −2,7, 0/6 cellules positives, côté short −5 176 USD).
- Le filtre « faible volume » n'aide pas : MNQ H3-VWAP passe de +1 165 USD sans filtre (k = 2,5) à des pertes avec filtre.
- La fenêtre creuse ne rend pas le prix plus mean-reverting.

## 4. Robustesse et contrôles

- **Coûts ×2 / entrée retardée d'une minute** : aucune cellule centrale positive à coût ×2. L'entrée retardée améliore légèrement H1-N1 sur MNQ (de −6 690 à −3 969 USD) sans le rendre positif.
- **Années** : aucune cellule centrale n'a plus de la moitié de ses années positives avec moins de 50 % du total dans une année. MNQ H1-N1 perd 3 798 USD en 2022 seule.
- **Régimes de volatilité** : pas de régime où H1 devient positive (R moyen de −0,04 à −0,17 dans tous les terciles). MNQ H2-D1 est positive en VIX bas et haut (+0,32 et +0,12 R) et très négative en VIX normal (−0,40 R) : non monotone, sans lecture économique.
- **Heure d'entrée** : aucune demi-heure systématiquement positive sur H1. Les rares cases positives changent de demi-heure d'un instrument à l'autre.
- **Stabilité des régions** : moyenne glissante sur 252 séances positive dans 1 à 12 % des fenêtres pour MNQ H1, 4 % pour H3-RANGE, 47 % au mieux (MNQ H2-D1).
- **PBO** sur les 42 cellules : MNQ 0,11, MGC 0,32, MCL 0,29. Le classement des cellules est assez stable, mais il classe des cellules négatives.
- **Walk-forward de la règle de sélection** (meilleur Sharpe lissé sur le voisinage, années glissantes) : négatif sur les 6 tests. Choisir la cellule ne sauve aucune hypothèse.
- **Heatmaps** : [results/heatmaps/](results/heatmaps/), IS par instrument et PRE. Aucune région positive contiguë.

## 5. Diversification

Non calculée. La règle pré-enregistrée la réserve aux candidats validés en OOS. Il n'y en a aucun, et la calculer aurait exigé de lire l'OOS, que l'échec IS interdit.

## 6. Ce qu'il faut retenir

1. Sur les futures indices, l'écart intraday au VWAP **ne se résorbe pas plus souvent qu'il ne s'étend**, même hors journées de tendance. C'est la troisième campagne (avec `mean_reversion_cross_asset_20261006` et le fast-alpha) qui trouve continuation ou neutralité plutôt que réversion intraday sur MNQ/NQ.
2. Sur l'or, un faible biais de réversion brut existe (+0,02 à +0,07 R), mais il est plus petit qu'un aller-retour (≈ 0,08 R) : même schéma que le momentum, coût-dominé.
3. Sur le pétrole, toutes les variantes sont négatives, déjà avant coûts, IS comme 2015–2021.
4. Le seul signal de timing (MNQ H2-D1, placebo p 0,018) ne survit ni aux coûts ni au holdout NQ.

## 7. Réouverture

Ce rejet s'applique à toutes les variantes de seuil, de normalisation, de filtre de tendance, de fenêtre horaire et de volume testées ici.

Une réouverture exige une nouvelle campagne pré-enregistrée sur une **information nouvelle** (carnet d'ordres, flux, données tick), pas un nouveau seuil. En particulier, l'épuisement MNQ avec volume (H2-D1, +910 USD, t 0,67) a été vu après coup : il ne peut pas être repris tel quel.

## Fichiers

- **Design et audit** : [research_design.md](research_design.md), [data_audit.md](data_audit.md), `audit_data.py`, `audit_thresholds.py`, [audit/](audit/).
- **Runner** : [run_r1.py](run_r1.py) (`selftest`, `is` ; `oos` refusé par construction).
- **Résultats** dans [results/](results/) :
  - toutes les cellules : `cells_all.csv` ;
  - portes : `gates_IS.csv` ;
  - contrôles : `placebo_IS.csv`, `cost_delay_IS.csv` ;
  - découpages : `long_short_IS.csv`, `years_IS.csv`, `regimes_IS.csv`, `entry_hour_IS.csv`, `distribution_mae_mfe_IS.csv` ;
  - stabilité : `walk_forward_IS.csv`, `pbo_IS.csv`, `rolling252_IS.csv` ;
  - trades des cellules centrales : `trades/` ;
  - `heatmaps/`.
