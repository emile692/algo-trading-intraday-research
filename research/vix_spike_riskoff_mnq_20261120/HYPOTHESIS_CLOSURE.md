# Risk-off VIX sur les sleeves MNQ — clôture

- **Statut** : `CLOSED_IS`, puis **OOS lu sur décision de l'utilisateur : échec de V1 et V2**. La cellule sélectionnée échoue au PBO (G2) et au bootstrap (G3).
- **Date** : 2026-11-20.
- **Spécification** : [FROZEN_SPECIFICATION.json](FROZEN_SPECIFICATION.json), sha256 `692433e2…c1be3` ([SPEC_LOCK.json](SPEC_LOCK.json)), figée avant tout calcul de PnL avec overlay.
- **Commande** : `python -m research.vix_spike_riskoff_mnq_20261120.run_campaign run`.
- **Demande** : de l'utilisateur, le 2026-11-20 : « si le VIX monte de plus de x % sur les y derniers jours alors on coupe z jours ».
- **VAL et EXT** : lus une fois le 2026-11-20, pour la cellule retenue seulement, sur demande explicite de l'utilisateur (« adopte la meilleure config et teste-la en OOS maintenant »). C'est une dérogation documentée au protocole. Voir la section dédiée.

## Règle testée

- **Déclencheur** : la clôture du VIX de la veille est en hausse d'au moins x % par rapport à celle d'il y a y séances VIX.
- **Action** : S01, S02 et S03 sont à plat sur les z séances MNQ suivantes. Un nouveau déclenchement prolonge la coupure.
- **Grille** : x ∈ {10, 20, 30, 50} %, y ∈ {1, 2, 3, 5, 10}, z ∈ {1, 2, 3, 5, 10}, soit 100 cellules.
- **Sélection, sur l'IS seul** : la cellule au meilleur score de plateau, c'est-à-dire la médiane du Sharpe de la cellule et de ses voisines.

## Contrôles techniques

- **T1** : les deux fichiers VIX sont identiques sur 9 132 jours communs.
- **T2** : le panel quotidien MNQ est égal à la somme des trois colonnes MNQ du lock du 15/11 sur 572 séances (écart 0).
- **T3** : chaque séance MNQ a une clôture VIX au plus 4 jours avant.

## Résultats sur l'IS 2019-05-06 → 2023-12-29

Sous-book MNQ (S01 + S02 + S03), 1 203 séances :

| | Sans overlay | Cellule sélectionnée (x 50 %, y 10, z 10) |
|---|---:|---:|
| Séances coupées | 0 | 149 (12,4 %) |
| PnL net | 26 025 USD | 27 147 USD |
| Sharpe | 1,418 | 1,623 |
| Drawdown max | −5 027 USD | −2 569 USD |
| Jours sous −500 USD | 19 | 11 |

| Gate | Seuil | Résultat |
|---|---|---|
| G1 : gain de Sharpe et plateau | ≥ +0,10 et plateau > base | +0,21, réussi |
| **G2 : PBO** (CSCV, 10 blocs, 101 configurations) | ≤ 0,40 | **0,90, échec** |
| **G3 : bootstrap apparié** | P(gain > 0) ≥ 0,90 | **0,80, échec** (intervalle 5–95 % : −0,12 à +0,61) |
| G4 : coupures et PnL | ≤ 20 % des séances, PnL ≥ 85 % | 12,4 % et 104 %, réussi |

Sur la grille, 71 cellules sur 100 battent le Sharpe sans overlay, dont 50 d'au moins 0,10.

## Lecture

- **Tout le gain vient d'un seul épisode, le krach Covid.**
  - Le drawdown maximal sans overlay est le −5 027 USD du 4 au 24 mars 2020.
  - Sur février–avril 2020, le sous-book perd 1 242 USD sans overlay et gagne 1 241 USD avec.
- **Hors février–avril 2020, l'overlay ne sert à rien.**
  - La cellule sélectionnée passe de 1,585 à 1,598 de Sharpe, mais coûte 1 360 USD.
  - Sur la grille, seules 22 cellules sur 100 battent la base, avec un écart médian de −0,06 de Sharpe.
  - Ce calcul est descriptif, sur l'IS seul, et fait après les gates.
- **Le PBO de 0,90 en est la conséquence.** La meilleure cellule dépend de la présence ou non de mars 2020 dans la moitié qui sert à la choisir. Ce qui ressemble à une règle n'est qu'un épisode unique.
- **C'est cohérent avec l'audit** : les sleeves MNQ gagnent aussi les jours de vol haute, et couper après une hausse du VIX retire ces jours-là.

## Ce que la clôture ne dit pas

Pour un compte Topstep, une perte de 5 000 USD en trois semaines suffit à perdre le compte (MLL de 2 000 USD). La règle protège contre ce type d'événement, mais un seul épisode dans l'IS ne permet pas de le prouver statistiquement.

L'adopter quand même comme assurance de crise serait une décision de gouvernance, et non une règle validée. Elle aurait un coût probable en dehors des crises, et il faudrait la nommer comme telle. Il s'agirait aussi d'une nouvelle hypothèse, puisque le critère passerait de « meilleur Sharpe » à « survie du compte ». Elle demanderait un nouvel identifiant, une spécification qui déclare vus l'IS de cette grille et cette lecture, et un verdict sur la VAL non lue, avec un critère de queue fixé avant.

## Lecture OOS sur décision de l'utilisateur (dérogation, 2026-11-20)

Après la clôture, l'utilisateur a demandé d'adopter la meilleure configuration et de la tester en OOS. La lecture a été faite une seule fois par [oos_user_override.py](oos_user_override.py), avec les critères V1 et V2 de la spécification inchangés. Résultat dans [results/oos_user_override.json](results/oos_user_override.json).

| Sous-book MNQ | VAL sans overlay | VAL avec overlay | EXT sans overlay | EXT avec overlay |
|---|---:|---:|---:|---:|
| Séances (coupées) | 572 | 572 (83, 14,5 %) | 137 | 137 (10) |
| PnL net | 15 907 USD | **10 630 USD** | 30 USD | −1 759 USD |
| Sharpe | 1,287 | **1,073** | 0,01 | −0,63 |
| Drawdown max | −3 894 USD | −2 961 USD | −3 644 USD | −3 644 USD |
| Pire série de 15 séances | −3 451 USD | −1 858 USD | −2 031 USD | −2 031 USD |
| Jours sous −1 000 USD | 2 | 2 | 2 | 2 |

**V1 (le Sharpe ne baisse pas) : échec. V2 (au moins 85 % du PnL) : échec, à 67 %.**

Le PnL perdu les jours coupés vaut 5 277 USD sur la VAL. Il se répartit ainsi :
- **avril 2025** (krach des droits de douane) : les sleeves y gagnaient +5 067 USD pendant la coupure ;
- **mars 2025** : la coupure évite −2 313 USD ;
- **les autres épisodes** (janvier, août et décembre 2024, mars 2026) : de faibles gains manqués.

L'overlay réduit la queue : le drawdown max baisse de 930 USD et la pire série de 15 séances de 1 600 USD. Mais il coûte 5 300 USD de PnL, soit environ 5,7 USD de gain abandonné par dollar de drawdown évité. Sur l'EXT, il coûte 1 789 USD sans rien réduire.

**Conclusion.** L'OOS confirme la lecture de l'IS : en mars 2020, le krach a frappé les sleeves MNQ, mais en avril 2025 elles ont gagné dans le krach. La règle n'est pas adoptée. Le lock du 15/11 est inchangé.

## Réouverture

- **Toute réouverture demande un nouvel identifiant.**
- **Ce qu'il ne faut pas faire** :
  - choisir une autre cellule de cette grille ;
  - relâcher le PBO ou le bootstrap ;
  - réutiliser la VAL pour cette famille de règles : elle est maintenant lue.

## Fichiers

| Fichier | Contenu |
|---|---|
| [results/selftest.json](results/selftest.json) | T1 à T3 |
| [results/is_grid.csv](results/is_grid.csv) | les 100 cellules, IS seul |
| [results/is.json](results/is.json) | base, cellule sélectionnée, PBO, bootstrap, gates, dates coupées |
| [results/validation.json](results/validation.json) | `CLOSED_IS`, `read: false` |
| [results/run_log.txt](results/run_log.txt) | log |
| [results/oos_user_override.json](results/oos_user_override.json) | lecture OOS sur décision de l'utilisateur |
