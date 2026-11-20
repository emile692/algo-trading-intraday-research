# Règles des setups price action (post Reddit)

> **Statut : FIGÉ le 2026-11-21.** L'utilisateur a demandé le passage hors
> échantillon : les règles sont donc validées telles qu'écrites ici, avec les
> valeurs par défaut et les choix marqués ⚠ tels quels. Le passage unique est
> fait par `run_oos.py`. Empreintes SHA-256, prises avant ce passage :
> - `src/price_action/data.py` `81861f52…`
> - `signals.py` `bb7049aa…`
> - `execution.py` `029913f5…`
> - `metrics.py` `0731ba4c…`
> - `run_oos.py` `c4c47226…`
> - ce fichier, avant cet en-tête : `e699675a…`

Code : [src/price_action/](../../src/price_action/) · tests : [tests/test_price_action.py](../../tests/test_price_action.py)

## 1. Données et conventions communes

| Élément | Règle |
|---|---|
| Instruments | **MNQ et MES** (micros, exécution réelle), séries Databento `MNQ.v.0` et `MES.v.0`. Ce sont des contrats continus qui suivent le contrat le plus échangé, sans ajustement de prix. |
| Barres de base | 1 minute. Chaque barre est horodatée à son **début**, en heure de New York (ET). |
| Jour (séance) ⚠ | Séance régulière, **RTH 09:30–16:00 ET**. La nuit (Globex) est ignorée pour les plus hauts et plus bas. |
| Validité d'un jour | Au moins 180 barres de 1 min en RTH, et un seul contrat sur la séance. |
| Bougie de signal ⚠ | **5 min**, alignée sur 09:30, construite à partir du 1 min. Une bougie n'est connue qu'à sa **clôture** (début + 5 min). |
| Contrat | Une comparaison entre jours (J-2, J-1, J) n'est faite que si les trois séances sont sur le **même contrat** (`instrument_id`). Sinon, pas de signal. |
| Fenêtre d'entrée | La bougie de signal doit **clôturer entre 09:35 et 11:00 ET**, bornes incluses. |
| Fin de position | À plat à **15:10 CT** (clôture Topstep, paramètre) moins 1 min, soit à la clôture de la barre 16:08–16:09 ET. Aucune position n'est gardée la nuit. |
| Trades par jour ⚠ | Au plus **1 trade par jour, par setup et par instrument** : le **premier** signal du jour, long ou short. Si ce premier signal est rejeté (stop trop petit), il n'y a pas de trade ce jour-là, sans chercher de signal suivant. |
| ATR quotidien ⚠ | ATR de Wilder sur 14 séances RTH, calculé sur les séances closes jusqu'à J-1. **Il sert uniquement à décrire les trades** (taille du stop en ATR) et ne filtre rien. |

## 2. Exécution et coûts

| Élément | Règle |
|---|---|
| Entrée | À la clôture de la bougie de signal (`entry_ref`). Prix d'exécution net : entry_ref + **1 tick** contre soi. |
| Risque R | `|entry_ref − stop|`, fixé au moment du signal. Les résultats en R sont divisés par ce risque prévu. |
| Objectif | entry_ref ± **2 R**, arrondi au tick vers l'extérieur. Ordre limite, **0 tick** de glissement, sans amélioration de prix. |
| Stop | Ordre stop, **2 ticks** de glissement. Si une barre ouvre au-delà du stop, l'exécution se fait à l'ouverture de cette barre, moins 2 ticks. |
| Ordre stop / objectif | Le parcours du prix est simulé en 1 min après la clôture du signal. Si le stop et l'objectif sont touchés dans la **même minute**, c'est le **STOP** qui compte. Il n'existe pas de donnée plus fine. |
| Sortie forcée | À la clôture de 16:08–16:09 ET, avec 1 tick de glissement ⚠. |
| Commission ⚠ | **2,50 USD aller-retour par micro** (valeur par défaut du dépôt, à remplacer par celle de Topstep). |
| Valeur du tick | MNQ 0,25 pt = 0,50 USD ; MES 0,25 pt = 1,25 USD. Pour mémoire, NQ 5 USD et ES 12,50 USD. |
| Stop minimum | Si le stop prévu fait **moins de 8 ticks**, le trade est rejeté. |

## 3. Setup A — Inside Day False-Break

1. **J-1 est un inside day** par rapport à J-2, en RTH et au sens strict : plus haut(J-1) < plus haut(J-2) **et** plus bas(J-1) > plus bas(J-2). J-1 et J-2 doivent être des jours valides, sur le même contrat que J.
2. **Cassure** : à partir de la bougie de 09:30 de J, une bougie dont le plus haut dépasse le plus haut de J-1 (H1) ouvre une cassure haussière. ⚠ Une ouverture en gap au-dessus de H1 compte comme une cassure. **L'extrême de la cassure** est le plus haut atteint depuis cette bougie jusqu'à la bougie de signal comprise. La cassure baissière est symétrique, sous le plus bas de J-1 (L1).
3. **Signal** : la première bougie, à partir de la bougie de cassure comprise, dont la **clôture revient strictement dans le range** de J-1 (L1 < clôture < H1).
   - Après une cassure haussière : **short**, stop = extrême + 1 tick.
   - Après une cassure baissière : **long**, stop = extrême − 1 tick.
4. Si une même bougie donne à la fois un signal long et un signal short, le jour est ignoré.
5. Objectif 2 R fixe. Pas de stop suiveur, pas de passage au point mort.

## 4. Setup B — Intraday 3-Push

Toutes les bougies sont des bougies de 5 min de la séance, à partir de 09:30. Les extrêmes de séance ignorent la nuit.

1. **Pivot haut** à la bougie p :
   - son plus haut est strictement supérieur aux plus hauts des N bougies qui précèdent ;
   - il est supérieur ou égal aux plus hauts des N bougies qui suivent ;
   - il n'est **confirmé** qu'à la clôture de la bougie p+N.

   Le pivot bas est symétrique.
2. **Poussée haussière** : un pivot haut confirmé qui est aussi un nouveau plus haut de séance au moment où il se forme.
   - Deux poussées successives doivent être séparées par un pivot bas confirmé.
   - Si une nouvelle poussée apparaît sans pivot bas depuis la précédente, elle remplace la précédente, plus basse.
3. **Signal short** à la clôture d'une bougie j, si toutes ces conditions sont réunies :
   - au moins 3 poussées ;
   - j ≥ p3 + N, c'est-à-dire que la 3ᵉ poussée est confirmée ;
   - le plus haut de la 3ᵉ poussée est **toujours** le plus haut de la séance ;
   - la clôture de j est sous le **plus bas de la bougie de la 3ᵉ poussée**.

   Si un nouveau plus haut de séance apparaît avant ce signal, la séquence continue avec les 3 dernières poussées.

   Le stop est le plus haut de la 3ᵉ poussée + 1 tick.
4. Le signal long (trois poussées baissières) est symétrique.
5. N = **2** par défaut. Objectif 2 R fixe.

## 5. Protocole

- **Dans l'échantillon (IS)** : du 2021-01-01 au 2023-12-31. Les données démarrent au 2020-11-01 pour avoir J-2 et l'ATR dès le début.
- **Hors échantillon (OOS)** : du 2024-01-01 jusqu'à la dernière séance téléchargée, le 2026-11-19. **Un seul passage**, avec les paramètres figés, après validation de ce fichier.
- **Paramètres libres** (3 au plus par setup), explorés sur l'IS seulement :
  - A : unité de temps des bougies {5, 15 min} × fin de fenêtre {10:30, 11:00, 11:30} ;
  - B : N {1, 2, 3} × fin de fenêtre {10:30, 11:00, 11:30}, en 5 min.
- **Règle de choix des paramètres** (écrite avant tout calcul) : **les valeurs par défaut (5 min, 11:00, N = 2) sont figées**. La grille sert seulement à voir si l'IS forme un plateau ou un pic isolé. Aucune autre cellule n'est retenue, sauf décision explicite de ta part avant le passage hors échantillon.
- **Robustesse** : sur l'IS, chaque paramètre varie de ±20 % autour du défaut, avec les autres fixés :
  - bougie 4 et 6 min ;
  - durée de la fenêtre 72 et 108 min, soit fin à 10:42 et 11:18 ;
  - N = 1 et 3 ;
  - stop minimum 6 et 10 ticks ;
  - objectif 1,6 et 2,4 R.

  Si le PF net s'effondre, c'est du surajustement.
- **Critères de validation hors échantillon** : PF net ≥ 1,2 sur MNQ **et** sur MES, au moins 100 trades par instrument, et une espérance positive au moins 2 années sur 3 (2024, 2025, 2026). Sinon, le setup est **rejeté**, sans ajouter de filtre.
