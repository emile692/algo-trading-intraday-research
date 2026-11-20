# S01 — les deux mécanismes d'assèchement, plan 2 × 2 — clôture

- **Statut** : `CLOSED_IS`. Le bras jugé, C2, échoue au gate G3 (Sharpe IS). Aucun holdout n'a été lu.
- **Date** : 2026-11-18.
- **Spécification** : [FROZEN_SPECIFICATION.json](FROZEN_SPECIFICATION.json), sha256 `91cb2632…f3e93` ([SPEC_LOCK.json](SPEC_LOCK.json)), figée avant tout calcul sur C1 et C2.
- **Commande** : `python -m research.s01_scale_invariant_20261118.run_campaign run`. Les contrôles techniques se relancent avec `… selftest`.
- **Demande** : de l'utilisateur, le 2026-11-18, « challenger les 2 mécanismes qui refroidissent S1 » sur l'IS habituel, avec un plafond de risque de 500 USD par contrat.
- **Holdouts non lus** :
  - PRE, NQ 2010-06 → 2019-05 : toujours vierge pour l'ORB ;
  - VAL et EXT, pour C1 et C2.

## Plan

| | Saut si un contrat risque plus de 250 USD (actuel) | Saut si la distance au stop dépasse X % du prix, plancher d'1 contrat, plafond 500 USD |
|---|---|---|
| **Bandes d'ATR en points (actuel)** | REF | C1, pour l'attribution seulement |
| **Bandes en ATR / prix** | A, IS déjà vu le 15/11 | **C2, seul bras jugé** |

Le seuil X est calibré sur l'IS sans lire le PnL. Il saute exactement la même proportion de signaux IS que la règle des 250 USD :

| Bras | Signaux IS | Sautés en IS | X calibré |
|---|---:|---:|---:|
| C1 | 553 | 39 (7,1 %) | 1,06 % du prix |
| C2 | 552 | 44 (8,0 %) | 0,99 % du prix |

Le plafond de 500 USD ne joue jamais sur l'IS.

## Contrôles techniques

- **T1** : les fonctions réutilisées reproduisent exactement les résultats IS de A et de B (trades, PnL, Sharpe).
- **T2** : les seuils ATR / prix recalibrés sont identiques aux seuils figés (écart de 1·10⁻¹⁶).
- **T3** : le saut en % reproduit la proportion de sauts de la règle en USD (39 sur 39 pour C1, 44 sur 44 pour C2).
- **Incident** : une première tentative s'est arrêtée dans la calibration, avant toute métrique. Le budget « illimité » de 1e9 USD est refusé par le moteur ; il a été remplacé par 50 000 USD, soit 100 % du compte et le maximum accepté. Aucun contrat MNQ ne risque autant, donc la règle de la spécification est inchangée. Log : [results/run_log_attempt1.txt](results/run_log_attempt1.txt).

## Résultats

### Fréquence (étape 0, sans PnL)

| Bras | Médiane trimestrielle IS | Médiane 2025 T1 → 2026 T3 | Ratio (seuil 0,60) | 2026 T1 / T2 / T3 |
|---|---:|---:|---:|---|
| REF | 32,5 | 10 | 0,31 | 2 / 9 / 2 |
| A | 29 | 15 | 0,52 | 9 / 15 / 8 |
| C1 | 32 | 15 | 0,47 | 3 / 14 / 9 |
| **C2** | 30 | 30 | **1,00** | 31 / 37 / 30 |

### IS 2019-05-05 → 2024-02-27

Sur 1 252 séances : PnL quotidien net en USD, Sharpe annualisé par √252 avec les jours sans trade, t HAC de Lo.

| Bras | Trades | Net USD | Sharpe | t HAC | maxDD | PF |
|---|---:|---:|---:|---:|---:|---:|
| REF | 514 | 13 852 | 1,251 | 3,09 | −2 149 | 1,32 |
| A | 508 | 10 598 | 0,976 | 2,46 | −2 291 | 1,25 |
| C1 | 514 | 13 685 | 1,235 | 3,08 | −2 130 | 1,32 |
| **C2** | 508 | 9 727 | **0,890** | 2,27 | −2 272 | 1,22 |

**Gates de C2** :
- G0 (fréquence ≥ 0,60) : 1,00, réussi ;
- G1 (au moins 386 trades) : 508, réussi ;
- G2 (net > 0 et t ≥ 2) : t = 2,27, réussi ;
- **G3 (Sharpe ≥ 0,938) : 0,890, échec** ;
- G4 (au moins 3 années positives sur 5) : 5 sur 5, réussi.

Le PSR(0) vaut 0,979 et le DSR sur 4 essais (A, B, C1, C2) 0,947.

### Attribution 2 × 2 sur l'IS

| Effet moyen | Net USD | Sharpe | Trades |
|---|---:|---:|---:|
| M1 : bandes en points → ATR / prix | **−3 606** | **−0,31** | −6 |
| M2 : saut à 250 USD → saut en % du prix, avec plancher et plafond | −519 | −0,05 | 0 |

## Lecture

- **M2 (le dimensionnement) se corrige presque sans coût sur l'IS.** Avec les bandes actuelles, passer au saut en % du prix (C1) donne 13 685 USD contre 13 852, pour un Sharpe de 1,235 contre 1,251. La règle en % est invariante d'échelle et ne dégrade presque rien. Seule, elle ne rend pourtant pas la fréquence (ratio 0,47), parce que les bandes en points coupent toujours.
- **Tout l'écart de performance IS vient de M1 (les bandes).** Passer les bandes en ATR / prix coûte environ 3 600 USD et 0,31 de Sharpe sur l'IS, avec ou sans le nouveau dimensionnement. Or c'est aussi M1 qui rend la fréquence : la fréquence et la performance IS reposent sur le même mécanisme.
- **Deux explications restent possibles, et l'IS ne permet pas de trancher.**
  - (a) Biais de sélection : les bandes en points de REF viennent d'un cluster choisi sur ce même IS, celles en ATR / prix non.
  - (b) L'ATR en points porte une information économique réelle. Les frictions de S01 sont en ticks (buffers de 2 ticks, 1 tick de slippage, commission fixe), donc un ATR faible en points signifie des frictions proportionnellement plus lourdes. Si c'est le cas, la borne basse en points aurait un sens, mais pas la borne haute, qui est celle qui assèche S01 en 2026.
- **Le holdout NQ 2010–2019 ne peut pas juger les bandes en points.** À des prix de 1 800 à 7 500, l'ATR en points est presque toujours sous la borne basse de 7,75 points, donc REF n'y traderait presque pas. Ce filtre ne se transpose pas d'un niveau de prix à l'autre : il ne peut donc pas être une règle structurelle telle quelle. Cela ne tranche pas pour autant entre (a) et (b).
- **Rien n'est sauvé après coup.** Lire maintenant le holdout pour C2, ou abaisser G3, serait un sauvetage interdit par la spécification.

## Conséquences

- **S01 ATR 18 reste la sleeve de référence, inchangée.**
- **Acquis utilisable** : la règle de saut en % du prix (X ≈ 1 %, plafond 500 USD) est neutre sur l'IS avec les bandes actuelles. Elle n'est pas validée hors échantillon : C1 n'était pas un bras jugé.
- **Le holdout NQ 2010–2019 reste vierge pour l'ORB.**

## Réouverture

- **Toute réouverture demande un nouvel identifiant de campagne**, et une spécification qui déclare vus les IS de A, B, C1 et C2 ainsi que ce document.
- **Piste non testée, issue de cette lecture** : borne basse de la bande en points (ou en ticks, pour les frictions) et borne haute en ATR / prix. C'est une idée née de l'IS (cinquième essai sur le même IS) : seuls le holdout NQ et le forward pourraient la juger proprement.
- **Ce qu'il ne faut pas faire** :
  - lire le holdout pour C2 ;
  - relancer C2 avec un G3 abaissé ;
  - ajuster X ou le plafond.

## Fichiers

| Fichier | Contenu |
|---|---|
| [results/selftest.json](results/selftest.json) | T1, T2 |
| [results/calibration_T3.json](results/calibration_T3.json) | seuils X et contrôle T3 |
| [results/stage0.json](results/stage0.json), [results/stage0_trades_per_quarter.csv](results/stage0_trades_per_quarter.csv) | fréquence, sans PnL |
| [results/stage1.json](results/stage1.json), [results/stage1_is_daily.csv](results/stage1_is_daily.csv) | métriques IS, gates, attribution, PSR et DSR |
| [results/stage2.json](results/stage2.json) | `CLOSED_IS`, `read: false` |
| [results/run_log.txt](results/run_log.txt), [results/run_log_attempt1.txt](results/run_log_attempt1.txt) | logs |
