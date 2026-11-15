# Phase 10 — Stop catastrophe opérationnel

*Stop contre une déconnexion ou un bug, pas un stop alpha. Positions nettes du book de référence (S01 = ATR 18), exécution agrégée sans limite de lots. MNQ et MGC du 2019-05-06 au 2026-03-19, MBT du 2021-05-03 au 2026-03-19. ATR14 quotidien RTH de J−1, en points. Le remplissage se fait au stop moins un tick ; les gaps à travers le stop ne sont pas modélisés. Script : [phase10_cat_stop.py](phase10_cat_stop.py). Tables : [10_triggers.csv](outputs/10_triggers.csv), [10_summary.csv](outputs/10_summary.csv), [10_bounds.csv](outputs/10_bounds.csv).*

## Règle de netting du stop

Un seul ordre stop par instrument, posé chez le broker, pour la position nette :

- **ouverture depuis flat** : prix moyen = fill ; stop = prix moyen ∓ k × ATR(J−1) ;
- **augmentation dans le même sens** : prix moyen pondéré, stop recalculé, quantité ajustée ;
- **réduction** : prix moyen et niveau conservés, quantité réduite ;
- **retournement** : prix moyen = fill du nouveau sens, stop recalculé de l'autre côté ;
- **retour à plat** : ordre annulé.

Une fois déclenché, l'instrument reste à plat jusqu'à la fin de la séance. Les sleeves continuent leur PnL virtuel. Ce stop est indépendant des ordres stop et cible de S01, qui restent posés.

## Résultats

| k | Déclenchements MNQ | MGC | MBT | PnL évité (+) / perdu (−) total | Borne de perte de la position ouverte : médiane | p99 | max | Jours où la borne dépasse 500 USD |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,5 | 4 | 0 | 2 | −418 | 1 250 | 4 196 | 6 515 | 81,5 % |
| 2,0 | 0 | 0 | 1 | −355 | 1 666 | 5 593 | 8 684 | 85,4 % |
| 3,0 | 0 | 0 | 0 | 0 | 2 499 | 8 385 | 13 021 | 89,1 % |

La « borne » est la perte maximale de la position ouverte si le stop est touché : position nette maximale de la journée × k × ATR × valeur du point, plus les coûts, sommée sur les instruments. Elle ne compte pas les pertes déjà réalisées dans la journée.

## Déclenchements historiques

| k | Date | Instrument | Position | Entrée moyenne | Stop | PnL réel | PnL avec stop | Évité (+) / perdu (−) |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1,5 | 2019-08-01 | MNQ | +1 | 7 919,25 | 7 792,15 | −274 | −258 | 16 |
| 1,5 | 2020-07-13 | MNQ | +1 | 11 021,00 | 10 721,38 | −887 | −790 | 97 |
| 1,5 | 2021-11-10 | MNQ | +1 | 16 191,00 | 15 918,11 | −448 | −549 | −101 |
| 1,5 | 2022-11-08 | MBT | -1 | 19 270,00 | 20 052,68 | 201 | −128 | −328 |
| 1,5 | 2023-11-09 | MBT | +2 | 37 677,50 | 35 924,11 | −300 | −414 | −114 |
| 1,5 | 2025-03-04 | MNQ | -1 | 20 102,50 | 20 662,91 | −1 138 | −1 124 | 14 |
| 2,0 | 2022-11-08 | MBT | -1 | 19 270,00 | 20 313,57 | 201 | −154 | −355 |

## Limite de perte journalière

La DLL d'un compte 50K Topstep est de 1 000 USD. Elle est appliquée automatiquement aux nouveaux comptes (Responsible Trading Program). Sur 1775 séances, le book de référence :

- a fini la journée à −1 000 USD ou moins **3 fois** ;
- a touché −1 000 USD en intraday (marquage adverse) **11 fois**.

Ces journées auraient été interrompues par la DLL. C'est un effet de la règle Topstep, distinct du stop catastrophe.

## Recommandation

**Aucun k de {1,5 ; 2 ; 3} ne borne la perte d'un jour à moins de X = 50 % de la DLL, soit 500 USD.** Même à k = 1,5, la borne de la position ouverte dépasse 500 USD sur 81 % des jours (médiane 1 250 USD). Un ATR journalier MNQ vaut plusieurs centaines de points, à 2 USD le point, et S01 peut porter 4 contrats. Pour borner le p99 à 500 USD, il faudrait k ≈ 0,18 : ce serait un stop alpha serré, qui couperait les trades ordinaires.

Je recommande donc de séparer les deux rôles.

- **Stop catastrophe chez le broker : k = 3.** Il ne s'est jamais déclenché sur l'historique (2019–2026 pour MNQ et MGC, 2021–2026 pour MBT), ce qui veut dire qu'il ne touche pas au comportement des sleeves. À k = 1,5, il se serait déclenché 6 fois pour −418 USD au total ; à k = 2, une fois pour −355 USD. Son rôle est de borner une position orpheline (déconnexion, bug) à environ 3 ATR, pas de borner une journée.
- **Borne journalière : la DLL Topstep de 1 000 USD (X = 100 %)**, appliquée par Topstep au niveau du compte. Historiquement, elle aurait interrompu 11 séances en intraday sur 1775. Un plafond plus serré relève d'une décision de risque au niveau du book. L'étude d'agrégation du 13/11 a rejeté un stop book de 1 000 USD, qui coupait plus de gains qu'il n'évitait de pertes.

Limites : remplissage au niveau du stop (pas de gap modélisé), barres minute uniquement, positions sans limite de lots (Combine). Le comportement réel sur un gap d'ouverture ou une coupure de flux n'est pas testé.
