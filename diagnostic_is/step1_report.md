# Étape 1 — anatomie des pertes IS

Périmètre strict: A du 2010-06-07 au 2024-02-22 pour NQ, avec MES/MNQ/M2K
depuis le 2019-05-06 en robustesse; B du 2015-01-02 au 2023-03-16. Aucun
résultat postérieur à ces bornes n'est chargé ou calculé.

## Résumé

1. Les seuils A sont parfaitement emboîtés, avec mêmes dates, prix, sorties et PnL.
2. Les analyses A utilisent donc quatre buckets exclusifs.
3. Sur NQ, `[0.25,0.50)` perd 54 667 $ net et `[0.50,0.75)` perd 48 353 $ net.
4. Les deux buckets supérieurs gagnent respectivement 18 479 $ et 1 138 $ net.
5. Le bucket prioritaire `[0.50,0.75)` compte 498 trades sur la nouvelle fenêtre globale.
6. Ses 10 pires trades expliquent 74,1 % de sa perte nette; les 20 pires, 121,2 %.
7. Sa perte 2021-2022 atteint 32 470 $; 2023 rebondit de 5 833 $.
8. Son coût relatif diminue fortement avec le niveau de NQ: la rupture vient du brut.
9. B perd 6 763 $ net: seulement -764 $ brut, contre 5 999 $ de coûts.
10. Pour B aussi, le coût normalisé baisse; le brut se dégrade surtout en 2019-2020.

## PnL global par configuration

| Signal | Actif | Bucket/configuration | Trades | Brut | Coûts | Net |
|---|---|---:|---:|---:|---:|---:|
| A | NQ | `[0.25,0.50)` | 914 | -43 242 | 11 425 | -54 667 |
| A | NQ | `[0.50,0.75)` | 498 | -42 128 | 6 225 | -48 353 |
| A | NQ | `[0.75,1.00)` | 260 | 21 729 | 3 250 | 18 479 |
| A | NQ | `>=1.00` | 282 | 4 663 | 3 525 | 1 138 |
| B | MGC | confirmed fade | 1 333 | -764 | 5 999 | -6 763 |

Les tableaux complets par année, trimestre, sortie, direction et instruments de
robustesse sont dans `step1_tables/`.

## Contrôle prioritaire NQ `[0.50,0.75)`

- Ancienne coupure au 2021-07-01: 400 trades, environ -35 941 $ net.
- Nouvelle borne validée au 2024-02-22: 498 trades, -48 353 $ net.
- Les 98 observations ajoutées contribuent donc environ -12 412 $.
- Dix pires trades: -35 835 $, soit 74,1 % de la perte nette.
- Pire trade/jour: 2022-01-24, -5 587,50 $ net.
- Drawdown principal: début 2010-06-11, creux 2023-02-02, profondeur
  -56 095,71 $; non récupéré sur l'IS.

Comme il n'existe qu'un trade par jour et bucket A, les dix pires jours sont les
dix pires trades.

## Brut normalisé et coûts

Pour NQ `[0.50,0.75)`, le coût moyen baisse d'environ 3,13 bp du notionnel en
2010 à 0,45 bp en 2023, et de 0,020 à 0,003 ATR14 journalier. Le brut normalisé
reste instable et devient particulièrement négatif en 2021-2022. La perte ne
vient donc pas d'une hausse du ratio coût/mouvement.

Pour B, le coût passe d'environ 3,93 bp en 2015 à 2,39 bp en 2023. Le brut est
quasi nul ou positif en 2017-2018 et 2021-2023, mais fortement négatif en
2019-2020. Les coûts expliquent l'essentiel de la perte cumulée, mais pas le
changement temporel de l'edge brut.

ATR14 désigne ici le Wilder ATR des barres quotidiennes de la session tradée,
calculé jusqu'à t-1. Pour B, les multiples MFE/MAE intraday emploient en plus
l'ATR causal du signal: médiane des true ranges des 20 barres de cinq minutes
précédentes de la même session, minimum 10.

## Données bloquées

A1, A2, B0b et B1 ne sont pas lancées. Voir `data_source_status.md`. B2 est figé
sur l'heure de la barre de climax: `08:20-09:30`, `09:30-11:00`, `11:00-13:25` ET.

