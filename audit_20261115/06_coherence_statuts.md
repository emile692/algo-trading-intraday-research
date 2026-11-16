# Phase 6 — Cohérence des statuts S04 / S09

*Mêmes règles pour les 9 sleeves, avec les seules données disponibles au 2026-11-01. Script : [phase6_status.py](phase6_status.py). Table : [06_status.csv](outputs/06_status.csv). Rien n'est changé en production.*

## Données et règle appliquée

- **S01** : historique gelé ATR 18 jusqu'au 2026-03-19, plus la fenêtre du one-shot (O), du 2026-03-23 au 2026-10-02.
- **S02 à S06** : replay forward des votes gelés jusqu'au 2026-10-02 (identique à l'historique gelé sur le recouvrement, vérifié).
- **S07 à S09** : historique gelé jusqu'au 2026-10-02.

Règle proposée en phase 5, appliquée en glissant sur toute la série :

- **shadow** si R0 ou R2 s'est déclenchée dans les 252 dernières séances, puisqu'aucune réintégration n'est encore possible ;
- **revue** si R3 s'est déclenchée dans les 120 dernières séances, ou si la classe de la phase 2 sur la validation réutilisée est « négative » ;
- **actif** sinon.

| Sleeve | Données jusqu'au | R0 : 126 s. final / seuil | R3 : 120 s. final / seuil | maxDD 250 s. final / seuil R2 | Séances en alarme R0 · R1 · R2 · R3 | Classe total / validation | **Statut règle** | Statut actuel | Motifs |
|---|---|---|---|---|---|---|---|---|---|
| S01 | 2026-10-02 | −390,00 / −890,50 | −449,00 / −639,52 | −840,50 / −2 806,05 | 0 · 0 · 0 · 8 | distinguable de zéro / distinguable de zéro | **actif** | actif | — |
| S02 | 2026-10-02 | −949,50 / −4 019,00 | −1 413,50 / −2 177,57 | −3 065,50 / −6 595,51 | 0 · 0 · 0 · 110 | distinguable de zéro / indistinguable | **actif** | actif | — |
| S03 | 2026-10-02 | 281,50 / −975,00 | 116,00 / −591,05 | −852,50 / −1 473,03 | 0 · 0 · 0 · 109 | indistinguable / indistinguable | **actif** | actif | — |
| S04 | 2026-10-02 | −6 574,00 / −2 974,00 | −6 283,00 / −2 497,03 | −8 174,50 / −6 281,53 | 36 · 55 · 17 · 54 | indistinguable / indistinguable | **shadow** | shadow | R0 du 2026-07-17 au 2026-10-02; R2 du 2026-09-09 au 2026-10-02; R3 du 2026-07-01 au 2026-10-02 |
| S05 | 2026-10-02 | 700,50 / −1 021,00 | 700,50 / −982,02 | −1 841,00 / −2 591,51 | 22 · 0 · 0 · 23 | indistinguable / indistinguable | **shadow** | actif | R0 du 2026-07-28 au 2026-08-25; R3 du 2026-07-20 au 2026-08-18 |
| S06 | 2026-10-02 | 995,00 / −1 006,50 | 995,00 / −924,02 | −909,50 / −2 308,12 | 0 · 0 · 0 · 1 | indistinguable / indistinguable | **actif** | actif | — |
| S07 | 2026-10-02 | −199,60 / −840,64 | −216,46 / −809,70 | −896,24 / −2 144,58 | 0 · 0 · 0 · 3 | indistinguable / indistinguable | **revue** | actif | R3 du 2026-08-18 au 2026-08-18 |
| S08 | 2026-10-02 | −739,78 / −1 284,10 | −750,68 / −1 149,57 | −1 653,28 / −3 009,66 | 0 · 0 · 0 · 7 | indistinguable / indistinguable | **revue** | actif | R3 du 2026-06-01 au 2026-08-12 |
| S09 | 2026-10-02 | −641,96 / −1 479,42 | −605,50 / −717,72 | −1 404,50 / −2 239,69 | 0 · 144 · 0 · 76 | distinguable de zéro / négative | **revue** | actif | R3 du 2026-06-16 au 2026-10-01; classe « négative » sur la validation réutilisée |

## Incohérences avec les statuts actuels

- **S05** : la règle donne « shadow », le statut actuel est « actif ». Motifs : R0 du 2026-07-28 au 2026-08-25; R3 du 2026-07-20 au 2026-08-18.
- **S07** est active, alors que la règle la placerait « en revue » : R3 du 2026-08-18 au 2026-08-18.
- **S08** est active, alors que la règle la placerait « en revue » : R3 du 2026-06-01 au 2026-08-12.
- **S09** est active, alors que la règle la placerait « en revue » : R3 du 2026-06-16 au 2026-10-01; classe « négative » sur la validation réutilisée.

## Avec la proposition corrigée du 2026-11-15

La phase 5 a été corrigée : faire passer en shadow sur « R0 ou R2 » coupe à tort 21 % des sleeves saines, contre 7 % pour R2 seule. La proposition corrigée réserve le shadow à R2 ; R0 et R3 deviennent des alertes « revue » (déclenchées dans les 120 dernières séances).

| Sleeve | Règle initiale (R0 ou R2 → shadow) | Proposition corrigée (R2 → shadow) | Statut actuel |
|---|---|---|---|
| S01 | actif | **actif** | actif |
| S02 | actif | **actif** | actif |
| S03 | actif | **actif** | actif |
| S04 | shadow | **shadow** | shadow |
| S05 | shadow | **revue** | actif |
| S06 | actif | **actif** | actif |
| S07 | revue | **revue** | actif |
| S08 | revue | **revue** | actif |
| S09 | revue | **revue** | actif |

Avec la proposition corrigée, aucune sleeve active ne passe en shadow. S04 y reste, puisque R2 s'est déclenchée pour elle. S05 passe « en revue » au lieu de « shadow ».

## S04 et S09 : deux poids, deux mesures

- **La décision du 03/11 sur S04 s'appuie sur un leave-one-out calculé sur la fenêtre réutilisée** 2024-01-02 → 2026-03-19 (validation réutilisée). Le book à 9 sleeves y passe d'un Sharpe de 1,539 à 1,751 sans S04, chiffres du lock publié. Après correction (S01 ATR 18, calendrier union), il passe de 1,509 à 1,712. Sur cette même fenêtre, S04 gagnait pourtant 5 383,00 USD : le leave-one-out mesure une baisse de risque, pas une perte.
- Le déclencheur réel a été le replay du 2026-03-20 au 2026-10-02, où S04 perd −6 363,50 USD. Cette fenêtre a ensuite été déclarée contaminée : la décision ne peut plus être validée que par le forward.
- **S09** a une validation réutilisée nulle (−73,22 USD, Sharpe −0,040). R1 l'aurait signalée dès le 2025-08-13 (phase 5), et elle reste active. Avec la règle uniforme, S09 passe au moins « en revue ».
- **Asymétrie de données.** Les seuils R0 sont le minimum de leur propre historique. Pour S07 à S09, cet historique va jusqu'au 2026-10-02 : R0 ne peut donc pas s'être déclenchée sur leurs données. Pour S01 à S06, elle peut se déclencher sur la fenêtre 2026-03-20 → 2026-10-02, absente de leur calibrage. À règle égale, les sleeves MBT ne peuvent pas être prises en défaut par R0.
- **S01** : le one-shot (O) est COHÉRENT (−390,00 USD sur 137 séances, seuil −617,50). Mais il ne contient que 11 trades : la sleeve s'assèche (phase 4).

Rien n'est modifié. Ces statuts sont une lecture de cohérence, à décider par vous.
