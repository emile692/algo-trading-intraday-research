# Agrégation et couche de risque du book à 8 sleeves — clôture

**Statut :** `ADOPT_AGGREGATION_ONLY`.
- Le netting par instrument (couche A) est adopté, avec le sizing actuel (B0 : 1 contrat par sleeve, S01 à son sizing natif de 1 à 4 contrats).
- Toutes les options de la couche de risque (B1, B2, B3) sont **rejetées**.

**Dates :** spec gelée le 2026-11-12, sélection IS le 2026-11-12, OOS lu une fois le 2026-11-13.

**Spec :** [FROZEN_SPECIFICATION.json](FROZEN_SPECIFICATION.json), sha256 `79dd2257…` ([SPEC_LOCK.json](SPEC_LOCK.json)). Les self-tests sont passés avant tout calcul de performance ([selftest_report.json](results/selftest_report.json)).

**Lecture unique de l'OOS 2024-01 → 2026-03 :** seulement pour la configuration retenue et B0. Les autres configurations, dont B2, n'ont pas été lues sur l'OOS. Cette fenêtre était déjà contaminée pour cette couche (voir la spec).

## Ce qui a été testé

- **Couche A :** position nette = somme des contributions des sleeves. Les ordres en delta partent au fill natif de chaque sleeve. Les SL/TP de S01 deviennent des ordres posés sur le compte.
- **Couche B :** testée sur le cycle Topstep 50K complet (Combine → XFA → payout), avec 8 000 chemins en bootstrap par blocs et un horizon de 126 séances. Les règles sont celles fournies le 2026-11-12, dont le Scaling Plan XFA (2, 3 puis 5 lots) et 1 MBT = 1 lot. Les options :
  - **B1** : budget ES statique, k ∈ {0,30 ; 0,40 ; 0,50} ;
  - **B2** : sizing modulé par la vol de l'instrument ;
  - **plafond MBT** à 1 ou 2 contrats ;
  - **stop journalier par instrument** à q ∈ {1,0 ; 1,5} × ES ;
  - **stop journalier du book** à 1 000 USD ;
  - **verrou de gain** de consistance.
- **Sélection séquentielle sur l'IS commune** (2021-06-22 → 2023-12-29, 653 séances). Une option n'est gardée que si elle apporte au moins +10 % de cash net, gagne sur au moins 55 % des chemins appariés, et ne relève pas la probabilité de perdre le XFA de plus de 2 points.

## Résultats

Cash net = 90 % des payouts − frais, sur 126 séances, en USD par compte.

| Configuration | IS : cash espéré | IS : cash médian | IS : P(payout) | IS : P(XFA perdu) | Décision |
|---|---:|---:|---:|---:|---|
| B0 (témoin) | 548 | −83 | 80 % | 72 % | — |
| B1, k = 0,30 / 0,40 / 0,50 | 486 | −112 | 78 % | 67 % | inerte : aucun λ admissible, donc 1 contrat partout |
| + B2 (vol de l'instrument) | **728** | +36 | 82 % | 74 % | rejeté : P(XFA perdu) +7,2 points |
| + plafond MBT 1 / 2 | 445 / 455 | | | 64 % / 66 % | rejeté |
| + stop instrument 1,0 / 1,5 ES | 356 / 472 | | | | rejeté |
| + stop book 1 000 USD | 482 | | | | rejeté |
| + verrou de gain | 486 | | | | inerte : meilleure journée IS 1 201 USD, sous le seuil de 1 600 |

- **CSCV :** PBO 0,63, au-dessus de la limite de 0,5. La spec impose alors de revenir à B1 k = 0,40, c'est-à-dire 1 contrat partout, avec S01 à 1 contrat.
- **OOS (572 séances), configuration retenue contre B0 :**

| | Retenue | B0 |
|---|---:|---:|
| Cash espéré | 866 | 1 015 |
| Cash médian | −168 | −70 |
| Chemins où la retenue fait mieux | 40 % | |
| P(XFA perdu) | 72,9 % | 75,5 % |
| Total historique | 28 958 | 31 180 |

  La retenue ne bat pas B0, d'où le verdict `ADOPT_AGGREGATION_ONLY`.
- **Sensibilités, non sélectionnées :**
  - le parcours Standard bat No Activation Fee (486 contre 401 USD) ;
  - compter MBT à 0,1 lot rapporte +77 USD ;
  - sur un 100K, le Combine seul passe 48 % du temps, contre 90 % sur un 50K.

## Lecture

1. **Le netting ne coûte rien et se met en place tel quel.** Le PnL brut est identique au dixième de cent près sur 1 203 séances IS, et le netting économise 41 USD de coûts. Les 562 trades IS de S01 sont reproduits exactement.
2. **Un budget de risque par sleeve est infaisable sur un 50K avec 1 MBT = 1 lot.** Une sleeve MBT est 2,9 à 4 fois moins risquée par contrat que S02. Dès λ = 1, elle recevrait donc 3 ou 4 contrats, ce qui dépasse les 5 lots du Combine. C'était prévisible à partir des règles. C'est un défaut de conception de B1, pas une information sur les signaux.
3. **Retirer le sizing natif de S01 coûte de l'argent.** Sur l'IS, S01 rapporte 12 209 USD à son sizing natif contre 10 023 à 1 contrat. C'est la principale raison pour laquelle le point de départ fait moins bien que B0, sur l'IS comme sur l'OOS.
4. **Les stops et les plafonds coupent plus de gains qu'ils n'évitent de pertes.**
5. **B2 est le seul effet notable** (+50 % de cash espéré sur l'IS). Il augmente cependant le risque de perdre le XFA, parce qu'il monte aussi la taille quand la vol est basse (pic à 4,4 lots). Diagnostic IS : pour 5 sleeves sur 8 (S01, S02, S05, S07, S08), le tercile de vol haute est le plus faible.
6. **Même la meilleure configuration perd de l'argent sur plus de la moitié des chemins.** Environ deux XFA sur trois sont perdus en 126 séances, quelle que soit la configuration.

## Défaut trouvé dans le lock `eight_sleeve_equal_contract_reference_20261103`

[relock_eight_sleeves.py](../locked_eight_sleeve_portfolio_20261103/relock_eight_sleeves.py) aligne le PnL des sleeves MGC et MBT sur le calendrier MNQ. Ce calendrier exclut des séances que MGC et MBT tradent, notamment les vendredis de roll trimestriel. Sur 2024-01-02 → 2026-03-19, 8 séances sont écartées, pour **−509 USD net** :

| Sleeve | Somme des trades | Dans le lock |
|---|---:|---:|
| S05 | 1 717,5 | 1 788 |
| S06 | 3 932,5 | 4 098 |
| S07 | 2 209 | 2 355 |
| S08 | 4 135 | 4 345 |
| S09 | 2 957 | 2 875 |

Le book du lock fait donc 31 661 USD, contre 31 152 pour la somme réelle des trades. Les estimations Topstep du lock reposent sur ces 564 lignes. Le lock n'a pas été modifié : la gouvernance interdit d'écraser une référence. Une correction passe par un nouveau lock versionné.

## Incident pendant la campagne

Le premier run IS contenait un bug dans le remplissage des records du verrou de gain : le verrou affichait 0 % de PASS. Il a été corrigé, deux tests ont été ajoutés, et le self-test puis la sélection ont été relancés avant toute lecture de l'OOS. Seules les configurations avec verrou étaient touchées. Le run est archivé dans [results/is_run1_profit_lock_padding_bug](results/is_run1_profit_lock_padding_bug/README.md).

## Réouverture

- **Ne pas reprendre** B1, les stops ou les plafonds avec d'autres seuils.
- **Une modulation qui ne fait que baisser la taille quand la vol monte** (B2 sans hausse en vol basse) peut faire l'objet d'une **nouvelle campagne pré-enregistrée**. Il faudra y déclarer que l'IS de B2 a été vu. L'OOS de B2 n'a pas été lu. Le verdict propre reste le forward après le 2026-11-03.
- Le sizing natif de S01 n'a pas d'équivalent pour les autres sleeves. L'étendre serait une nouvelle hypothèse.
