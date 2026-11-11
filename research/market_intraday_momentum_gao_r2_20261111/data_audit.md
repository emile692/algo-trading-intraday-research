# R2 — Market Intraday Momentum : audit des données

Date : 2026-11-11. Statut : **audit terminé, aucun backtest lancé**.

Script : `audit_data.py` (sorties dans `audit/`). Il ne calcule que des comptages, des volumes et des amplitudes non signées. Aucun rendement signé par le signal, aucun croisement première demi-heure / dernière demi-heure n'a été calculé.

## 1. Fichiers

Barres d'une minute, horodatées au début de la barre, converties en `America/New_York` (zone IANA, donc changements d'heure suivis). Aucun doublon d'horodatage, aucun prix négatif ou nul.

| Instrument | Fichier | Début | Fin | sha256 (8 premiers) |
|---|---|---|---|---|
| MNQ | `data/processed/parquet/MNQ_c_0_1m_20190505_20261003.parquet` | 2019-05-05 | 2026-10-02 | 72a78506 |
| MES | `data/processed/parquet/MES_c_0_1m_20260322_135702.parquet` | 2019-05-05 | 2026-03-20 | a2381ba2 |
| MGC | `data/processed/parquet/MGC_v_0_1m_20150101_20261003.parquet` | 2015-01-01 | 2026-10-02 | 98ed4a18 |
| MCL | `data/downloaded/MCL_v_0_1m_20210712_20261111.parquet` | 2021-07-12 | 2026-11-10 | 5073f06d |
| NQ (benchmark) | `data/processed/parquet/NQ_c_0_1m_20260321_095036.parquet` | 2010-06-06 | 2026-03-20 | a8c85d03 |
| CL (proxy MCL avant 2021) | `data/downloaded/CL_v_0_1m_20150101_20210712.parquet` | 2015-01-01 | 2021-07-11 | 6fe3bd6a |

**Absents du disque : ES et SPY.** Le benchmark académique disponible est NQ 2010–2019, qui recouvre la fin de l'échantillon du papier (2010-06 → 2013-12). Un benchmark ES demanderait un téléchargement Databento payant.

## 2. Peut-on reconstruire le signal ?

Oui sur MNQ, MES et NQ, sans convention de remplacement.

| Prix | Barre utilisée | Disponibilité MNQ / MES / NQ |
|---|---|---|
| Clôture précédente | close de la barre 15:59 de la dernière séance cash | 100 % des séances complètes |
| Prix à 10:00 | close de la barre 09:59 (dernier prix avant 10:00:00) | 100 % |
| Entrée 15:30 | open de la barre 15:30 | 100 % |
| Sortie 16:00 | close de la barre 15:59 | 100 % |

Sur MGC et MCL les barres exactes manquent parfois (marché peu actif à ces heures). La convention proposée est une tolérance de 5 minutes : première barre dans [nœud, nœud + 4] pour une entrée, dernière barre dans [nœud − 5, nœud − 1] pour un prix de signal ou de sortie.

| Instrument, horloge | Barre d'entrée exacte | Volume médian minute d'entrée | 5e centile | Remarque |
|---|---|---|---|---|
| MNQ 15:30 | 100 % | 1 594 lots | 222 | exécutable |
| MES 15:30 | 100 % | 1 650 lots | 262 | exécutable |
| NQ 15:30 | 100 % | 690 lots | 124 | exécutable |
| MGC 15:30, 2020+ | ≥ 99 % | 20–200 lots | 3–80 | exécutable pour 1 lot |
| MGC 15:30, 2015–2018 | 34–59 % | 0–1 lot | 0 | **non exécutable, période écartée** |
| MCL 15:30 | 95–100 % | 20 lots | 2 | exécutable pour 1 lot, fragile |
| MCL 14:00 (horloge native) | 98–100 % | 73 lots | 9 | exécutable pour 1 lot |

## 3. Séances atypiques

Classification faite sur les barres elles-mêmes (aucun paquet de calendrier n'est installé dans l'environnement).

| Type | Critère | MNQ | MES | NQ | Traitement proposé |
|---|---|---|---|---|---|
| Séance complète | les quatre nœuds présents | 1 817 | 1 684 | 3 866 | tradée |
| Clôture anticipée (13:00 cash) | dernière barre 13:14 ; 3 juillet, lendemain de Thanksgiving, 24 décembre | 15 | 15 | 32 | non tradée ; son close de 12:59 sert de clôture précédente à la séance suivante |
| Jour férié avec Globex | dernière barre 12:59 | 50 | 46 | 75 | non tradé ; pas de séance cash, donc pas de clôture : la clôture précédente reste celle de la dernière séance cash |
| Pas de séance jour | aucune barre 09:30–16:00 | 44 | 41 | 98 | non tradé (voir roll ci-dessous) |
| Trou de données | séance interrompue | 2 | 2 | 20 | non tradé (2020-02-28 et 2020-06-30 sur les trois fichiers ; NQ 2010–2012 jours fériés tronqués à 11:29) |

Clôture précédente : dernière séance cash, à 5 jours calendaires au plus (week-end plus un férié). Au-delà, la séance est exclue.

Séances éligibles : **MNQ 1 786, MES 1 656, NQ 3 804, MGC 2 572, MCL 1 226** (horloge 09:30–16:00).

## 4. Roll des contrats continus — limite importante

Les fichiers `c.0` suivent le contrat de tête jusqu'à son expiration. Conséquences mesurées sur MNQ :

- Le jour d'expiration trimestrielle (troisième vendredi), le contrat de tête s'arrête à 09:30 : **pas de séance ce jour-là**. Ce sont les « quadruple witching », justement des jours de flux de clôture importants. Environ 4 séances par an perdues.
- Le lundi suivant, la clôture précédente appartient à un autre contrat : séance exclue (règle : même `instrument_id` à la clôture précédente, au signal, à l'entrée et à la sortie). Environ 4 séances par an.
- La semaine avant l'expiration, la liquidité a déjà migré vers le contrat suivant (volume de la première demi-heure : 145 000 → 15 000 lots entre le 8 et le 14 décembre 2023). Les prix restent ceux d'un contrat réel et négociable, mais un trader réel serait sur le contrat suivant. Le filtre de volume de la campagne C devra en tenir compte.

MGC et MCL (`v.0`, roll sur volume) : 65 à 86 séances exclues pour roll sur tout l'historique.

## 5. Coût rapporté à l'amplitude — avertissement avant test

Modèle de coûts du repository : 1 tick de slippage par côté + commission par côté. Amplitude = moyenne de |prix sortie / prix entrée − 1| sur la fenêtre de 30 minutes, période IS, non signée.

| Instrument, horloge | Coût aller-retour | Coût (bp) | Amplitude moyenne (bp) | Coût / amplitude | Taux de réussite d'équilibre approx. |
|---|---|---|---|---|---|
| NQ 15:30–16:00 | 12,50 USD | 0,9 | 19,6 | 0,02 | 51 % |
| MNQ 15:30–16:00 | 3,50 USD | 1,2 | 22,8 | 0,06 | 53 % |
| MES 15:30–16:00 | 5,00 USD | 2,3 | 20,3 | 0,12 | 56 % |
| MCL 14:00–14:30 | 4,50 USD | 5,8 | 33,2 | 0,15 | 57 % |
| MGC 13:00–13:30 | 4,50 USD | 2,5 | 9,5 | 0,29 | 64 % |
| MGC 15:30–16:00 | 4,50 USD | 2,5 | 7,8 | 0,34 | 67 % |
| MCL 15:30–16:00 | 4,50 USD | 5,8 | 15,2 | 0,37 | 69 % |

Le taux d'équilibre suppose que la réussite ne dépend pas de l'amplitude (approximation). Le papier rapporte un pouvoir prédictif faible (R² de l'ordre de 1 à 3 %), compatible avec un taux de réussite de quelques points au-dessus de 50 %. Lecture a priori : **MNQ est le seul contrat micro où le coût laisse une marge ; MES est à la limite ; MGC et MCL à 15:30–16:00 ne peuvent pas être rentables sauf effet très supérieur à celui du papier.** Ils restent testés pour la valeur de réplication, pas comme candidats.

## 6. Autres données

- **Book gelé** : `research/locked_eight_sleeve_portfolio_20261103/daily_pnl.csv`, 564 séances, 2024-01-02 → 2026-03-19, soit exactement la fenêtre OOS. L'analyse de diversification ne peut donc se faire que sur OOS.
- **VIX** : `data/raw/vix-cboe.csv`, quotidien jusqu'au 2026-10-07. Utilisable pour le régime de volatilité (clôture de la veille).
- **Calendrier macro** : `data/raw/economic_calendar_us.csv` contient 10 événements de 2026. **Inutilisable** pour un test historique.
