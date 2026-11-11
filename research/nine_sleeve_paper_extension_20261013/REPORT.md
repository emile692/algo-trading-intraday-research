# Book neuf sleeves — extension jusqu'au 2 octobre 2026

## Gouvernance des periodes

- Fin du ledger historique de recherche commun : **19 mars 2026**.
- Extension MNQ/MGC sans retuning : **20 mars au 2 octobre 2026**.
- Cette extension est un forward replay fige pour les six sleeves MNQ/MGC, mais pas un paper OOS vierge pour MBT : les trois ensembles MBT ont ete figes avec des donnees allant jusqu'au 2 octobre.
- Freeze officiel du book neuf sleeves : **4 octobre 2026**.
- Debut officiel du paper trading prospectif : **5 octobre 2026**, premiere seance apres le freeze. Aucun chiffre jusqu'au 2 octobre ne doit etre etiquete comme paper neuf-sleeves.

## Performance

| Periode | Debut | Fin | Seances | PnL net USD | Sharpe | Max DD USD | Pire jour USD |
|---|---|---|---:|---:|---:|---:|---:|
| Historique verrouille | 2024-01-02 | 2026-03-19 | 564 | 43,225.22 | 1.828 | -6,014.42 | -1,377.82 |
| Extension gelee | 2026-03-20 | 2026-10-02 | 141 | -4,534.82 | -1.093 | -8,963.04 | -1,058.96 |
| Total | 2024-01-02 | 2026-10-02 | 705 | 38,690.40 | 1.378 | -8,963.04 | -1,377.82 |

![Equity du book](figures/nine_sleeve_equity_through_20261002.png)

## Acquisition

Deux appels payants Databento ont complete MNQ.c.0 et MGC.v.0. Cout pre-requete total estime : **0,739163 USD**. La cle API n'est stockee dans aucun fichier.
Databento marque quatre dates UTC `degraded`, dont une seule seance de semaine (`2026-04-10`). MNQ et MGC ont ce jour-la leurs 390 et 310 minutes attendues dans les fenetres de recherche; la seance est conservee et documentee, pas supprimee silencieusement. Les ledgers historiques verrouilles n'ont pas ete reecrits.
