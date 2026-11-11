# Audit documentaire — étape 0 : inventaire et reproduction

Date de l'audit : **27 octobre 2026**. Document audité : `00_portefeuille_9_sleeves_complet.pdf`, édition du 13 octobre 2026.

## 1. Périmètre et statut

Cette étape est strictement descriptive. Aucun signal, paramètre, lock, sizing, vote ni élément de composition du book n'a été modifié. Les sources client et les PDF n'ont pas été modifiés ni régénérés. Les calculs de l'audit sont isolés dans `research/signal_sheets_audit_20261014/`.

Configuration écrite avant calcul : `step0_config.json`. Reproduction :

```powershell
uv run --cache-dir .tmp\uv-cache python scripts\audit_signal_sheets_step0.py
```

## 2. Chaîne actuelle de génération du PDF

1. Les neuf fiches sont les fichiers `docs/signal_sheets/01_*.md` à `09_*.md`; le book est `docs/signal_sheets/BOOK.md`.
2. `scripts/build_indicator_illustrations.py` génère les illustrations pédagogiques des indicateurs.
3. `scripts/build_signal_sheet_heatmaps.py` génère les heatmaps et les graphiques de book. Il lit des artefacts de recherche hétérogènes et rejoue le moteur pour S03. Depuis l'ajout FS/OOS, il ne doit plus écraser les equity curves individuelles.
4. `scripts/build_equity_curves_fs_oos.py` rejoue les locks MNQ/MGC, lit les exports ORB/MBT et génère les deux panneaux FS et OOS/validation.
5. `scripts/export_signal_sheets_pdf.py` transforme les Markdown en dix HTML, embarque Mermaid localement et assemble le recueil complet.
6. Le passage HTML → PDF est réalisé avec Microsoft Edge headless. Il n'existe pas encore de pilote de rendu PDF versionné qui encapsule cette dernière commande.

Sources de performance principales :

- S01 : `export/mnq_orb_vwap_atr_retained/trades.csv` et parquet MNQ gelé ;
- S02–S04 : configuration `locked_four_sleeve_portfolio` et moteur baseline gelé ;
- S05–S06 : specs de `mgc_final_two_sleeve_lock_20260926` et moteur baseline gelé ;
- S07–S09 : exports exécutables de `mbt_three_sleeve_freeze_20261004/vote_ensemble` ;
- book : `mbt_nine_sleeve_topstep_20261004/daily_nine_sleeves_nominal.csv`, borné au 2 janvier 2024–19 mars 2026.

## 3. Définitions contrôlées

La définition cible demandée est : **PnL net quotidien, jours sans trade inclus, écart-type échantillonnal (`ddof=1`), annualisation √252**. La colonne « Sharpe canonique » ci-dessous applique exactement cette définition.

Le PDF actuel mélange trois conventions :

- S01 : PnL quotidien divisé par un capital statique, `ddof=0`, √252 ;
- S02–S06 : rendement quotidien calculé sur l'equity précédente, `ddof=1`, √252 ;
- S07–S09 et book : PnL net quotidien brut, `ddof=1`, √252.

Les différences S02–S06 ne viennent donc pas d'un autre PnL, mais du dénominateur variable utilisé avant le calcul du Sharpe.

## 4. Reproduction par sleeve et période

Tous les PnL sont nets de coûts nominaux. « Sharpe source » reproduit la convention historique de l'artefact ; « Sharpe canonique » applique la définition cible.

| Sleeve | Période | Séances | Trades | PnL net | Sharpe source | Sharpe canonique | Max DD |
|---|---|---:|---:|---:|---:|---:|---:|
| S01 | IS | 1 500 | 584 | 12 524,50 | 0,962 | 0,962 | −2 061,50 |
| S01 | OOS | 643 | 239 | 6 013,00 | 1,153 | 1,152 | −928,50 |
| S01 | Total | 2 143 | 823 | 18 537,50 | 1,016 | 1,016 | −2 061,50 |
| S02 | IS | 1 186 | 608 | 7 386,00 | 0,683 | 0,687 | −4 655,50 |
| S02 | Validation | 564 | 291 | 11 211,00 | 1,263 | 1,227 | −3 554,00 |
| S02 | Total | 1 750 | 899 | 18 597,00 | 0,903 | 0,897 | −4 655,50 |
| S03 | IS | 1 186 | 221 | 1 539,00 | 0,495 | 0,494 | −1 190,00 |
| S03 | Validation | 564 | 110 | 1 078,50 | 0,508 | 0,503 | −730,00 |
| S03 | Total | 1 750 | 331 | 2 617,50 | 0,493 | 0,489 | −1 190,00 |
| S04 | IS | 1 186 | 1 460 | 4 103,50 | 0,339 | 0,310 | −3 574,00 |
| S04 | Validation | 564 | 659 | 8 274,00 | 0,830 | 0,800 | −4 056,00 |
| S04 | Total | 1 750 | 2 119 | 12 377,50 | 0,532 | 0,509 | −4 056,00 |
| S05 | IS | 1 203 | 134 | 1 167,00 | 0,431 | 0,428 | −1 148,00 |
| S05 | Validation réutilisée | 572 | 65 | 2 073,50 | 0,485 | 0,470 | −1 481,50 |
| S05 | Total | 1 775 | 199 | 3 240,50 | 0,394 | 0,384 | −1 481,50 |
| S06 | IS | 1 203 | 141 | 3 087,50 | 0,872 | 0,861 | −968,00 |
| S06 | Validation réutilisée | 572 | 95 | 4 516,50 | 0,864 | 0,844 | −1 320,00 |
| S06 | Total | 1 775 | 236 | 7 604,00 | 0,757 | 0,733 | −1 320,00 |
| S07 | Legacy IS, couverture exécutable partielle | 320 | 145 | 1 102,10 | 0,753 | 0,753 | −1 285,80 |
| S07 | Legacy OOS réutilisé | 326 | 143 | 891,24 | 0,595 | 0,595 | −952,24 |
| S07 | Total exécutable | 646 | 288 | 1 993,34 | 0,673 | 0,673 | −1 285,80 |
| S08 | Legacy IS, couverture exécutable partielle | 320 | 331 | 2 100,58 | 0,954 | 0,954 | −1 542,56 |
| S08 | Legacy OOS réutilisé | 326 | 362 | 1 364,66 | 0,687 | 0,687 | −1 653,28 |
| S08 | Total exécutable | 646 | 693 | 3 465,24 | 0,827 | 0,827 | −1 653,28 |
| S09 | Legacy IS, couverture exécutable partielle | 320 | 221 | 2 317,28 | 1,083 | 1,083 | −984,84 |
| S09 | Legacy OOS réutilisé | 326 | 221 | −73,22 | −0,040 | −0,040 | −1 525,38 |
| S09 | Total exécutable | 646 | 442 | 2 244,06 | 0,565 | 0,565 | −1 916,40 |

### Limite MBT constatée

Les heatmaps MBT couvrent un legacy IS commençant le 3 mai 2021, mais l'export de l'ensemble **exécutable** commence le 3 janvier 2024. Les lignes legacy IS ci-dessus ne couvrent donc que le 3 janvier 2024–28 mai 2025. Il n'existe pas dans l'export actuel de ledger de trades de l'ensemble exécutable pour le 3 mai 2021–2 janvier 2024. Les Sharpes des heatmaps 2021–2025 sont des Sharpes **par cellule**, pas ceux de l'ensemble exécutable. Un nombre de trades de l'ensemble sur le legacy IS complet ne peut pas être produit à partir des artefacts présents sans reconstruire un nouvel export historique.

## 5. Reproduction du book

Fenêtre figée : **2 janvier 2024–19 mars 2026**, 564 séances. Les données postérieures sont exclues par la configuration et par le script.

| Mesure | PDF | Reproduction | Écart |
|---|---:|---:|---:|
| PnL net | 43 225,22 USD | 43 225,22 USD | 0,00 |
| Volatilité quotidienne | 665,46 USD | 665,46 USD | 0,00 après arrondi |
| Pire jour | −1 377,82 USD | −1 377,82 USD | 0,00 |
| Max drawdown | −6 014,42 USD | −6 014,42 USD | 0,00 |
| Sharpe canonique | 1,828 | 1,828266 | arrondi conforme |

### Contributions au risque

Formule reproduite : `w_i × (Σw)_i / (w'Σw)`, avec `w=[1,…,1]` sur les neuf colonnes de PnL. Les contributions totalisent 100 %.

| Sleeve | PDF | Reproduction |
|---|---:|---:|
| S01 | 5,8 % | 5,781 % |
| S02 | 28,9 % | 28,887 % |
| S03 | 3,5 % | 3,506 % |
| S04 | 32,9 % | 32,889 % |
| S05 | 5,8 % | 5,772 % |
| S06 | 7,2 % | 7,219 % |
| S07 | 3,5 % | 3,546 % |
| S08 | 5,7 % | 5,677 % |
| S09 | 6,7 % | 6,724 % |

### Leave-one-out

| Sleeve retirée | PnL restant | Sharpe | Max DD | Δ PnL | Δ Max DD |
|---|---:|---:|---:|---:|---:|
| S01 | 36 896,22 | 1,616 | −6 020,80 | −6 329,00 | −6,38 |
| S02 | 32 014,22 | 1,791 | −5 568,60 | −11 211,00 | +445,82 |
| S03 | 42 146,72 | 1,841 | −6 001,92 | −1 078,50 | +12,50 |
| S04 | 34 951,22 | 2,024 | −4 466,12 | −8 274,00 | +1 548,30 |
| S05 | 41 082,22 | 1,813 | −5 957,92 | −2 143,00 | +56,50 |
| S06 | 38 554,22 | 1,713 | −5 673,18 | −4 671,00 | +341,24 |
| S07 | 40 926,18 | 1,784 | −6 316,64 | −2 299,04 | −302,22 |
| S08 | 38 880,12 | 1,722 | −6 305,32 | −4 345,10 | −290,90 |
| S09 | 40 350,64 | 1,811 | −5 668,74 | −2 874,58 | +345,68 |

Tous les chiffres du book, des contributions au risque et du LOO sont reproduits au centime ou à l'arrondi affiché.

## 6. Audit des heatmaps et du cube

Le détail machine lisible est dans `heatmap_metric_audit.csv`.

- **S01** : la couleur est réellement `is_sharpe_ratio`. Le cube varie de 1,153 à 2,342 et les six points retenus de 1,291 à 1,612. Ces valeurs sont celles de stratégies à **un point ATR** évaluées séparément, alors que le 0,962 de la fiche est celui de la sleeve finale à vote de six filtres. L'incompatibilité visuelle vient donc d'un changement d'objet, auquel s'ajoute `ddof=0`; ce n'est pas une fausse valeur de couleur.
- **S02** : la cellule 240 minutes vaut 0,683 en IS et 1,263 en validation. Le maximum IS est à 300 minutes (0,833), puis 270 minutes (0,827) ; le maximum validation est à 240 minutes.
- **S03** : la cellule active NR10 / 1 tick vaut 0,495 en IS et 0,508 en validation selon la convention historique du moteur.
- **S04** : le stop 1 ATR vaut 0,339 en IS et 0,830 en validation selon cette même convention.
- **S05** : le point 30 / 1,10 vaut 0,431 en IS et 0,484 en validation.
- **S06** : le point final 0,50 / 0,60 vaut 0,872 / 0,863 (rapport arrondi 0,872 / 0,864). Le médoïde 0,50 / 0,50 vaut 0,805 / 0,621.
- **S07–S09** : les heatmaps utilisent bien le Sharpe canonique sur PnL quotidien, mais au niveau des cellules et non de l'ensemble exécutable.

Conclusion : seules les heatmaps MBT utilisent déjà partout la définition cible. S01 nécessite une clarification de l'objet visualisé ou un recalcul de l'ensemble ; S02–S06 nécessitent un recalcul en PnL quotidien brut pour harmonisation.

## 7. Écarts et incohérences constatés

1. **Aucun écart de ledger** : tous les nombres de trades, PnL et max drawdowns imprimés qui ont une valeur chiffrée dans le PDF sont reproduits.
2. **Aucun écart du book** : métriques de tête, risque et LOO sont reproduits.
3. **Sharpe non harmonisé** : les nombres imprimés se reproduisent sous leurs conventions historiques, mais S02–S06 diffèrent de la définition canonique demandée. Les écarts les plus visibles sur le total sont S02 `0,903 → 0,897`, S03 `0,493 → 0,489` et S04 `0,532 → 0,509`.
4. **Cube S01 non comparable au chiffre de fiche** : cellules individuelles contre ensemble exécutable.
5. **Couverture MBT incomplète pour l'ensemble exécutable** : impossible de donner les quatre métriques de l'ensemble sur le legacy IS complet 2021–2025 avec les exports présents.
6. **S05 et S06** : la section 9 actuelle décrit les périodes sans afficher trades, PnL, Sharpe et max DD, alors que le recalcul fournit ces valeurs.
7. **Sizing de couverture** : `Equal-contract [1,…,1]` est factuellement inexact pour S01, dont la quantité historique est calculée à partir d'un budget de risque de 250 USD.
8. **Statut du book** : les paragraphes expliquent que la fenêtre n'est pas un nouvel OOS, mais cette qualification n'est pas répétée dans tous les titres et graphiques.
9. **Version documentaire** : `docs/signal_sheets/README.md` indique encore « 11 octobre 2026 », tandis que le PDF porte l'édition du 13 octobre 2026.
10. **Dernière étape de build non encapsulée** : le rendu Edge headless est manuel, donc la chaîne n'est pas entièrement portée par un script unique.

## 8. Artefacts de l'étape 0

- `step0_config.json` : fenêtres et définition canonique écrites avant calcul ;
- `sleeve_period_reproduction.csv` : métriques par sleeve et période ;
- `document_claim_comparison.csv` : comparaison avec les valeurs imprimées ;
- `book_reproduction.csv`, `book_risk_reproduction.csv`, `book_leave_one_out_reproduction.csv` ;
- `heatmap_metric_audit.csv` ;
- `number_traceability.csv` : chiffre, section, métrique, fichier source et fonction ;
- `scripts/audit_signal_sheets_step0.py` : reproduction complète.

## 9. Point nécessitant validation avant l'étape 1

Pour S07–S09, faut-il que les futurs tableaux IS/validation utilisent :

- l'**ensemble exécutable**, avec un IS seulement partiel à partir du 3 janvier 2024 ; ou
- le **composite de recherche des cellules** disponible depuis le 3 mai 2021, qui n'est pas le ledger tradable et ne possède pas un nombre de trades agrégé comparable ?

L'étape 0 s'arrête ici, conformément à la demande.
