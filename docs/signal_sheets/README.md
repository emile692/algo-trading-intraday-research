# Dossier client — neuf fiches signal

Version documentaire : **1er novembre 2026**. Portefeuille de référence : `nine_sleeve_equal_contract_reference_20261101`.

> **Mise à jour du 21 novembre 2026.** Les fiches 01 à 04 (MNQ) et le Book sont calculés sur MNQ `v.0` (le contrat le plus échangé, celui que trade la production), ajusté en arrière de l'écart de roll, et non plus sur `c.0`. La référence active est `eight_sleeve_equal_contract_reference_20261121` ([`ACTIVE_EIGHT_SLEEVES.md`](../ACTIVE_EIGHT_SLEEVES.md)). Les règles et paramètres ne changent pas. Les cubes et grilles de paramètres des fiches restent ceux calculés sur `c.0` lors de la sélection.

Ce dossier décrit les neuf sleeves actives avec un gabarit identique. Une sleeve est une règle de signal autonome ; le portefeuille additionne une copie entière de chacune (`[1,1,1,1,1,1,1,1,1]`). Les chiffres sont des backtests nets historiques, pas des promesses de performance.

Le recueil complet se termine par une section [Book](BOOK.md) : equity agrégée, matrice de corrélation, présence et nature des stops, risque prévu par position, mesures de risque réalisées et analyse leave-one-out. Tous les diagnostics de book restent figés sur la période de recherche commune du 2 janvier 2024 au 19 mars 2026, même si des données plus récentes sont disponibles.

## Éditions PDF

- [Recueil client complet — 9 sleeves](pdf/00_portefeuille_9_sleeves_complet.pdf)
- [Recueil client complet — mise à jour cubes IS/OOS](pdf/00_portefeuille_9_sleeves_complet_IS_OOS.pdf)
- [01 — MNQ ORB + VWAP + ATR](pdf/01_mnq_orb_vwap_atr.pdf)
- [02 — MNQ volatility breakout](pdf/02_mnq_volatility_breakout.pdf)
- [03 — MNQ compression breakout](pdf/03_mnq_compression_breakout.pdf)
- [04 — MNQ ATR trend flip](pdf/04_mnq_atr_trend_flip.pdf)
- [05 — MGC volatility breakout](pdf/05_mgc_volatility_breakout.pdf)
- [06 — MGC gap continuation](pdf/06_mgc_gap_continuation.pdf)
- [07 — MBT rolling range breakout](pdf/07_mbt_rolling_range_breakout.pdf)
- [08 — MBT trend pullback MA](pdf/08_mbt_trend_pullback_ma_structure.pdf)
- [09 — MBT volatility breakout](pdf/09_mbt_volatility_breakout.pdf)

## Éditions HTML interactives

- [Recueil client complet — 9 sleeves](html/00_portefeuille_9_sleeves_complet.html)
- [01 — MNQ ORB + VWAP + ATR](html/01_mnq_orb_vwap_atr.html)
- [02 — MNQ volatility breakout](html/02_mnq_volatility_breakout.html)
- [03 — MNQ compression breakout](html/03_mnq_compression_breakout.html)
- [04 — MNQ ATR trend flip](html/04_mnq_atr_trend_flip.html)
- [05 — MGC volatility breakout](html/05_mgc_volatility_breakout.html)
- [06 — MGC gap continuation](html/06_mgc_gap_continuation.html)
- [07 — MBT rolling range breakout](html/07_mbt_rolling_range_breakout.html)
- [08 — MBT trend pullback MA](html/08_mbt_trend_pullback_ma_structure.html)
- [09 — MBT volatility breakout](html/09_mbt_volatility_breakout.html)

| # | Fiche | Instrument | Famille | Unité exécutable |
|---:|---|---|---|---|
| 1 | [MNQ ORB + VWAP + ATR](01_mnq_orb_vwap_atr.md) | MNQ | breakout d'ouverture | sizing risque historique |
| 2 | [MNQ volatility breakout](02_mnq_volatility_breakout.md) | MNQ | expansion de volatilité | 1 contrat agrégé |
| 3 | [MNQ compression breakout](03_mnq_compression_breakout.md) | MNQ | compression → expansion | 1 contrat agrégé |
| 4 | [MNQ ATR trend flip](04_mnq_atr_trend_flip.md) | MNQ | suivi de tendance | 1 contrat agrégé |
| 5 | [MGC volatility breakout](05_mgc_volatility_breakout.md) | MGC | expansion de volatilité | 1 contrat agrégé |
| 6 | [MGC gap continuation](06_mgc_gap_continuation.md) | MGC | continuation de gap | 1 contrat agrégé |
| 7 | [MBT rolling range breakout](07_mbt_rolling_range_breakout.md) | MBT | vote Donchian | 1 contrat agrégé |
| 8 | [MBT trend pullback MA](08_mbt_trend_pullback_ma_structure.md) | MBT | vote tendance/pullback | 1 contrat agrégé |
| 9 | [MBT volatility breakout](09_mbt_volatility_breakout.md) | MBT | vote expansion de volatilité | 1 contrat agrégé |

## Lecture commune

```mermaid
flowchart LR
    A[Données 1 minute] --> B[Barres et indicateurs causaux]
    B --> C[Signal sur information terminée]
    C --> D[Ordre ou cible de position]
    D --> E[Fill avec slippage et commission]
    E --> F[Stop / cible / sortie horaire]
    F --> G[PnL net de la sleeve]
    G --> H[Somme equal-contract des 9 sleeves]
```

**Exécution agrégée (13 novembre 2026).** Les sleeves d'un même instrument ne sont plus envoyées séparément : une seule position nette par instrument (MNQ = S01 + S02 + S03, MGC = S05 + S06, MBT = S07 + S08 + S09), avec les mêmes signaux et les mêmes tailles. Règle et exemples réels : [Book, section 7](BOOK.md).

Conventions transversales : horodatage `America/New_York`, aucune pyramide, information de barre disponible seulement à sa clôture, arrondi au tick défavorable, commission par côté, et aucune position après le hard flatten applicable. Si stop et cible sont touchés dans la même barre minute et que l'ordre intrabar est inconnu, le stop est prioritaire. Une barre d'exécution requise mais absente entraîne l'abandon du signal.

## Comment lire les fiches

- **OR** (*opening range*) : plus haut et plus bas construits pendant une fenêtre définie après l'ouverture.
- **VWAP** : prix moyen de séance pondéré par les volumes ; il matérialise le prix d'équilibre intrajournalier.
- **ATR** : mesure d'amplitude calculée sans donnée future ; `ATR14 journalier` signifie quatorze séances entières, pas quatorze minutes.
- **SL / TP** : stop-loss d'invalidation / take-profit. Lorsqu'une sleeve n'a pas de TP, la fiche l'indique explicitement.
- **Cellule** : une combinaison de paramètres testée. Dans les sleeves 2 à 9, les cellules votent mais ne sont pas autant de contrats exécutés : seule la cible agrégée `−1/0/+1` est tradée.
- **IS / validation / OOS** : l'IS sert à construire ou choisir ; une validation n'est indépendante que si elle n'a jamais influencé le choix. Dès qu'elle est consultée pour geler une région, elle devient un « legacy OOS » et seul le suivi ultérieur est véritablement prospectif.

Chaque fiche présente la logique dans cet ordre : définitions, paramètres, chaîne causale, exemple numérique, règles de taille et de sortie, puis périodes de validation séparées. Pour toutes les fiches, la section 4 montre la vue 3D des paramètres : le Sharpe annualisé de chaque cellule testée en IS et en validation, un cube quand la sleeve a trois paramètres, une surface de réponse quand elle en a deux, avec les cellules gelées encadrées en bleu. Les fiches 3 à 9 détaillent en section 12 l'inventaire complet, les grilles et le récapitulatif. Elle sert à juger la continuité d'un plateau, jamais à promouvoir rétroactivement la meilleure cellule.

## Statut de validation

- Pour les trois baselines MNQ hors ORB, **IS = 2019-05-06–2023-12-31** et **validation = 2024-01-01–2026-03-19**. La fenêtre commune 2024–2026 sert aussi à l'assemblage du portefeuille ; elle n'est donc pas un test supplémentaire.
- L'ORB dispose de son propre découpage chronologique historique, indiqué dans sa fiche, puis de la même fenêtre portefeuille commune.
- Les deux MGC sont des locks issus d'une période explicitement **réutilisée/adaptative**, pas d'un OOS vierge.
- Pour MBT, le **legacy IS** va du 2021-05-03 au 2025-05-28 et le **legacy OOS** du 2025-05-29 au 2026-10-02. Les trois ensembles sont des régions de cellules gelées ; le legacy OOS a servi à définir ces régions et n'est donc plus indépendant.
- Les clusters S3–S9 ont été gelés le **1er novembre 2026**, après lecture des fenêtres de validation : celles-ci sont des validations réutilisées pour toutes les sleeves.
- Le seul suivi réellement prospectif commence après le **2 octobre 2026**.

## Clusters gelés le 1er novembre 2026

Les sleeves 3 à 9 sont désormais, comme la sleeve 2, des **ensembles de cellules qui votent** : chaque cellule tient une position virtuelle `−1/0/+1`, la sleeve prend la moyenne et ne trade qu'une cible agrégée `−1/0/+1` à partir d'un seuil de `±0,50`. Pour S3 à S6, le cluster est le voisinage immédiat de l'ancien point (un cran de grille de part et d'autre) ; pour S7 à S9, c'est la région MBT déjà gelée, dont S7 perd un doublon exact (`M=15`, identique à `M=12`).

Ce choix ne regarde pas les performances : une validation croisée combinatoire dans l'IS a montré que toute règle choisissant un cluster d'après son Sharpe sur-ajuste (probabilité de sur-ajustement de 54 à 65 %). Détails : [sélection](../../research/signal_sheet_parameter_refinement/cluster_selection/cluster_selection_cscv.json), [manifestes gelés](../../strategy_registry/psf_mnq/frozen_ensembles/).

| Fiche | Cluster | Cellules | Sharpe IS du vote | Sharpe OOS réutilisé du vote |
|---|---|---:|---:|---:|
| [S3](03_mnq_compression_breakout.md) | NR 8–12 × buffer 0–2 × cible 1,5–2,5 | 27 | 0,546 | 0,283 |
| [S4](04_mnq_atr_trend_flip.md) | ATR 8–12 × bande 1,5–2,0 | 9 | 0,354 | 0,543 |
| [S5](05_mgc_volatility_breakout.md) | ATR 20–40 × multiple 0,9–1,3 | 9 | 0,519 | 0,388 |
| [S6](06_mgc_gap_continuation.md) | gap 0,4–0,6 × volume 0,5–0,7 | 9 | 0,878 | 0,732 |
| [S7](07_mbt_rolling_range_breakout.md) | canal 20–24 × sortie 7–12 | 6 | 0,390 | 0,638 |
| [S8](08_mbt_trend_pullback_ma_structure.md) | échelle rapide 0,50–0,75 × lente 0,75–1,25 | 9 | 0,621 | 0,687 |
| [S9](09_mbt_volatility_breakout.md) | ATR 10–30 × multiple 0,30/0,40/0,60 | 12 | 1,314 | −0,040 |

Les Sharpe sont ceux du vote réellement exécuté, sur le PnL quotidien net. **S9 est nul sur le legacy OOS** alors que ses cellules y sont positives : la sleeve doit être revue avant toute nouvelle allocation. La section 4 de chaque fiche montre la vue 3D des paramètres, cellules gelées encadrées en bleu ; la section 12 en conserve le détail.

## Sources de gouvernance

- [Référence active — huit sleeves, S04 en shadow](../ACTIVE_EIGHT_SLEEVES.md)
- [Configuration verrouillée (21 novembre 2026, MNQ v.0)](../../research/locked_eight_sleeve_portfolio_20261121/locked_eight_sleeve_config.json)
- [Référence remplacée du 1er novembre 2026](../ACTIVE_NINE_SLEEVES.md)
- [Référence remplacée du 4 octobre 2026](../../research/locked_nine_sleeve_portfolio/SUPERSEDED.md)
- [Ensembles de vote gelés](../../strategy_registry/psf_mnq/frozen_ensembles/)
- [Manifeste MBT exécutable](../../research/mbt_three_sleeve_freeze_20261004/vote_ensemble/ensemble_manifest.json)
- [Registre des recherches non promues](../RESEARCH_FAILURE_REGISTER_2026.md)

