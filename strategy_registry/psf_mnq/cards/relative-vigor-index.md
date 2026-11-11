# Relative Vigor Index

- **ID** : `relative-vigor-index`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `momentum`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/relative-vigor-index) · [page MNQ](https://pinescriptforge.com/mnq/relative-vigor-index/backtest)

## Description publiée

Compares closing strength to trading range for momentum assessment.

- **Entrée annoncée** : Enter long when RVI crosses above its signal line below the zero line. Enter short on cross below signal above zero. Require price confirmation via breakout.
- **Sortie annoncée** : Exit on RVI signal line cross against position. Divergence between RVI and price triggers early exit.

## Données et formalisation

- **Indicateurs cités** : RVI 10, RVI Signal, SMA 20
- **Horizons cités** : 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).
- **Nom partagé avec** : `rvi-divergence`. Dédupliquer la logique avant expérimentation.

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
