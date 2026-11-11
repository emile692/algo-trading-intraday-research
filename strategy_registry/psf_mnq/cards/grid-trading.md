# Grid Trading

- **ID** : `grid-trading`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `mean-reversion`
- **Famille de recherche (provisoire)** : `money_management`
- **Rôle envisagé (provisoire)** : `exclude_as_alpha`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/grid-trading) · [page MNQ](https://pinescriptforge.com/mnq/grid-trading/backtest)

## Description publiée

Places buy and sell orders at regular intervals above and below price.

- **Entrée annoncée** : Set grid levels at every 0.5× ATR above and below the current price. Buy at each lower grid level, sell at each upper grid level.
- **Sortie annoncée** : Each position targets the next grid level (0.5× ATR profit). Stop all positions if price moves beyond the grid boundary (3× ATR from center).

## Données et formalisation

- **Indicateurs cités** : ATR, Grid Levels, Price Range
- **Horizons cités** : 5min, 15min, 1H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
