# Schaff Trend Cycle

- **ID** : `schaff-trend-cycle`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `momentum`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/schaff-trend-cycle) · [page MNQ](https://pinescriptforge.com/mnq/schaff-trend-cycle/backtest)

## Description publiée

Combines MACD with double-smoothed stochastic for faster cycle detection.

- **Entrée annoncée** : Enter long when STC crosses above 25 from oversold. Enter short when STC crosses below 75 from overbought.
- **Sortie annoncée** : Exit long when STC reaches 75. Exit short at 25. Stop at recent swing high/low.

## Données et formalisation

- **Indicateurs cités** : STC, MACD, Stochastic
- **Horizons cités** : 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
