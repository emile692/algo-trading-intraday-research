# Ultimate Oscillator

- **ID** : `ultimate-oscillator`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `momentum`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/ultimate-oscillator) · [page MNQ](https://pinescriptforge.com/mnq/ultimate-oscillator/backtest)

## Description publiée

Multi-timeframe momentum using three weighted periods for divergence trading.

- **Entrée annoncée** : Enter long on bullish divergence when UO is below 30 and then crosses above 50. Enter short on bearish divergence above 70.
- **Sortie annoncée** : Exit long when UO reaches 70 or bearish divergence appears. Exit short when UO reaches 30. Stop at swing point.

## Données et formalisation

- **Indicateurs cités** : Ultimate Oscillator, Divergence Detection
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
