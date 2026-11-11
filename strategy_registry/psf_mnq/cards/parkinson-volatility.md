# Parkinson Range Volatility

- **ID** : `parkinson-volatility`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `volatility`
- **Famille de recherche (provisoire)** : `volatility_regime`
- **Rôle envisagé (provisoire)** : `feature_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/parkinson-volatility) · [page MNQ](https://pinescriptforge.com/mnq/parkinson-volatility/backtest)

## Description publiée

Uses the high-low range for more efficient volatility estimation.

- **Entrée annoncée** : Enter long when Parkinson volatility breaks above its 20-period upper Bollinger Band while price is above VWAP. Enter short when volatility breaks out with price below VWAP.
- **Sortie annoncée** : Exit when Parkinson volatility returns to its moving average. Stop at 2× current ATR.

## Données et formalisation

- **Indicateurs cités** : Parkinson Vol 10, Parkinson Vol BB 20, VWAP
- **Horizons cités** : 5min, 15min, 1H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
