# Stochastic Divergence

- **ID** : `stochastic-divergence`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `momentum`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/stochastic-divergence) · [page MNQ](https://pinescriptforge.com/mnq/stochastic-divergence/backtest)

## Description publiée

Detects hidden and regular divergences between price and Stochastic.

- **Entrée annoncée** : Enter long on bullish divergence (price lower low, Stochastic higher low) in oversold zone. Opposite for shorts.
- **Sortie annoncée** : Exit when Stochastic reaches opposite extreme or on Stochastic crossover in profit. Stop below swing low.

## Données et formalisation

- **Indicateurs cités** : Stochastic %K, Stochastic %D, Divergence Detection
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
