# Dual Moving Average Crossover

- **ID** : `dual-ma-crossover`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `trend`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/dual-ma-crossover) · [page MNQ](https://pinescriptforge.com/mnq/dual-ma-crossover/backtest)

## Description publiée

The classic fast/slow moving average crossover trend system.

- **Entrée annoncée** : Enter long when 9 EMA crosses above 21 EMA and ADX > 20. Short on bearish crossover with ADX > 20.
- **Sortie annoncée** : Exit on opposing crossover. Optional: exit early if ADX drops below 15 (trend dying).

## Données et formalisation

- **Indicateurs cités** : EMA 9, EMA 21, ADX
- **Horizons cités** : 15min, 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
