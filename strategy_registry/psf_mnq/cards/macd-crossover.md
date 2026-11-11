# MACD Crossover

- **ID** : `macd-crossover`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `momentum`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/macd-crossover) · [page MNQ](https://pinescriptforge.com/mnq/macd-crossover/backtest)

## Description publiée

Classic MACD line/signal crossover for momentum entries.

- **Entrée annoncée** : Enter long when MACD line crosses above signal line. Enter short when MACD line crosses below signal line.
- **Sortie annoncée** : Exit on opposing crossover or when histogram contracts 50% from peak. Stop-loss at 2× ATR.

## Données et formalisation

- **Indicateurs cités** : MACD, Signal Line, Histogram, ATR
- **Horizons cités** : 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
