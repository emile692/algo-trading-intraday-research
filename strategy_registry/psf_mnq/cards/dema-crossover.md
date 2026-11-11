# DEMA Crossover

- **ID** : `dema-crossover`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `trend`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/dema-crossover) · [page MNQ](https://pinescriptforge.com/mnq/dema-crossover/backtest)

## Description publiée

Double Exponential Moving Average crossover for reduced-lag trend signals.

- **Entrée annoncée** : Enter long when 9 DEMA crosses above 21 DEMA. Enter short on bearish crossover. Volume confirmation recommended.
- **Sortie annoncée** : Exit on opposing DEMA crossover. Trail stop at the slower DEMA.

## Données et formalisation

- **Indicateurs cités** : DEMA 9, DEMA 21, Volume
- **Horizons cités** : 5min, 15min, 1H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
