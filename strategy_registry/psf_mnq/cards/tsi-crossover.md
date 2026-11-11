# TSI Crossover

- **ID** : `tsi-crossover`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `momentum`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/tsi-crossover) · [page MNQ](https://pinescriptforge.com/mnq/tsi-crossover/backtest)

## Description publiée

True Strength Index double-smoothed momentum for clean, low-noise signals.

- **Entrée annoncée** : Enter long when TSI crosses above its signal line below zero. Enter short when TSI crosses below signal above zero.
- **Sortie annoncée** : Exit on opposing signal crossover. Secondary exit when TSI reaches +25/-25 extreme zones.

## Données et formalisation

- **Indicateurs cités** : TSI, TSI Signal, Zero Line
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
