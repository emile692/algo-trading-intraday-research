# True Strength Index (TSI)

- **ID** : `tsi-momentum`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `momentum`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/tsi-momentum) · [page MNQ](https://pinescriptforge.com/mnq/tsi-momentum/backtest)

## Description publiée

Double-smoothed momentum oscillator for trend direction and crossover signals.

- **Entrée annoncée** : Enter long when TSI crosses above its signal line while both are below zero (early momentum). Enter short on bearish signal crossover above zero.
- **Sortie annoncée** : Exit when TSI crosses below signal line or reaches extreme levels (±25). Trail at 1.5× ATR.

## Données et formalisation

- **Indicateurs cités** : TSI, TSI Signal, ATR
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
