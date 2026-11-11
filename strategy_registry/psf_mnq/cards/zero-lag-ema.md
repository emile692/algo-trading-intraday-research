# Zero Lag EMA

- **ID** : `zero-lag-ema`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `trend`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/zero-lag-ema) · [page MNQ](https://pinescriptforge.com/mnq/zero-lag-ema/backtest)

## Description publiée

Lag-reduced EMA that responds faster to price changes for earlier trend signals.

- **Entrée annoncée** : Enter long when Zero Lag EMA turns up and price crosses above it. Short when it turns down.
- **Sortie annoncée** : Exit when price closes below Zero Lag EMA (for longs). Trail at 1× ATR.

## Données et formalisation

- **Indicateurs cités** : Zero Lag EMA, ATR
- **Horizons cités** : 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
