# Narrow Body Candle

- **ID** : `narrow-body-candle`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `range`
- **Famille de recherche (provisoire)** : `breakout_expansion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/narrow-body-candle) · [page MNQ](https://pinescriptforge.com/mnq/narrow-body-candle/backtest)

## Description publiée

Trades expansion moves after clusters of narrow-body (doji-like) candles.

- **Entrée annoncée** : Enter in the direction of the first wide-body candle after 3+ consecutive narrow bodies (<30% of ATR).
- **Sortie annoncée** : Target 2× ATR from entry. Stop at the opposite side of the narrow-body cluster.

## Données et formalisation

- **Indicateurs cités** : Candle Body Size, ATR, Volume
- **Horizons cités** : 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
