# Marubozu Momentum

- **ID** : `marubozu-momentum`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `price-action`
- **Famille de recherche (provisoire)** : `price_action`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/marubozu-momentum) · [page MNQ](https://pinescriptforge.com/mnq/marubozu-momentum/backtest)

## Description publiée

Full-bodied candles with no wicks signal extreme conviction.

- **Entrée annoncée** : Enter on a pullback to the midpoint of a Marubozu candle in the direction of the Marubozu. Volume must be above average.
- **Sortie annoncée** : Target 1× the Marubozu range beyond the entry. Stop at the opposite end of the Marubozu. Trail at swing highs/lows.

## Données et formalisation

- **Indicateurs cités** : Candlestick Patterns, Volume, ATR
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
