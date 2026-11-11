# Bullish Engulfing

- **ID** : `bullish-engulfing`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `price-action`
- **Famille de recherche (provisoire)** : `price_action`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/bullish-engulfing) · [page MNQ](https://pinescriptforge.com/mnq/bullish-engulfing/backtest)

## Description publiée

Enters long on bullish engulfing candles at support with trend confirmation.

- **Entrée annoncée** : Enter long when a bullish engulfing candle forms at or near the 50 EMA or a recent swing low. Volume must exceed the 20-period average.
- **Sortie annoncée** : Target 2× the engulfing candle range. Stop below the engulfing candle low. Trail at 1.5× ATR after 1R profit.

## Données et formalisation

- **Indicateurs cités** : Candlestick Patterns, EMA 50, Volume, ATR
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
