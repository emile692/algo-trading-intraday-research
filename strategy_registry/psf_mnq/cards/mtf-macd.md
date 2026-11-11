# Multi-Timeframe MACD

- **ID** : `mtf-macd`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `multi-timeframe`
- **Famille de recherche (provisoire)** : `composite`
- **Rôle envisagé (provisoire)** : `meta_signal`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/mtf-macd) · [page MNQ](https://pinescriptforge.com/mnq/mtf-macd/backtest)

## Description publiée

Aligns MACD signals across 3 timeframes for high-conviction entries.

- **Entrée annoncée** : Enter long when Daily MACD histogram is positive, 4H MACD line is above signal, and 1H MACD crosses bullish. Opposite for short.
- **Sortie annoncée** : Exit when 4H MACD crosses against position. Trail at 1H swing lows/highs. Stop at 1.5× ATR on 1H chart.

## Données et formalisation

- **Indicateurs cités** : MACD (Daily), MACD (4H), MACD (1H), ATR
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
