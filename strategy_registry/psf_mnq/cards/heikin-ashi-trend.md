# Heikin Ashi Trend

- **ID** : `heikin-ashi-trend`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `trend`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/heikin-ashi-trend) · [page MNQ](https://pinescriptforge.com/mnq/heikin-ashi-trend/backtest)

## Description publiée

Uses smoothed Heikin Ashi candles to filter noise and ride clean trends.

- **Entrée annoncée** : Enter long when Heikin Ashi candles turn green with no lower wick (strong bullish) and price is above the 50 EMA. Enter short on red candles with no upper wick below 50 EMA.
- **Sortie annoncée** : Exit when a Heikin Ashi doji or reversal candle appears (both upper and lower wicks). Re-entry allowed on next clean trend candle. Trail stop at 2× ATR.

## Données et formalisation

- **Indicateurs cités** : Heikin Ashi Candles, EMA 50, ATR
- **Horizons cités** : 15min, 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
