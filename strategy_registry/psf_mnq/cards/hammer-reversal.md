# Hammer Reversal

- **ID** : `hammer-reversal`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `price-action`
- **Famille de recherche (provisoire)** : `price_action`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/hammer-reversal) · [page MNQ](https://pinescriptforge.com/mnq/hammer-reversal/backtest)

## Description publiée

Identifies hammer candles at support for bullish reversal entries.

- **Entrée annoncée** : Enter long when a hammer forms at a support level, moving average, or Fibonacci retracement. Lower shadow must be ≥2× the body.
- **Sortie annoncée** : Target 2× the hammer range above the high. Stop below the hammer low. Trail at 1× ATR after initial target.

## Données et formalisation

- **Indicateurs cités** : Candlestick Patterns, Support Levels, EMA 50, ATR
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
