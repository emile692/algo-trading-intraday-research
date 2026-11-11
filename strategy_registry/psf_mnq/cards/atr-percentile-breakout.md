# ATR Percentile Breakout

- **ID** : `atr-percentile-breakout`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `volatility`
- **Famille de recherche (provisoire)** : `volatility_regime`
- **Rôle envisagé (provisoire)** : `feature_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/atr-percentile-breakout) · [page MNQ](https://pinescriptforge.com/mnq/atr-percentile-breakout/backtest)

## Description publiée

Enters when ATR drops to extreme lows then expands, catching volatility expansions.

- **Entrée annoncée** : When ATR is below its 10th percentile of the last 100 bars, enter long on break above 5-bar high or short on break below 5-bar low.
- **Sortie annoncée** : Target 2× ATR from entry. Trail at 1× ATR. Exit if ATR contracts again within 5 bars.

## Données et formalisation

- **Indicateurs cités** : ATR, ATR Percentile, Price Breakout
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
