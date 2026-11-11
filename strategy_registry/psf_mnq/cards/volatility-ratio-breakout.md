# Volatility Ratio Breakout

- **ID** : `volatility-ratio-breakout`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `volatility`
- **Famille de recherche (provisoire)** : `volatility_regime`
- **Rôle envisagé (provisoire)** : `feature_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/volatility-ratio-breakout) · [page MNQ](https://pinescriptforge.com/mnq/volatility-ratio-breakout/backtest)

## Description publiée

Detects breakout bars using the ratio of true range to average range.

- **Entrée annoncée** : Enter in the direction of the breakout bar when Volatility Ratio > 2.0. Use the bar's direction (close vs open) for trade direction.
- **Sortie annoncée** : Trail stop at 1.5× ATR. Take profit when Volatility Ratio normalizes below 1.0 for 3 consecutive bars.

## Données et formalisation

- **Indicateurs cités** : Volatility Ratio, True Range, ATR
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
