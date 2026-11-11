# Connors RSI

- **ID** : `connors-rsi`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `momentum`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/connors-rsi) · [page MNQ](https://pinescriptforge.com/mnq/connors-rsi/backtest)

## Description publiée

Composite RSI combining price RSI, streak RSI, and percent rank for timing.

- **Entrée annoncée** : Enter long when Connors RSI drops below 10 (deep oversold). Enter short when Connors RSI exceeds 90. Confirm with price above/below 200 SMA for trend filter.
- **Sortie annoncée** : Exit long when Connors RSI crosses above 60. Exit short when it crosses below 40. Stop at 2× ATR.

## Données et formalisation

- **Indicateurs cités** : Connors RSI, RSI, Streak RSI, Percent Rank
- **Horizons cités** : 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
