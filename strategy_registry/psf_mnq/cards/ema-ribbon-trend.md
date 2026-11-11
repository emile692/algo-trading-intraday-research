# EMA Ribbon Trend

- **ID** : `ema-ribbon-trend`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `trend`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/ema-ribbon-trend) · [page MNQ](https://pinescriptforge.com/mnq/ema-ribbon-trend/backtest)

## Description publiée

Multiple EMA ribbon alignment for powerful trend-following entries.

- **Entrée annoncée** : Enter long when EMAs are stacked bullish and price pulls back to the 8 or 13 EMA. Enter short on bearish stack pullbacks.
- **Sortie annoncée** : Exit when the 8 EMA crosses below the 21 EMA (long) or above (short). Stop at the 55 EMA.

## Données et formalisation

- **Indicateurs cités** : EMA 8, EMA 13, EMA 21, EMA 34, EMA 55
- **Horizons cités** : 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
