# VWAP Crossover Trend

- **ID** : `vwap-crossover`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `trend`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/vwap-crossover) · [page MNQ](https://pinescriptforge.com/mnq/vwap-crossover/backtest)

## Description publiée

Trades directional bias using VWAP as institutional support/resistance.

- **Entrée annoncée** : Enter long when price crosses above VWAP with increasing volume and positive momentum (MACD histogram). Enter short on cross below VWAP with volume.
- **Sortie annoncée** : Exit on cross back through VWAP or at the session close. Stop at 1× ATR from VWAP.

## Données et formalisation

- **Indicateurs cités** : VWAP, Volume, MACD Histogram
- **Horizons cités** : 5min, 15min, 1H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Calendrier de séance et horodatage New York à définir.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
