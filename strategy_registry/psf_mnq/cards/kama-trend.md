# KAMA Trend

- **ID** : `kama-trend`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `trend`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/kama-trend) · [page MNQ](https://pinescriptforge.com/mnq/kama-trend/backtest)

## Description publiée

Adaptive moving average that speeds up in trends and slows in ranges.

- **Entrée annoncée** : Enter long when price crosses above KAMA and KAMA slope turns positive. Short when price crosses below with negative slope.
- **Sortie annoncée** : Exit when price crosses back through KAMA in the opposite direction. Trail at KAMA level. Stop at 2× ATR.

## Données et formalisation

- **Indicateurs cités** : KAMA, KAMA Slope, ATR
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
