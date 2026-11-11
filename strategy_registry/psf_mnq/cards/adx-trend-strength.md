# ADX Trend Strength

- **ID** : `adx-trend-strength`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `trend`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/adx-trend-strength) · [page MNQ](https://pinescriptforge.com/mnq/adx-trend-strength/backtest)

## Description publiée

Uses ADX to identify strong trending periods and ride momentum.

- **Entrée annoncée** : Enter long when +DI crosses above -DI and ADX > 25. Enter short when -DI crosses above +DI and ADX > 25.
- **Sortie annoncée** : Exit when ADX drops below 20 or opposing DI crossover occurs. Trailing stop at 2× ATR.

## Données et formalisation

- **Indicateurs cités** : ADX, +DI, -DI, ATR
- **Horizons cités** : 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
