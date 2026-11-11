# SuperTrend

- **ID** : `supertrend`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `trend`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/supertrend) · [page MNQ](https://pinescriptforge.com/mnq/supertrend/backtest)

## Description publiée

ATR-based trend indicator that flips between bullish and bearish.

- **Entrée annoncée** : Enter long when SuperTrend flips bullish (line moves below price). Enter short when it flips bearish.
- **Sortie annoncée** : Exit on SuperTrend flip. Optional: tighten ATR multiplier for faster exits in volatile markets.

## Données et formalisation

- **Indicateurs cités** : SuperTrend, ATR
- **Horizons cités** : 15min, 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
