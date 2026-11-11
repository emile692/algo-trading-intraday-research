# Fractal Adaptive MA

- **ID** : `fractal-adaptive-ma`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `trend`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/fractal-adaptive-ma) · [page MNQ](https://pinescriptforge.com/mnq/fractal-adaptive-ma/backtest)

## Description publiée

Uses fractal dimension to create an adaptive moving average for trend detection.

- **Entrée annoncée** : Enter long when FRAMA slopes up and price > FRAMA. Short on downslope. Fractal dim < 1.4 = trending.
- **Sortie annoncée** : Exit when fractal dimension > 1.6 (choppy market) or FRAMA flattens.

## Données et formalisation

- **Indicateurs cités** : FRAMA, Fractal Dimension, ATR
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
