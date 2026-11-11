# Rate of Change (ROC)

- **ID** : `rate-of-change`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `momentum`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/rate-of-change) · [page MNQ](https://pinescriptforge.com/mnq/rate-of-change/backtest)

## Description publiée

Measures percentage price change over a fixed period for momentum signals.

- **Entrée annoncée** : Enter long when 14-period ROC crosses above zero and 50 SMA is rising. Enter short when ROC crosses below zero with declining SMA.
- **Sortie annoncée** : Exit when ROC crosses back through zero or on ROC divergence against position. Trail stop at 1.5× ATR.

## Données et formalisation

- **Indicateurs cités** : ROC 14, SMA 50, ATR
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
