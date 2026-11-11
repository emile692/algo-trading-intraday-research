# Rate of Change Momentum

- **ID** : `rate-of-change-momentum`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `momentum`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/rate-of-change-momentum) · [page MNQ](https://pinescriptforge.com/mnq/rate-of-change-momentum/backtest)

## Description publiée

Measures percentage price change over a period for pure momentum signals.

- **Entrée annoncée** : Enter long when ROC crosses above zero with increasing magnitude. Enter short on bearish zero-line cross.
- **Sortie annoncée** : Exit when ROC diverges from price or reverses from extreme. Stop at 2× ATR from entry.

## Données et formalisation

- **Indicateurs cités** : ROC, ROC SMA 9, Zero Line
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
