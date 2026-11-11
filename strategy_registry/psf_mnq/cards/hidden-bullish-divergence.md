# Hidden Bullish Divergence

- **ID** : `hidden-bullish-divergence`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `divergence`
- **Famille de recherche (provisoire)** : `reversal_exhaustion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/hidden-bullish-divergence) · [page MNQ](https://pinescriptforge.com/mnq/hidden-bullish-divergence/backtest)

## Description publiée

Detects hidden bullish divergence for trend continuation long entries.

- **Entrée annoncée** : Enter long when price higher low + RSI lower low in an uptrend (price above 50 SMA).
- **Sortie annoncée** : Target previous swing high. Stop below the higher low. Trail at 1.5× ATR.

## Données et formalisation

- **Indicateurs cités** : RSI, SMA 50, Swing Detection
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
