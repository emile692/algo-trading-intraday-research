# Hidden Bearish Divergence

- **ID** : `hidden-bearish-divergence`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `divergence`
- **Famille de recherche (provisoire)** : `reversal_exhaustion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/hidden-bearish-divergence) · [page MNQ](https://pinescriptforge.com/mnq/hidden-bearish-divergence/backtest)

## Description publiée

Detects hidden bearish divergence for trend continuation short entries.

- **Entrée annoncée** : Enter short when price lower high + RSI higher high in a downtrend (price below 50 SMA).
- **Sortie annoncée** : Target previous swing low. Stop above the lower high. Trail at 1.5× ATR.

## Données et formalisation

- **Indicateurs cités** : RSI, SMA 50, Swing Detection
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
