# Order Flow Scalp

- **ID** : `order-flow-scalp`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `scalping`
- **Famille de recherche (provisoire)** : `short_horizon`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/order-flow-scalp) · [page MNQ](https://pinescriptforge.com/mnq/order-flow-scalp/backtest)

## Description publiée

Footprint chart scalping using delta divergences and imbalances.

- **Entrée annoncée** : Enter when footprint shows 3:1 buy/sell imbalance at a key level with delta divergence. Confirm with absorption visible in the footprint.
- **Sortie annoncée** : Target the next volume cluster. Stop at the imbalance level break. Hold 1-5 minutes maximum.

## Données et formalisation

- **Indicateurs cités** : Footprint Chart, Delta, Cumulative Delta, Volume Imbalance
- **Horizons cités** : 1min, 5min
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
