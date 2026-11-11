# Aggressive Exhaustion Reversal

- **ID** : `aggressive-exhaustion`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `advanced-order-flow`
- **Famille de recherche (provisoire)** : `order_flow`
- **Rôle envisagé (provisoire)** : `data_gated_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/aggressive-exhaustion) · [page MNQ](https://pinescriptforge.com/mnq/aggressive-exhaustion/backtest)

## Description publiée

Trades reversals when aggressive buying or selling shows exhaustion patterns.

- **Entrée annoncée** : Enter counter-trend when aggressive volume > 90th percentile but price progress per unit of volume drops below 50th percentile (exhaustion).
- **Sortie annoncée** : Target: VWAP or mean price of the aggressive move. Stop at the exhaustion extreme + 3 ticks. Quick scalp — 5-15 minute hold.

## Données et formalisation

- **Indicateurs cités** : Aggressive Volume, Price Progress, Volume Efficiency, VWAP
- **Horizons cités** : 1min, 5min
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Vérifier la nécessité de données tick, carnet ou ordres; pas de substitution OHLCV implicite.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
