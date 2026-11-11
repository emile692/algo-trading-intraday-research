# Spread Scalp

- **ID** : `spread-scalp`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `scalping`
- **Famille de recherche (provisoire)** : `short_horizon`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/spread-scalp) · [page MNQ](https://pinescriptforge.com/mnq/spread-scalp/backtest)

## Description publiée

Exploits temporary spread widening for mean-reversion scalps.

- **Entrée annoncée** : Place limit buy at bid when spread widens to 2× normal on a liquid contract. Limit sell at ask when spread widens on the opposite side.
- **Sortie annoncée** : Target 1-2 ticks when spread normalizes. Cancel if not filled within 5 seconds. Maximum 30-second hold.

## Données et formalisation

- **Indicateurs cités** : Bid-Ask Spread, Level 2, Latency
- **Horizons cités** : 1min
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Vérifier la nécessité de données tick, carnet ou ordres; pas de substitution OHLCV implicite.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
