# Tape Reading Scalp

- **ID** : `tape-reading-scalp`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `scalping`
- **Famille de recherche (provisoire)** : `short_horizon`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/tape-reading-scalp) · [page MNQ](https://pinescriptforge.com/mnq/tape-reading-scalp/backtest)

## Description publiée

Order flow-based scalping using time & sales and level 2 data.

- **Entrée annoncée** : Enter long when large buy prints appear at bid with absorption (large resting orders absorbing selling). Short on large sell prints at ask.
- **Sortie annoncée** : Target 4-8 ticks. Stop 4 ticks. Exit immediately if flow reverses. Maximum 2-minute hold.

## Données et formalisation

- **Indicateurs cités** : Time & Sales, Level 2, Volume Delta, Cumulative Delta
- **Horizons cités** : 1min, 5min
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Vérifier la nécessité de données tick, carnet ou ordres; pas de substitution OHLCV implicite.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
