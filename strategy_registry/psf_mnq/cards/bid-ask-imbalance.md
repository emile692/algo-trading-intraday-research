# Bid-Ask Imbalance

- **ID** : `bid-ask-imbalance`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `microstructure`
- **Famille de recherche (provisoire)** : `order_flow`
- **Rôle envisagé (provisoire)** : `data_gated_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/bid-ask-imbalance) · [page MNQ](https://pinescriptforge.com/mnq/bid-ask-imbalance/backtest)

## Description publiée

Trades directional order flow imbalances in the order book for short-term momentum.

- **Entrée annoncée** : Enter long when bid imbalance ratio exceeds 2.0 with rising cumulative delta. Short when ask imbalance exceeds 2.0.
- **Sortie annoncée** : Exit when imbalance normalizes (ratio returns to 0.8–1.2). Quick 5-tick stop. Target 2× the imbalance thrust distance.

## Données et formalisation

- **Indicateurs cités** : Bid-Ask Ratio, Cumulative Delta, Volume, Tick Count
- **Horizons cités** : 1min, 5min, 15min
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Vérifier la nécessité de données tick, carnet ou ordres; pas de substitution OHLCV implicite.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
