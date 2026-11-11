# Volume Imbalance Stacking

- **ID** : `volume-imbalance-stacking`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `advanced-order-flow`
- **Famille de recherche (provisoire)** : `order_flow`
- **Rôle envisagé (provisoire)** : `data_gated_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/volume-imbalance-stacking) · [page MNQ](https://pinescriptforge.com/mnq/volume-imbalance-stacking/backtest)

## Description publiée

Identifies stacked volume imbalances across price levels for directional bias.

- **Entrée annoncée** : Enter long when 3+ consecutive price levels show buy imbalance > 300% of sell volume. Short on stacked sell imbalances.
- **Sortie annoncée** : Exit when a contrary imbalance appears. Trail stop at the bottom of the imbalance stack. Target 2× stack height.

## Données et formalisation

- **Indicateurs cités** : Volume Imbalance, Footprint Chart, Bid-Ask Volume, ATR
- **Horizons cités** : 1min, 5min, 15min
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Vérifier la nécessité de données tick, carnet ou ordres; pas de substitution OHLCV implicite.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
