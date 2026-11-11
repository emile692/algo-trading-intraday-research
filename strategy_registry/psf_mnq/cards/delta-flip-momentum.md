# Delta Flip Momentum

- **ID** : `delta-flip-momentum`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `advanced-order-flow`
- **Famille de recherche (provisoire)** : `order_flow`
- **Rôle envisagé (provisoire)** : `data_gated_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/delta-flip-momentum) · [page MNQ](https://pinescriptforge.com/mnq/delta-flip-momentum/backtest)

## Description publiée

Enters on cumulative delta sign flips as early momentum shift signals.

- **Entrée annoncée** : Enter long when session cumulative delta flips from negative to positive with rising volume. Short on positive-to-negative flip.
- **Sortie annoncée** : Trail at prior delta flip level. Stop at 1× ATR below entry. Exit on opposing delta flip or at session close.

## Données et formalisation

- **Indicateurs cités** : Cumulative Delta, Delta Flip Point, Volume, ATR
- **Horizons cités** : 5min, 15min, 30min
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Calendrier de séance et horodatage New York à définir.
- **Donnée à vérifier** : Vérifier la nécessité de données tick, carnet ou ordres; pas de substitution OHLCV implicite.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
