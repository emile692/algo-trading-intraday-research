# Sweeping the Book

- **ID** : `sweeping-the-book`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `advanced-order-flow`
- **Famille de recherche (provisoire)** : `order_flow`
- **Rôle envisagé (provisoire)** : `data_gated_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/sweeping-the-book) · [page MNQ](https://pinescriptforge.com/mnq/sweeping-the-book/backtest)

## Description publiée

Detects large sweeping orders that clear multiple price levels for momentum continuation.

- **Entrée annoncée** : Enter in sweep direction when a single order clears > 5 price levels. Require follow-through volume within 2 bars of the sweep.
- **Sortie annoncée** : Trail stop at the sweep origin price. Target: 2× sweep distance from sweep end. Exit if volume dries up within 5 bars.

## Données et formalisation

- **Indicateurs cités** : Sweep Detection, Level 2 Data, Volume, Price Levels Cleared
- **Horizons cités** : 1min, 5min
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Vérifier la nécessité de données tick, carnet ou ordres; pas de substitution OHLCV implicite.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
