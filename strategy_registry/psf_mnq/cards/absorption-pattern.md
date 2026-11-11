# Absorption Pattern

- **ID** : `absorption-pattern`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `advanced-order-flow`
- **Famille de recherche (provisoire)** : `order_flow`
- **Rôle envisagé (provisoire)** : `data_gated_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/absorption-pattern) · [page MNQ](https://pinescriptforge.com/mnq/absorption-pattern/backtest)

## Description publiée

Detects price levels where limit orders absorb aggressive orders, signaling support/resistance.

- **Entrée annoncée** : Enter long at absorption support (high buy volume, price not falling). Short at absorption resistance (high sell volume, price not rising).
- **Sortie annoncée** : Target: VWAP or prior session POC. Stop if price trades through absorption level by > 3 ticks (absorption failed).

## Données et formalisation

- **Indicateurs cités** : Absorption Ratio, Volume, Price Progress, Limit Order Flow
- **Horizons cités** : 1min, 5min, 15min
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Calendrier de séance et horodatage New York à définir.
- **Donnée à vérifier** : Vérifier la nécessité de données tick, carnet ou ordres; pas de substitution OHLCV implicite.
- **Donnée à vérifier** : Distribution volume-prix nécessaire pour reproduire un vrai profil de marché.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
