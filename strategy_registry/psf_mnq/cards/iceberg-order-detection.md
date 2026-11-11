# Iceberg Order Detection

- **ID** : `iceberg-order-detection`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `microstructure`
- **Famille de recherche (provisoire)** : `order_flow`
- **Rôle envisagé (provisoire)** : `data_gated_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/iceberg-order-detection) · [page MNQ](https://pinescriptforge.com/mnq/iceberg-order-detection/backtest)

## Description publiée

Identifies hidden institutional iceberg orders to trade alongside smart money.

- **Entrée annoncée** : Enter long when iceberg buy detected (repeated refills at bid) with rising price. Short on iceberg sell detection at ask.
- **Sortie annoncée** : Exit when iceberg order is fully consumed (no more refills). Target 1× daily ATR from entry. Stop 5 ticks beyond iceberg level.

## Données et formalisation

- **Indicateurs cités** : Order Flow, Iceberg Detection, Cumulative Delta, Volume
- **Horizons cités** : 1min, 5min
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Vérifier la nécessité de données tick, carnet ou ordres; pas de substitution OHLCV implicite.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
