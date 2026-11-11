# Cumulative Delta Divergence

- **ID** : `cumulative-delta-divergence`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `advanced-order-flow`
- **Famille de recherche (provisoire)** : `order_flow`
- **Rôle envisagé (provisoire)** : `data_gated_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/cumulative-delta-divergence) · [page MNQ](https://pinescriptforge.com/mnq/cumulative-delta-divergence/backtest)

## Description publiée

Trades divergences between cumulative delta and price for order flow reversal signals.

- **Entrée annoncée** : Enter short when price makes higher high but cum delta makes lower high (bearish divergence). Long on bullish divergence (price lower low, delta higher low).
- **Sortie annoncée** : Target: prior swing low/high. Stop above/below the divergence pivot. Confirm with declining aggressor volume.

## Données et formalisation

- **Indicateurs cités** : Cumulative Delta, Price Swing, Volume, Aggressor Ratio
- **Horizons cités** : 5min, 15min, 1H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Vérifier la nécessité de données tick, carnet ou ordres; pas de substitution OHLCV implicite.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
