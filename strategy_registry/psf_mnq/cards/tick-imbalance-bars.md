# Tick Imbalance Bars

- **ID** : `tick-imbalance-bars`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `microstructure`
- **Famille de recherche (provisoire)** : `order_flow`
- **Rôle envisagé (provisoire)** : `data_gated_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/tick-imbalance-bars) · [page MNQ](https://pinescriptforge.com/mnq/tick-imbalance-bars/backtest)

## Description publiée

Uses information-driven tick imbalance bars for adaptive signal generation.

- **Entrée annoncée** : Enter long when a bullish tick imbalance bar forms above VWAP with expanding bar frequency. Short on bearish imbalance bars below VWAP.
- **Sortie annoncée** : Exit on opposing imbalance bar or after 3 bars without continuation. Trail stop at prior imbalance bar low/high.

## Données et formalisation

- **Indicateurs cités** : Tick Imbalance, VWAP, Bar Frequency, Cumulative Ticks
- **Horizons cités** : Tick, 1min, 5min
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Vérifier la nécessité de données tick, carnet ou ordres; pas de substitution OHLCV implicite.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
