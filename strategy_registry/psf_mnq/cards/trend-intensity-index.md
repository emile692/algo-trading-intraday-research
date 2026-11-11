# Trend Intensity Index

- **ID** : `trend-intensity-index`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `trend`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/trend-intensity-index) · [page MNQ](https://pinescriptforge.com/mnq/trend-intensity-index/backtest)

## Description publiée

Quantifies trend strength by measuring price deviation from its moving average.

- **Entrée annoncée** : Enter long when TII crosses above 80 with price above 50 SMA. Enter short when TII crosses below 20 with price below 50 SMA.
- **Sortie annoncée** : Exit when TII crosses the 50 line against position. Trail stop at 1.5× ATR.

## Données et formalisation

- **Indicateurs cités** : Trend Intensity Index 30, SMA 50, ATR 14
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
