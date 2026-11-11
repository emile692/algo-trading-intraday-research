# MACD Histogram Divergence

- **ID** : `macd-histogram-divergence`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `momentum`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/macd-histogram-divergence) · [page MNQ](https://pinescriptforge.com/mnq/macd-histogram-divergence/backtest)

## Description publiée

Detects momentum shifts through MACD histogram divergence before price confirms.

- **Entrée annoncée** : Enter short when price makes a higher high but MACD histogram makes a lower high (bearish divergence), confirmed by a histogram bar turning negative. Enter long on bullish divergence (lower low in price, higher low in histogram).
- **Sortie annoncée** : Target the prior swing low (bearish) or swing high (bullish). Stop above the divergence high/low. Trail stop when histogram crosses zero in your favor.

## Données et formalisation

- **Indicateurs cités** : MACD Histogram, Divergence Detection, Swing High/Low
- **Horizons cités** : 15min, 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
