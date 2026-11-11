# CCI Momentum

- **ID** : `cci-momentum`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `momentum`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/cci-momentum) · [page MNQ](https://pinescriptforge.com/mnq/cci-momentum/backtest)

## Description publiée

Commodity Channel Index for momentum detection and overbought/oversold extremes.

- **Entrée annoncée** : Enter long when CCI crosses above +100 with price above 50 EMA. Enter short when CCI crosses below -100 with price below 50 EMA.
- **Sortie annoncée** : Exit long when CCI drops below +100 from above +200. Exit short when CCI rises above -100 from below -200. Stop at 2× ATR.

## Données et formalisation

- **Indicateurs cités** : CCI, EMA 50, ATR
- **Horizons cités** : 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
