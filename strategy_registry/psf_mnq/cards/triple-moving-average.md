# Triple Moving Average

- **ID** : `triple-moving-average`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `trend`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/triple-moving-average) · [page MNQ](https://pinescriptforge.com/mnq/triple-moving-average/backtest)

## Description publiée

Three moving average alignment system for trend confirmation and entry timing.

- **Entrée annoncée** : Enter long when 10 SMA > 20 SMA > 50 SMA and price pulls back to the 20 SMA with a bullish candle. Opposite for shorts.
- **Sortie annoncée** : Exit when the 10 SMA crosses below the 20 SMA. Trail stop at the 50 SMA.

## Données et formalisation

- **Indicateurs cités** : SMA 10, SMA 20, SMA 50
- **Horizons cités** : 15min, 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
