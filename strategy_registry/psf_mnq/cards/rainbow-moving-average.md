# Rainbow Moving Average

- **ID** : `rainbow-moving-average`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `trend`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/rainbow-moving-average) · [page MNQ](https://pinescriptforge.com/mnq/rainbow-moving-average/backtest)

## Description publiée

Stacks 10 SMAs to create a visual rainbow for trend identification.

- **Entrée annoncée** : Enter long when all 10 rainbow layers are in bullish order (shortest on top) and price pulls back to the 3rd layer. Enter short in bearish order with pullback to 3rd layer.
- **Sortie annoncée** : Exit when the first 3 layers cross through the middle layers. Trail at the 5th rainbow layer.

## Données et formalisation

- **Indicateurs cités** : Rainbow SMA 2-20, Price Action
- **Horizons cités** : 15min, 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).
- **Nom partagé avec** : `rainbow-ma`. Dédupliquer la logique avant expérimentation.

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
