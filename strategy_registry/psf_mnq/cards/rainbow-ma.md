# Rainbow Moving Average

- **ID** : `rainbow-ma`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `trend`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/rainbow-ma) · [page MNQ](https://pinescriptforge.com/mnq/rainbow-ma/backtest)

## Description publiée

10-layer moving average rainbow for visual trend strength and entry timing.

- **Entrée annoncée** : Enter long when rainbow layers are bullish-ordered and price touches the 3rd or 4th layer. Enter short on bearish-ordered rainbow with pullback to layers.
- **Sortie annoncée** : Exit when the rainbow layers begin to converge or cross. Stop below the 7th layer.

## Données et formalisation

- **Indicateurs cités** : Rainbow MA (10 layers), SMA, ATR
- **Horizons cités** : 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).
- **Nom partagé avec** : `rainbow-moving-average`. Dédupliquer la logique avant expérimentation.

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
