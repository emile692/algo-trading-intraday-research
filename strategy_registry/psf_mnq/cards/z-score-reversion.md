# Z-Score Mean Reversion

- **ID** : `z-score-reversion`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `mean-reversion`
- **Famille de recherche (provisoire)** : `mean_reversion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/z-score-reversion) · [page MNQ](https://pinescriptforge.com/mnq/z-score-reversion/backtest)

## Description publiée

Statistical z-score of price for mathematically precise mean reversion entries.

- **Entrée annoncée** : Enter long when price z-score drops below -2.0. Enter short when z-score exceeds +2.0. Use 50-period lookback.
- **Sortie annoncée** : Exit when z-score returns to ±0.5. Stop at z-score ±3.0. Position size inversely proportional to z-score magnitude.

## Données et formalisation

- **Indicateurs cités** : Z-Score, Rolling Mean, Rolling StdDev
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).
- **Nom partagé avec** : `z-score-mean-reversion`. Dédupliquer la logique avant expérimentation.

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
