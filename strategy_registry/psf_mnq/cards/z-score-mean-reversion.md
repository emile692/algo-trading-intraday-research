# Z-Score Mean Reversion

- **ID** : `z-score-mean-reversion`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `mean-reversion`
- **Famille de recherche (provisoire)** : `mean_reversion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/z-score-mean-reversion) · [page MNQ](https://pinescriptforge.com/mnq/z-score-mean-reversion/backtest)

## Description publiée

Statistical z-score for identifying prices significantly away from the mean.

- **Entrée annoncée** : Enter long when z-score drops below -2.0 (price 2σ below mean). Enter short when z-score rises above +2.0.
- **Sortie annoncée** : Exit when z-score reverts to ±0.5. Stop at z-score ±3.0. Partial profit at z-score ±1.0.

## Données et formalisation

- **Indicateurs cités** : Z-Score, SMA 20, Standard Deviation
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).
- **Nom partagé avec** : `z-score-reversion`. Dédupliquer la logique avant expérimentation.

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
