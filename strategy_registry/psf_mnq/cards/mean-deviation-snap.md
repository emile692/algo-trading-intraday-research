# Mean Deviation Snap

- **ID** : `mean-deviation-snap`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `mean-reversion`
- **Famille de recherche (provisoire)** : `mean_reversion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/mean-deviation-snap) · [page MNQ](https://pinescriptforge.com/mnq/mean-deviation-snap/backtest)

## Description publiée

Trades snap-back moves when price deviates >2σ from its rolling mean.

- **Entrée annoncée** : Enter long when z-score < -2.0. Short when z-score > +2.0. Scale in at -2.5/+2.5.
- **Sortie annoncée** : Exit when z-score returns to ±0.5. Stop at z-score ±3.0.

## Données et formalisation

- **Indicateurs cités** : Z-Score, Rolling Mean, Standard Deviation
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
