# Standard Deviation Bands

- **ID** : `std-dev-bands-reversion`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `mean-reversion`
- **Famille de recherche (provisoire)** : `mean_reversion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/std-dev-bands-reversion) · [page MNQ](https://pinescriptforge.com/mnq/std-dev-bands-reversion/backtest)

## Description publiée

Trades price reversions from 2σ and 3σ statistical bands.

- **Entrée annoncée** : Enter long at -2σ band touch with target at the mean (SMA). Aggressive entry at -3σ with double position. Enter short at +2σ and +3σ respectively.
- **Sortie annoncée** : Primary exit at the 20-period SMA (mean). Partial at ±1σ. Stop at 3.5σ.

## Données et formalisation

- **Indicateurs cités** : SMA 20, 1σ Bands, 2σ Bands, 3σ Bands
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
