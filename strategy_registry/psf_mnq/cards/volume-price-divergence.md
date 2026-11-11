# Volume-Price Divergence

- **ID** : `volume-price-divergence`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `divergence`
- **Famille de recherche (provisoire)** : `reversal_exhaustion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/volume-price-divergence) · [page MNQ](https://pinescriptforge.com/mnq/volume-price-divergence/backtest)

## Description publiée

Trades when price makes new extremes on declining volume — a classic exhaustion signal.

- **Entrée annoncée** : Short when price makes new 20-bar high but volume is below its 20-bar SMA. Long on new low with declining volume.
- **Sortie annoncée** : Target the 20-bar mean price. Stop beyond the divergence extreme. Risk 1× ATR.

## Données et formalisation

- **Indicateurs cités** : Volume, Volume SMA, Price Highs/Lows
- **Horizons cités** : 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
