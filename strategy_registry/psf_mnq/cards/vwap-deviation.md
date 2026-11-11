# VWAP Deviation

- **ID** : `vwap-deviation`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `mean-reversion`
- **Famille de recherche (provisoire)** : `mean_reversion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/vwap-deviation) · [page MNQ](https://pinescriptforge.com/mnq/vwap-deviation/backtest)

## Description publiée

Trades reversals at standard deviation bands around VWAP.

- **Entrée annoncée** : Enter long at -2σ VWAP band with confirmation candle. Enter short at +2σ VWAP band.
- **Sortie annoncée** : Target VWAP for full reversion. Partial exit at -1σ/+1σ. Stop at -3σ/+3σ.

## Données et formalisation

- **Indicateurs cités** : VWAP, VWAP StdDev Bands, Volume
- **Horizons cités** : 5min, 15min, 1H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
