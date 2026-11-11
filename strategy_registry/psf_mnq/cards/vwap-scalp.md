# VWAP Scalp

- **ID** : `vwap-scalp`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `scalping`
- **Famille de recherche (provisoire)** : `short_horizon`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/vwap-scalp) · [page MNQ](https://pinescriptforge.com/mnq/vwap-scalp/backtest)

## Description publiée

Ultra-fast scalps around VWAP with tight risk on 1-5 minute charts.

- **Entrée annoncée** : Enter long when price touches -1σ VWAP band on a 1-min chart with a bullish reversal candle. Short at +1σ band.
- **Sortie annoncée** : Target VWAP for full exit. Stop at -1.5σ. Maximum hold time: 10 minutes. Close all positions before session end.

## Données et formalisation

- **Indicateurs cités** : VWAP, VWAP Bands, Tick Chart, Volume Delta
- **Horizons cités** : 1min, 5min
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Calendrier de séance et horodatage New York à définir.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
