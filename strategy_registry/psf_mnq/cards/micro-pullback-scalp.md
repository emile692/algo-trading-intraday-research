# Micro Pullback Scalp

- **ID** : `micro-pullback-scalp`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `scalping`
- **Famille de recherche (provisoire)** : `short_horizon`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/micro-pullback-scalp) · [page MNQ](https://pinescriptforge.com/mnq/micro-pullback-scalp/backtest)

## Description publiée

Scalps tiny pullbacks within strong intraday trends for quick continuation trades.

- **Entrée annoncée** : Enter long on pullback to 9 EMA when slope is steeply positive and price hasn't closed below it. Short on steep negative slope pullbacks.
- **Sortie annoncée** : Target 6-10 ticks from entry. Stop at 2× the pullback depth below the 9 EMA. Trail at each new micro swing low.

## Données et formalisation

- **Indicateurs cités** : EMA 9, EMA Slope, ATR, Tick Chart
- **Horizons cités** : 1min, 5min
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
