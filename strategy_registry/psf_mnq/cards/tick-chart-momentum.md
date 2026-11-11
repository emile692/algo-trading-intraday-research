# Tick Chart Momentum

- **ID** : `tick-chart-momentum`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `scalping`
- **Famille de recherche (provisoire)** : `short_horizon`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/tick-chart-momentum) · [page MNQ](https://pinescriptforge.com/mnq/tick-chart-momentum/backtest)

## Description publiée

Momentum bursts on tick charts for rapid directional scalps.

- **Entrée annoncée** : Enter long after 3 consecutive bullish tick bars with expanding range. Enter short after 3 bearish bars. Volume must be above session average.
- **Sortie annoncée** : Target 4-8 ticks. Stop at the low of the 3-bar sequence. Exit on first opposing tick bar close.

## Données et formalisation

- **Indicateurs cités** : Tick Chart, Range Expansion, Volume, Momentum
- **Horizons cités** : 1min, 5min
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Calendrier de séance et horodatage New York à définir.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
