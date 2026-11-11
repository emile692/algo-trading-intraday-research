# Range Midpoint Reversion

- **ID** : `range-midpoint-reversion`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `range`
- **Famille de recherche (provisoire)** : `breakout_expansion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/range-midpoint-reversion) · [page MNQ](https://pinescriptforge.com/mnq/range-midpoint-reversion/backtest)

## Description publiée

Trades mean reversion to the midpoint of established trading ranges.

- **Entrée annoncée** : Enter long at the bottom 25% of a 20-bar range. Enter short at the top 25%. Confirm with RSI extreme.
- **Sortie annoncée** : Target the range midpoint. Stop 1 tick beyond the range boundary. Trail at midpoint.

## Données et formalisation

- **Indicateurs cités** : Donchian Channel, Range Midpoint, RSI
- **Horizons cités** : 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
