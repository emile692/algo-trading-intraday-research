# Box Range Trading

- **ID** : `box-range-trading`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `range`
- **Famille de recherche (provisoire)** : `breakout_expansion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/box-range-trading) · [page MNQ](https://pinescriptforge.com/mnq/box-range-trading/backtest)

## Description publiée

Defines price boxes and trades bounces within and breakouts from the box.

- **Entrée annoncée** : Buy at box bottom, sell at box top for range trades. Breakout: buy on close above box top with volume.
- **Sortie annoncée** : Range: target opposite boundary. Breakout: target box height projected above. Stop inside box.

## Données et formalisation

- **Indicateurs cités** : Darvas Box, Volume, ATR
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
