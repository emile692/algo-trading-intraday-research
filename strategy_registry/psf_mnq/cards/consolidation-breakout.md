# Consolidation Breakout

- **ID** : `consolidation-breakout`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `range`
- **Famille de recherche (provisoire)** : `breakout_expansion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/consolidation-breakout) · [page MNQ](https://pinescriptforge.com/mnq/consolidation-breakout/backtest)

## Description publiée

Measures consolidation tightness then trades the breakout direction.

- **Entrée annoncée** : Enter when price closes outside a range that has contracted for 10+ bars. Direction of close = trade direction.
- **Sortie annoncée** : Target 1× the prior range height projected from breakout. Stop at range midpoint.

## Données et formalisation

- **Indicateurs cités** : Bollinger Bandwidth, Range Detection, Volume
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
