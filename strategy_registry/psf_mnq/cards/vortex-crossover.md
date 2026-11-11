# Vortex Indicator Crossover

- **ID** : `vortex-crossover`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `trend`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/vortex-crossover) · [page MNQ](https://pinescriptforge.com/mnq/vortex-crossover/backtest)

## Description publiée

Captures trend direction using positive and negative vortex lines.

- **Entrée annoncée** : Enter long when VI+ crosses above VI- and both lines are above 1.0. Enter short on the opposite crossover. Require ATR expansion for confirmation.
- **Sortie annoncée** : Exit when VI lines cross against position or converge within 0.05 of each other.

## Données et formalisation

- **Indicateurs cités** : Vortex VI+ 21, Vortex VI- 21, ATR 14
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
