# Fisher Transform

- **ID** : `fisher-transform`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `mean-reversion`
- **Famille de recherche (provisoire)** : `mean_reversion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/fisher-transform) · [page MNQ](https://pinescriptforge.com/mnq/fisher-transform/backtest)

## Description publiée

Transforms price into a Gaussian distribution for sharper turning point signals.

- **Entrée annoncée** : Enter long when Fisher Transform reverses from below -1.5. Enter short when it reverses from above +1.5.
- **Sortie annoncée** : Exit when Fisher Transform crosses zero. Stop at the extreme Fisher reading level.

## Données et formalisation

- **Indicateurs cités** : Fisher Transform, Fisher Signal
- **Horizons cités** : 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
