# Linear Regression Channel

- **ID** : `regression-channel`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `mean-reversion`
- **Famille de recherche (provisoire)** : `mean_reversion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/regression-channel) · [page MNQ](https://pinescriptforge.com/mnq/regression-channel/backtest)

## Description publiée

Trades reversals at the boundaries of a linear regression channel.

- **Entrée annoncée** : Enter long at the lower channel boundary (-2σ from regression line). Enter short at the upper channel (+2σ). Confirm with momentum reversal.
- **Sortie annoncée** : Target the regression line (midpoint). Trail stop at ±1σ. Stop beyond ±3σ.

## Données et formalisation

- **Indicateurs cités** : Linear Regression, Regression Channel, StdDev Bands
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
