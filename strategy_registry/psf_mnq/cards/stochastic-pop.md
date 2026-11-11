# Stochastic Pop

- **ID** : `stochastic-pop`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `mean-reversion`
- **Famille de recherche (provisoire)** : `mean_reversion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/stochastic-pop) · [page MNQ](https://pinescriptforge.com/mnq/stochastic-pop/backtest)

## Description publiée

Captures quick snapback moves when stochastics pop from oversold extremes.

- **Entrée annoncée** : Enter long when %K pops above 20 from below while price is above the 200 SMA (uptrend filter). Enter short when %K drops below 80 from above while below 200 SMA.
- **Sortie annoncée** : Exit when %K reaches the opposite extreme (80 for longs, 20 for shorts) or after 5 bars. Quick trade.

## Données et formalisation

- **Indicateurs cités** : Stochastic %K 14, Stochastic %D 3, SMA 200
- **Horizons cités** : 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
