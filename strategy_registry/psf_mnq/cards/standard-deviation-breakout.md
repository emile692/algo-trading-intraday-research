# Standard Deviation Breakout

- **ID** : `standard-deviation-breakout`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `volatility`
- **Famille de recherche (provisoire)** : `volatility_regime`
- **Rôle envisagé (provisoire)** : `feature_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/standard-deviation-breakout) · [page MNQ](https://pinescriptforge.com/mnq/standard-deviation-breakout/backtest)

## Description publiée

Enters when price moves beyond 2 standard deviations, signaling unusual activity.

- **Entrée annoncée** : Enter long when the current bar's range exceeds 2σ of the 20-period return distribution to the upside. Short on 2σ downside move.
- **Sortie annoncée** : Trail at 1σ from the high/low. Exit if the next bar fails to follow through. Stop at 0.5σ inside the breakout bar.

## Données et formalisation

- **Indicateurs cités** : Standard Deviation, Return Distribution, ATR
- **Horizons cités** : 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
