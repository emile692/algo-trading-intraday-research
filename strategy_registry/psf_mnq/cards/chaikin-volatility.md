# Chaikin Volatility

- **ID** : `chaikin-volatility`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `volatility`
- **Famille de recherche (provisoire)** : `volatility_regime`
- **Rôle envisagé (provisoire)** : `feature_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/chaikin-volatility) · [page MNQ](https://pinescriptforge.com/mnq/chaikin-volatility/backtest)

## Description publiée

Measures the rate of change of the trading range to detect volatility shifts.

- **Entrée annoncée** : Enter when Chaikin Volatility reverses from a low and starts rising (volatility expansion beginning). Direction based on price momentum at the time.
- **Sortie annoncée** : Exit when Chaikin Volatility peaks and begins declining (volatility contraction starting). Trail at 2× ATR.

## Données et formalisation

- **Indicateurs cités** : Chaikin Volatility, ATR, Momentum
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
