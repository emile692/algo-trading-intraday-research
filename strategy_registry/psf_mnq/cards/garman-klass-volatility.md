# Garman-Klass Volatility

- **ID** : `garman-klass-volatility`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `volatility`
- **Famille de recherche (provisoire)** : `volatility_regime`
- **Rôle envisagé (provisoire)** : `feature_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/garman-klass-volatility) · [page MNQ](https://pinescriptforge.com/mnq/garman-klass-volatility/backtest)

## Description publiée

Advanced OHLC volatility estimator for more accurate volatility regime detection.

- **Entrée annoncée** : Enter breakout when GK volatility begins expanding from a low percentile. Direction based on the first 1× ATR breakout from the compression range.
- **Sortie annoncée** : Exit when GK volatility peaks and begins contracting. Trail at 1.5× ATR.

## Données et formalisation

- **Indicateurs cités** : Garman-Klass Vol, ATR, Percentile Rank
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
