# Bollinger Band Walk

- **ID** : `bollinger-band-walk`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `mean-reversion`
- **Famille de recherche (provisoire)** : `mean_reversion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/bollinger-band-walk) · [page MNQ](https://pinescriptforge.com/mnq/bollinger-band-walk/backtest)

## Description publiée

Fades extended Bollinger Band walks when momentum exhaustion signals appear.

- **Entrée annoncée** : After 5+ closes beyond the upper/lower Bollinger Band, enter counter-trend when RSI shows divergence and volume declines 30%+.
- **Sortie annoncée** : Target the 20 SMA (middle band). Stop above/below the most extreme band walk candle.

## Données et formalisation

- **Indicateurs cités** : Bollinger Bands, RSI, Volume
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
