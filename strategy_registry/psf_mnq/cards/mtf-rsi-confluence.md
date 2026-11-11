# MTF RSI Confluence

- **ID** : `mtf-rsi-confluence`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `multi-timeframe`
- **Famille de recherche (provisoire)** : `composite`
- **Rôle envisagé (provisoire)** : `meta_signal`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/mtf-rsi-confluence) · [page MNQ](https://pinescriptforge.com/mnq/mtf-rsi-confluence/backtest)

## Description publiée

RSI alignment across multiple timeframes for overbought/oversold setups.

- **Entrée annoncée** : Enter long when Daily RSI < 35, 4H RSI < 30, and 1H RSI hooks up from below 25. Short when all three show overbought.
- **Sortie annoncée** : Exit when 4H RSI crosses 50. Take partial at 1H RSI reaching 60. Stop at the recent 1H swing low.

## Données et formalisation

- **Indicateurs cités** : RSI (Daily), RSI (4H), RSI (1H)
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
