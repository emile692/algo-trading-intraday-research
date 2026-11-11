# RSI Mean Reversion

- **ID** : `rsi-mean-reversion`
- **Univers** : `supplemental_page`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `mean-reversion`
- **Famille de recherche (provisoire)** : `mean_reversion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/rsi-mean-reversion) · [page MNQ](https://pinescriptforge.com/mnq/rsi-mean-reversion/backtest)

## Description publiée

Buys oversold and sells overbought RSI extremes for mean reversion trades.

- **Entrée annoncée** : Enter long when RSI drops below 30 and price is above the 200 SMA. Enter short when RSI rises above 70 and price is below the 200 SMA.
- **Sortie annoncée** : Exit long when RSI crosses above 50. Exit short when RSI crosses below 50. Hard stop at 1.5× ATR.

## Données et formalisation

- **Indicateurs cités** : RSI, SMA 200, ATR
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).
- **Hors filtre du classement courant** : page MNQ publiée, ajoutée pour compléter les 185 fiches demandées.

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
