# Spinning Top Reversal

- **ID** : `spinning-top-reversal`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `price-action`
- **Famille de recherche (provisoire)** : `price_action`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/spinning-top-reversal) · [page MNQ](https://pinescriptforge.com/mnq/spinning-top-reversal/backtest)

## Description publiée

Trades reversals when spinning top candles appear at support/resistance levels.

- **Entrée annoncée** : Enter reversal direction when spinning top forms at support (long) or resistance (short) with volume decline.
- **Sortie annoncée** : Target the opposite S/R level. Stop beyond the spinning top's range.

## Données et formalisation

- **Indicateurs cités** : Candlestick Pattern, Support/Resistance, Volume
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
