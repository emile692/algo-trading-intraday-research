# RSI + Bollinger Band Combo

- **ID** : `rsi-bollinger-band-combo`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `mean-reversion`
- **Famille de recherche (provisoire)** : `mean_reversion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/rsi-bollinger-band-combo) · [page MNQ](https://pinescriptforge.com/mnq/rsi-bollinger-band-combo/backtest)

## Description publiée

Dual confirmation from RSI extremes and Bollinger Band touches for high-probability reversals.

- **Entrée annoncée** : Enter long when price touches or closes below the lower Bollinger Band AND RSI < 30 simultaneously. Enter short when price touches the upper band AND RSI > 70. Additional confirmation: wait for first candle to close back inside the bands.
- **Sortie annoncée** : Target the Bollinger Band middle line (20 SMA). Stop 1× ATR beyond the band. Scale out: 50% at middle band, 50% at opposing band.

## Données et formalisation

- **Indicateurs cités** : RSI, Bollinger Bands, SMA 20, ATR
- **Horizons cités** : 15min, 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
