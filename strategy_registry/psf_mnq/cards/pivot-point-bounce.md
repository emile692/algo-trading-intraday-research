# Pivot Point Bounce

- **ID** : `pivot-point-bounce`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `mean-reversion`
- **Famille de recherche (provisoire)** : `mean_reversion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/pivot-point-bounce) · [page MNQ](https://pinescriptforge.com/mnq/pivot-point-bounce/backtest)

## Description publiée

Trades bounces and rejections at daily, weekly, and monthly pivot point levels.

- **Entrée annoncée** : Enter long when price touches S1 or S2 with a bullish rejection candle and RSI < 40. Enter short at R1 or R2 with bearish rejection and RSI > 60. Double size at confluence of daily + weekly pivots.
- **Sortie annoncée** : Target the next pivot level (S1 to PP, PP to R1, etc.). Stop 0.5× ATR beyond the pivot level. Partial exit at the central pivot if entering from S1/R1.

## Données et formalisation

- **Indicateurs cités** : Daily Pivot Points, Weekly Pivot Points, RSI, ATR
- **Horizons cités** : 5min, 15min, 1H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
