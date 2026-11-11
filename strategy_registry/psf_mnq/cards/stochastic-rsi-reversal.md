# Stochastic RSI Reversal

- **ID** : `stochastic-rsi-reversal`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `mean-reversion`
- **Famille de recherche (provisoire)** : `mean_reversion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/stochastic-rsi-reversal) · [page MNQ](https://pinescriptforge.com/mnq/stochastic-rsi-reversal/backtest)

## Description publiée

Combines Stochastic and RSI for ultra-sensitive overbought/oversold detection.

- **Entrée annoncée** : Enter long when StochRSI %K crosses above %D below 0.10. Enter short when %K crosses below %D above 0.90.
- **Sortie annoncée** : Exit long when StochRSI reaches 0.80. Exit short at 0.20. Stop at the recent swing extreme.

## Données et formalisation

- **Indicateurs cités** : Stochastic RSI %K, Stochastic RSI %D
- **Horizons cités** : 5min, 15min, 1H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
