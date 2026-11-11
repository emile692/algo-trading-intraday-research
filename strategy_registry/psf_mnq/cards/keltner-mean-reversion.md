# Keltner Mean Reversion

- **ID** : `keltner-mean-reversion`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `mean-reversion`
- **Famille de recherche (provisoire)** : `mean_reversion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/keltner-mean-reversion) · [page MNQ](https://pinescriptforge.com/mnq/keltner-mean-reversion/backtest)

## Description publiée

Fades moves to Keltner Channel extremes for mean reversion back to the EMA.

- **Entrée annoncée** : Enter long when price touches lower Keltner band and RSI < 35. Short at upper band with RSI > 65.
- **Sortie annoncée** : Target the EMA 20 midline. Stop 1× ATR beyond the channel. Partial profit at half-distance.

## Données et formalisation

- **Indicateurs cités** : Keltner Channel, EMA 20, RSI
- **Horizons cités** : 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
