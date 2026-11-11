# Three Bar Reversal

- **ID** : `three-bar-reversal`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `mean-reversion`
- **Famille de recherche (provisoire)** : `mean_reversion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/three-bar-reversal) · [page MNQ](https://pinescriptforge.com/mnq/three-bar-reversal/backtest)

## Description publiée

Classic three-bar price pattern for catching short-term reversals.

- **Entrée annoncée** : Bullish: bar 2 makes a lower low than bar 1, bar 3 closes above bar 1's high. Enter long on bar 3 close. Opposite for bearish setup.
- **Sortie annoncée** : Target 1.5× the pattern range. Stop below bar 2's low (bullish) or above bar 2's high (bearish).

## Données et formalisation

- **Indicateurs cités** : Candlestick Pattern, ATR, Volume
- **Horizons cités** : 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
