# RSI(2) Pullback

- **ID** : `rsi-2-pullback`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `mean-reversion`
- **Famille de recherche (provisoire)** : `mean_reversion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/rsi-2-pullback) · [page MNQ](https://pinescriptforge.com/mnq/rsi-2-pullback/backtest)

## Description publiée

Ultra-short RSI period for aggressive mean reversion on short-term pullbacks.

- **Entrée annoncée** : Enter long when RSI(2) drops below 10 and price is above 200 SMA. Enter short when RSI(2) exceeds 90 and price is below 200 SMA.
- **Sortie annoncée** : Exit when RSI(2) crosses above 65 (long) or below 35 (short). Typical hold time: 1-4 bars.

## Données et formalisation

- **Indicateurs cités** : RSI(2), SMA 200, ATR
- **Horizons cités** : 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
