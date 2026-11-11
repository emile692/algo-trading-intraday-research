# Quad Witching Volatility

- **ID** : `quad-witching`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `seasonal`
- **Famille de recherche (provisoire)** : `calendar`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/quad-witching) · [page MNQ](https://pinescriptforge.com/mnq/quad-witching/backtest)

## Description publiée

Trades elevated volatility during quarterly options expiration.

- **Entrée annoncée** : Buy straddle-equivalent positions (long ATR breakout) 2 days before quad witching. Trade breakouts in either direction using 0.5× ATR entries.
- **Sortie annoncée** : Exit on the close of quad witching Friday. Trail stops at 1× ATR. Capture at least 1 full ATR move in either direction.

## Données et formalisation

- **Indicateurs cités** : Expiration Calendar, ATR, Volume, Open Interest
- **Horizons cités** : 15min, 1H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Calendrier de séance et horodatage New York à définir.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
