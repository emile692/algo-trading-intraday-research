# Volatility Breakout

- **ID** : `volatility-breakout`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `breakout`
- **Famille de recherche (provisoire)** : `breakout_expansion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/volatility-breakout) · [page MNQ](https://pinescriptforge.com/mnq/volatility-breakout/backtest)

## Description publiée

Enters when price moves more than a threshold ATR amount from the open.

- **Entrée annoncée** : Enter long when price moves above Open + (0.6 × previous day ATR). Short when price drops below Open - (0.6 × ATR).
- **Sortie annoncée** : Exit at the session close (day trade) or trail at 1× ATR. Stop at the open price.

## Données et formalisation

- **Indicateurs cités** : ATR, Session Open, Range
- **Horizons cités** : 5min, 15min, 1H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Calendrier de séance et horodatage New York à définir.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
