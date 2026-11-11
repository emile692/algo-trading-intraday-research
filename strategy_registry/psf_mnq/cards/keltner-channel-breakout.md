# Keltner Channel Breakout

- **ID** : `keltner-channel-breakout`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `breakout`
- **Famille de recherche (provisoire)** : `breakout_expansion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/keltner-channel-breakout) · [page MNQ](https://pinescriptforge.com/mnq/keltner-channel-breakout/backtest)

## Description publiée

ATR-based Keltner Channels for volatility breakout trades.

- **Entrée annoncée** : Enter long on close above upper Keltner Channel. Enter short on close below lower channel.
- **Sortie annoncée** : Exit when price closes back inside the channel. Trailing stop at the channel midline (EMA 20).

## Données et formalisation

- **Indicateurs cités** : Keltner Channel, EMA 20, ATR
- **Horizons cités** : 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
