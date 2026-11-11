# ICT Order Block

- **ID** : `ict-order-block`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `smart-money`
- **Famille de recherche (provisoire)** : `price_action_reversal`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/ict-order-block) · [page MNQ](https://pinescriptforge.com/mnq/ict-order-block/backtest)

## Description publiée

Enters at institutional order blocks — the last bearish candle before a bullish impulse (and vice versa).

- **Entrée annoncée** : Identify the last down-close candle before a strong bullish impulse (bullish OB). Enter long when price returns to the 50–75% level of that candle with a rejection pattern. Opposite for bearish OBs.
- **Sortie annoncée** : Target the high of the impulse that created the OB, or the next liquidity pool. Stop below the OB low (bullish) or above OB high (bearish). Aim for 2–3R.

## Données et formalisation

- **Indicateurs cités** : Order Block Detection, Higher Timeframe Structure, Market Structure Shift
- **Horizons cités** : 5min, 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
