# ICT Change of Character (CHoCH)

- **ID** : `ict-change-of-character`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `smart-money`
- **Famille de recherche (provisoire)** : `price_action_reversal`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/ict-change-of-character) · [page MNQ](https://pinescriptforge.com/mnq/ict-change-of-character/backtest)

## Description publiée

Detects the first structural break that signals a potential trend reversal — the Change of Character.

- **Entrée annoncée** : In an uptrend, enter short when price closes below the most recent higher low (CHoCH to the downside). In a downtrend, enter long when price closes above the most recent lower high (CHoCH to the upside). Requires displacement candle confirmation.
- **Sortie annoncée** : Target the origin of the previous impulse move or the nearest untested Order Block. Stop above the CHoCH swing high/low. Minimum 2.5R target.

## Données et formalisation

- **Indicateurs cités** : Swing Structure, CHoCH Detection, Displacement Candles, Order Blocks
- **Horizons cités** : 5min, 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
