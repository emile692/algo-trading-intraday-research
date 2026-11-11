# Market Structure Shift (MSS)

- **ID** : `market-structure-shift`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `smart-money`
- **Famille de recherche (provisoire)** : `price_action_reversal`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/market-structure-shift) · [page MNQ](https://pinescriptforge.com/mnq/market-structure-shift/backtest)

## Description publiée

Enters on confirmed market structure breaks that signal a new trend direction.

- **Entrée annoncée** : In uptrend: enter short when price closes below the most recent higher low on a strong bearish displacement candle. In downtrend: enter long on break above the most recent lower high.
- **Sortie annoncée** : Target the previous swing high/low of the prior trend. Stop above/below the MSS candle high/low. Aim for 3R minimum as these mark full trend reversals.

## Données et formalisation

- **Indicateurs cités** : Market Structure, Swing High/Low Detection, Displacement Candles
- **Horizons cités** : 15min, 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
