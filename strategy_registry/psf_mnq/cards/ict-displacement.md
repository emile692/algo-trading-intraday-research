# ICT Displacement

- **ID** : `ict-displacement`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `smart-money`
- **Famille de recherche (provisoire)** : `price_action_reversal`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/ict-displacement) · [page MNQ](https://pinescriptforge.com/mnq/ict-displacement/backtest)

## Description publiée

Trades aggressive displacement candles that signal strong institutional order flow.

- **Entrée annoncée** : Identify a displacement move (3+ consecutive large candles in one direction). Wait for price to retrace to the FVG or Order Block created by the displacement. Enter with confirmation.
- **Sortie annoncée** : Target the displacement high/low or the next liquidity level. Stop beyond the retracement structure.

## Données et formalisation

- **Indicateurs cités** : Displacement Candles, Fair Value Gap, Order Block
- **Horizons cités** : 5min, 15min, 1H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
