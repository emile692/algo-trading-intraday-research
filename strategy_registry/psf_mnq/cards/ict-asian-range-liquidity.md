# ICT Asian Range Liquidity

- **ID** : `ict-asian-range-liquidity`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `smart-money`
- **Famille de recherche (provisoire)** : `price_action_reversal`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/ict-asian-range-liquidity) · [page MNQ](https://pinescriptforge.com/mnq/ict-asian-range-liquidity/backtest)

## Description publiée

Uses the Asian session range as a liquidity pool for London/NY session entries.

- **Entrée annoncée** : Enter on reversal after Asian range high/low is swept during London or NY open.
- **Sortie annoncée** : Target the opposite Asian range extreme. Stop beyond the sweep wick.

## Données et formalisation

- **Indicateurs cités** : Asian Range H/L, Session Times, FVG
- **Horizons cités** : 5min, 15min
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Calendrier de séance et horodatage New York à définir.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
