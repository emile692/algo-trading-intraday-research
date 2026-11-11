# ICT Killzone

- **ID** : `ict-killzone`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `smart-money`
- **Famille de recherche (provisoire)** : `price_action_reversal`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/ict-killzone) · [page MNQ](https://pinescriptforge.com/mnq/ict-killzone/backtest)

## Description publiée

Enters during high-probability ICT killzone windows at key liquidity levels.

- **Entrée annoncée** : During a killzone, identify the prior session high/low. Wait for price to sweep just beyond it (liquidity grab), then enter aggressively on the reversal candle back through the level.
- **Sortie annoncée** : Target the opposing session high/low or nearest Fair Value Gap. Stop above/below the liquidity sweep wick. Aim for minimum 2R.

## Données et formalisation

- **Indicateurs cités** : Session Highs/Lows, Time-of-Day Filter, Liquidity Sweep Detection
- **Horizons cités** : 1min, 5min, 15min
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Calendrier de séance et horodatage New York à définir.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
