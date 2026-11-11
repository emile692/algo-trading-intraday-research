# ICT Optimal Trade Entry (OTE)

- **ID** : `ict-optimal-trade-entry`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `smart-money`
- **Famille de recherche (provisoire)** : `price_action_reversal`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/ict-optimal-trade-entry) · [page MNQ](https://pinescriptforge.com/mnq/ict-optimal-trade-entry/backtest)

## Description publiée

Enters at the 62–79% Fibonacci retracement zone after a displacement move — ICT's signature entry model.

- **Entrée annoncée** : After a bullish displacement, draw Fibonacci from swing low to swing high. Enter long between the 62% and 79% retracement level, ideally at an order block or FVG within the OTE zone. Opposite for bearish setups.
- **Sortie annoncée** : Target the -27% or -62% Fibonacci extension beyond the original swing high. Stop below the 100% level (swing low). Standard 3R minimum.

## Données et formalisation

- **Indicateurs cités** : Fibonacci Retracement, OTE Zone (62-79%), Order Blocks, Fair Value Gaps, Displacement
- **Horizons cités** : 5min, 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
