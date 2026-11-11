# ICT Judas Swing

- **ID** : `ict-judas-swing`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `smart-money`
- **Famille de recherche (provisoire)** : `price_action_reversal`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/ict-judas-swing) · [page MNQ](https://pinescriptforge.com/mnq/ict-judas-swing/backtest)

## Description publiée

Identifies fake breakouts (Judas swings) that trap traders before reversing into the true move.

- **Entrée annoncée** : Wait for the first 30 minutes of London or NY open. If price sweeps the previous session high/low then reverses with a displacement candle, enter in the reversal direction.
- **Sortie annoncée** : Target the opposing session liquidity level. Stop-loss above/below the Judas swing high/low.

## Données et formalisation

- **Indicateurs cités** : Session High/Low, Killzone Timer, Displacement Detection
- **Horizons cités** : 1min, 5min, 15min
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Calendrier de séance et horodatage New York à définir.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
