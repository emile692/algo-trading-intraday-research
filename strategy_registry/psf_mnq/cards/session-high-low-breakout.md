# Session High/Low Breakout

- **ID** : `session-high-low-breakout`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `breakout`
- **Famille de recherche (provisoire)** : `breakout_expansion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/session-high-low-breakout) · [page MNQ](https://pinescriptforge.com/mnq/session-high-low-breakout/backtest)

## Description publiée

Trades breakouts beyond prior session highs and lows where stop clusters accumulate.

- **Entrée annoncée** : Enter long on a 15min close above prior session high with volume ≥ 1.5× average. Enter short on close below prior session low with similar volume confirmation.
- **Sortie annoncée** : Target 1× the prior session range projected from the breakout level. Stop re-entry back inside the session range. Trail at prior session high/low once target 1 is hit.

## Données et formalisation

- **Indicateurs cités** : Prior Session High/Low, Volume MA, ATR
- **Horizons cités** : 5min, 15min, 1H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Calendrier de séance et horodatage New York à définir.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
