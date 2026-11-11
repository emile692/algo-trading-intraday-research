# Opening Range Breakout (ORB)

- **ID** : `opening-range-breakout`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `breakout`
- **Famille de recherche (provisoire)** : `breakout_expansion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/opening-range-breakout) · [page MNQ](https://pinescriptforge.com/mnq/opening-range-breakout/backtest)

## Description publiée

Trades breakouts from the first 30-minute opening range with momentum confirmation.

- **Entrée annoncée** : Mark the high and low of the first 30 minutes. Enter long on a 1-tick close above the ORB high with expanding volume. Enter short on a close below the ORB low.
- **Sortie annoncée** : Target 1× the ORB range beyond the breakout level. Stop-loss inside the range at the midpoint. Optional: trail at the ORB level after initial target hit.

## Données et formalisation

- **Indicateurs cités** : Opening Range High/Low, Volume, ATR
- **Horizons cités** : 1min, 5min, 15min
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).
- **Référence locale** : cette fiche du site est une idée externe; elle ne désigne pas l'ORB MNQ retenu dans ce dépôt.

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
