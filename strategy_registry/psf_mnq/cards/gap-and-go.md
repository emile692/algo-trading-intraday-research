# Gap & Go

- **ID** : `gap-and-go`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `gap`
- **Famille de recherche (provisoire)** : `session_gap`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/gap-and-go) · [page MNQ](https://pinescriptforge.com/mnq/gap-and-go/backtest)

## Description publiée

Rides large gaps that show continuation momentum instead of filling.

- **Entrée annoncée** : Enter in the gap direction after the first 15-minute candle closes in the gap's direction. Gap must be > 1× ATR. Volume must be above average.
- **Sortie annoncée** : Trail at each 15-minute swing low/high. Target 2× the gap size as a measured move. Stop if price re-enters the gap by 50%.

## Données et formalisation

- **Indicateurs cités** : Gap Size, 15min Chart, Volume, ATR
- **Horizons cités** : 5min, 15min
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Calendrier de séance et horodatage New York à définir.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
