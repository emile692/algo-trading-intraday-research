# Gap Fill

- **ID** : `gap-fill`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `gap`
- **Famille de recherche (provisoire)** : `session_gap`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/gap-fill) · [page MNQ](https://pinescriptforge.com/mnq/gap-fill/backtest)

## Description publiée

Trades the statistical tendency for overnight gaps to fill during the session.

- **Entrée annoncée** : If the open gaps up, sell short targeting the prior close. If gaps down, buy long. Only trade gaps smaller than 1× ATR (larger gaps may not fill).
- **Sortie annoncée** : Target the prior session close (100% gap fill). Stop at 50% extension of the gap. Time stop: exit by midday if unfilled.

## Données et formalisation

- **Indicateurs cités** : Gap Size, Prior Close, ATR, Volume
- **Horizons cités** : 5min, 15min
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Calendrier de séance et horodatage New York à définir.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
