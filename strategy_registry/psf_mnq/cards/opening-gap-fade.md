# Opening Gap Fade

- **ID** : `opening-gap-fade`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `gap`
- **Famille de recherche (provisoire)** : `session_gap`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/opening-gap-fade) · [page MNQ](https://pinescriptforge.com/mnq/opening-gap-fade/backtest)

## Description publiée

Fades the opening gap with confirmation from the first 5-minute bar.

- **Entrée annoncée** : Wait for the first 5-minute bar. If it closes against the gap direction (bearish bar on gap up, bullish on gap down), enter the fade trade.
- **Sortie annoncée** : Target 50% then 100% gap fill. Stop above/below the first 5-minute bar extreme. Hold up to 2 hours.

## Données et formalisation

- **Indicateurs cités** : Gap Size, 5min Chart, Prior Close, Volume
- **Horizons cités** : 5min, 15min
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Calendrier de séance et horodatage New York à définir.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
