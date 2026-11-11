# Cup and Handle

- **ID** : `cup-and-handle`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `pattern-recognition`
- **Famille de recherche (provisoire)** : `chart_pattern`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/cup-and-handle) · [page MNQ](https://pinescriptforge.com/mnq/cup-and-handle/backtest)

## Description publiée

Identifies the classic cup-and-handle continuation pattern for breakout entries.

- **Entrée annoncée** : Enter long on breakout above handle resistance with volume > 1.5× average. Handle should retrace < 50% of cup depth.
- **Sortie annoncée** : Target: cup depth added to breakout point. Stop below handle low. Trail at rising 10-day MA after initial target.

## Données et formalisation

- **Indicateurs cités** : Cup Depth, Handle Pattern, Volume, Resistance Breakout
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
