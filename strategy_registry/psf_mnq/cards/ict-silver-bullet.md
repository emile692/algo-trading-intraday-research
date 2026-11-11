# ICT Silver Bullet

- **ID** : `ict-silver-bullet`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `smart-money`
- **Famille de recherche (provisoire)** : `price_action_reversal`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/ict-silver-bullet) · [page MNQ](https://pinescriptforge.com/mnq/ict-silver-bullet/backtest)

## Description publiée

Time-based precision entry during three specific 1-hour windows where algorithmic displacement occurs.

- **Entrée annoncée** : During a Silver Bullet window (3-4 AM, 10-11 AM, or 2-3 PM ET), identify the displacement candle that creates a Fair Value Gap. Enter on the first retest of the FVG in the direction of the displacement.
- **Sortie annoncée** : Target the liquidity pool that the displacement was seeking (prior high/low). Stop beyond the FVG. Time-based exit: close any remaining position at the end of the Silver Bullet window.

## Données et formalisation

- **Indicateurs cités** : Time-of-Day Filter, Fair Value Gap, Displacement Detection, Liquidity Pools
- **Horizons cités** : 1min, 5min, 15min
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
