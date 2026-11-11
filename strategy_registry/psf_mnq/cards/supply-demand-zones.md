# Supply & Demand Zones

- **ID** : `supply-demand-zones`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `mean-reversion`
- **Famille de recherche (provisoire)** : `mean_reversion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/supply-demand-zones) · [page MNQ](https://pinescriptforge.com/mnq/supply-demand-zones/backtest)

## Description publiée

Identifies institutional supply and demand zones where price is likely to reverse.

- **Entrée annoncée** : Identify a strong impulsive move up — the consolidation base before the move is the demand zone. Enter long when price returns to this zone with a confirmation candle. Opposite for supply zones (base before a drop).
- **Sortie annoncée** : Target the origin of the opposing zone or a 2:1 reward-to-risk. Stop below the demand zone (or above supply zone). Scale out 50% at 1R, remainder at 2R.

## Données et formalisation

- **Indicateurs cités** : Supply Zone Detection, Demand Zone Detection, Impulse Move Identification, Volume Confirmation
- **Horizons cités** : 15min, 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
