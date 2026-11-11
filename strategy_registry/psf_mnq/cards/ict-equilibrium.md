# ICT Equilibrium (50% Level)

- **ID** : `ict-equilibrium`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `smart-money`
- **Famille de recherche (provisoire)** : `price_action_reversal`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/ict-equilibrium) · [page MNQ](https://pinescriptforge.com/mnq/ict-equilibrium/backtest)

## Description publiée

Trades rejections at the 50% equilibrium level of a dealing range — where fair value lives.

- **Entrée annoncée** : Identify the current dealing range (recent swing high to swing low). Calculate the 50% equilibrium level. Enter long in discount territory (below 50%) at an Order Block or FVG. Enter short in premium territory (above 50%) at a supply zone.
- **Sortie annoncée** : Target the opposing extreme of the dealing range. Stop beyond the OB/FVG entry zone. Partial profit at equilibrium if entering from deep discount/premium.

## Données et formalisation

- **Indicateurs cités** : Dealing Range, Equilibrium Level, Premium/Discount Zones, Order Blocks
- **Horizons cités** : 15min, 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
