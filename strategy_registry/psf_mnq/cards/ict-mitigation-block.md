# ICT Mitigation Block

- **ID** : `ict-mitigation-block`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `smart-money`
- **Famille de recherche (provisoire)** : `price_action_reversal`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/ict-mitigation-block) · [page MNQ](https://pinescriptforge.com/mnq/ict-mitigation-block/backtest)

## Description publiée

Identifies failed Order Blocks that become mitigation zones for reversal entries.

- **Entrée annoncée** : Identify a breached Order Block where price broke through and continued. When price retraces back to this failed Order Block zone, enter in the direction of the break (continuation).
- **Sortie annoncée** : Target the swing created after the initial break. Stop beyond the Mitigation Block.

## Données et formalisation

- **Indicateurs cités** : Mitigation Block, Order Block, Break of Structure
- **Horizons cités** : 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
