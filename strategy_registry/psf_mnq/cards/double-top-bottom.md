# Double Top / Double Bottom

- **ID** : `double-top-bottom`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `pattern-recognition`
- **Famille de recherche (provisoire)** : `chart_pattern`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/double-top-bottom) · [page MNQ](https://pinescriptforge.com/mnq/double-top-bottom/backtest)

## Description publiée

Identifies double top and double bottom reversal patterns at key price levels.

- **Entrée annoncée** : Enter short on break below double top neckline. Enter long on break above double bottom neckline. Volume must expand on break.
- **Sortie annoncée** : Target: pattern height projected from neckline. Stop above double top high / below double bottom low. Partial exit at 50% target.

## Données et formalisation

- **Indicateurs cités** : Pattern Detection, Neckline, Volume, Support/Resistance
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
