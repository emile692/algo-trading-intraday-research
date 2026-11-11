# DMI/ADX Complete System

- **ID** : `dmi-adx-system`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `trend`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/dmi-adx-system) · [page MNQ](https://pinescriptforge.com/mnq/dmi-adx-system/backtest)

## Description publiée

Full trend system using Directional Movement Index with ADX filtering.

- **Entrée annoncée** : Enter long when +DI > -DI AND ADX > 20 AND ADX is rising. Enter short when -DI > +DI AND ADX > 20 AND ADX is rising.
- **Sortie annoncée** : Exit when ADX starts falling even if above 25 (trend weakening). Also exit on DI crossover against position. Trail at 2× ATR.

## Données et formalisation

- **Indicateurs cités** : +DI, -DI, ADX, ADX Slope, ATR
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
