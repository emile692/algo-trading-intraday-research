# McGinley Dynamic

- **ID** : `mcginley-dynamic`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `trend`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/mcginley-dynamic) · [page MNQ](https://pinescriptforge.com/mnq/mcginley-dynamic/backtest)

## Description publiée

Self-adjusting moving average that eliminates whipsaws in choppy markets.

- **Entrée annoncée** : Enter long when price crosses above McGinley Dynamic and MD is rising. Short when price crosses below and MD is falling.
- **Sortie annoncée** : Exit when price crosses back through the McGinley Dynamic line. Use ATR-based stop for protection.

## Données et formalisation

- **Indicateurs cités** : McGinley Dynamic, ATR
- **Horizons cités** : 15min, 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
