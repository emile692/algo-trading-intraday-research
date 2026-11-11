# Compression-Expansion

- **ID** : `compression-expansion`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `range`
- **Famille de recherche (provisoire)** : `breakout_expansion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/compression-expansion) · [page MNQ](https://pinescriptforge.com/mnq/compression-expansion/backtest)

## Description publiée

Measures price compression via ATR contraction and trades the subsequent expansion.

- **Entrée annoncée** : Enter when ATR rises 50%+ from its 20-period low. Direction determined by price vs 50 EMA.
- **Sortie annoncée** : Exit when ATR peaks (starts declining after expansion). Trail stop at 1× current ATR.

## Données et formalisation

- **Indicateurs cités** : ATR, ATR Percentile, EMA 50
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
