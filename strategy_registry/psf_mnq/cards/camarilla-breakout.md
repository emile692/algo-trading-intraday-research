# Camarilla Pivot Breakout

- **ID** : `camarilla-breakout`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `breakout`
- **Famille de recherche (provisoire)** : `breakout_expansion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/camarilla-breakout) · [page MNQ](https://pinescriptforge.com/mnq/camarilla-breakout/backtest)

## Description publiée

Uses Camarilla pivot equations for precise intraday breakout levels.

- **Entrée annoncée** : Enter long on break above H4 with stop at H3. Enter short on break below L4 with stop at L3. Mean reversion: buy at L3 with target at H3, sell at H3 with target at L3.
- **Sortie annoncée** : Breakout: trail at 1.5× ATR. Mean reversion: exit at opposite L3/H3 level. Time exit at session close.

## Données et formalisation

- **Indicateurs cités** : Camarilla H1-H4, Camarilla L1-L4, ATR 14
- **Horizons cités** : 5min, 15min, 1H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Calendrier de séance et horodatage New York à définir.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
