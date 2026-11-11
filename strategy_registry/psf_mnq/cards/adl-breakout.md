# ADL Breakout Confirmation

- **ID** : `adl-breakout`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `breakout`
- **Famille de recherche (provisoire)** : `breakout_expansion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/adl-breakout) · [page MNQ](https://pinescriptforge.com/mnq/adl-breakout/backtest)

## Description publiée

Uses Advance-Decline concepts to confirm genuine breakouts with breadth.

- **Entrée annoncée** : Enter on price breakout above resistance when volume is distributed across multiple bars (not one spike). Confirm with A/D Line trending in breakout direction.
- **Sortie annoncée** : Exit if volume distribution narrows (single-bar dominance) after entry. Trail at breakout level + 1× ATR.

## Données et formalisation

- **Indicateurs cités** : A/D Line, Volume Distribution, ATR
- **Horizons cités** : 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
