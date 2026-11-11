# Force Index Breakout

- **ID** : `force-index-breakout`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `volume`
- **Famille de recherche (provisoire)** : `volume_auction`
- **Rôle envisagé (provisoire)** : `alpha_or_feature`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/force-index-breakout) · [page MNQ](https://pinescriptforge.com/mnq/force-index-breakout/backtest)

## Description publiée

Uses Force Index spikes to confirm breakout strength and filter false breaks.

- **Entrée annoncée** : Enter breakout only when 2-period Force Index > 2σ above its mean. Direction matches the breakout.
- **Sortie annoncée** : Trail stop at Force Index reversal to negative (for longs). Target 2× ATR.

## Données et formalisation

- **Indicateurs cités** : Force Index, Breakout Levels, Volume
- **Horizons cités** : 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
