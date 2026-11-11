# Liquidity Sweep Reversal

- **ID** : `liquidity-sweep-reversal`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `mean-reversion`
- **Famille de recherche (provisoire)** : `mean_reversion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/liquidity-sweep-reversal) · [page MNQ](https://pinescriptforge.com/mnq/liquidity-sweep-reversal/backtest)

## Description publiée

Fades stop hunts at equal highs/lows and swing point clusters.

- **Entrée annoncée** : Identify a cluster of equal highs or swing highs. Wait for price to spike 2–5 ticks above (or below for lows) with a wick, then enter short (long) on the first candle that closes back below (above) the swept level.
- **Sortie annoncée** : Target the opposing liquidity pool or a Fair Value Gap. Stop above the sweep wick. Minimum 2R target.

## Données et formalisation

- **Indicateurs cités** : Equal Highs/Lows Detection, Wick Analysis, Session Filter
- **Horizons cités** : 1min, 5min, 15min
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Calendrier de séance et horodatage New York à définir.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
