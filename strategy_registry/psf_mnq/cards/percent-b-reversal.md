# Percent B Reversal

- **ID** : `percent-b-reversal`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `mean-reversion`
- **Famille de recherche (provisoire)** : `mean_reversion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/percent-b-reversal) · [page MNQ](https://pinescriptforge.com/mnq/percent-b-reversal/backtest)

## Description publiée

Trades Bollinger %B extremes for precise mean reversion entries.

- **Entrée annoncée** : Enter long when %B drops below 0 then rises back above 0 (re-entering bands from below). Short when %B drops back below 1.0 from above.
- **Sortie annoncée** : Target %B = 0.5 (middle band). Stop at %B = -0.2 (long) or 1.2 (short). Trail at 20 SMA.

## Données et formalisation

- **Indicateurs cités** : %B, Bollinger Bands, SMA 20
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
