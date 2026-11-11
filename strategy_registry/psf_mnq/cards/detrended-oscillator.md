# Detrended Oscillator

- **ID** : `detrended-oscillator`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `mean-reversion`
- **Famille de recherche (provisoire)** : `mean_reversion`
- **Rôle envisagé (provisoire)** : `alpha_candidate`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/detrended-oscillator) · [page MNQ](https://pinescriptforge.com/mnq/detrended-oscillator/backtest)

## Description publiée

Removes trend bias to isolate pure cyclical overbought/oversold signals.

- **Entrée annoncée** : Enter long when DPO reaches -2σ level. Short at +2σ. Confirm with volume below average.
- **Sortie annoncée** : Exit at DPO zero line. Stop at -3σ/+3σ. Risk 1× ATR per trade.

## Données et formalisation

- **Indicateurs cités** : DPO, Standard Deviation, Volume
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
