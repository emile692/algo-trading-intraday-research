# HTF Trend + LTF Entry

- **ID** : `htf-trend-ltf-entry`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `multi-timeframe`
- **Famille de recherche (provisoire)** : `composite`
- **Rôle envisagé (provisoire)** : `meta_signal`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/htf-trend-ltf-entry) · [page MNQ](https://pinescriptforge.com/mnq/htf-trend-ltf-entry/backtest)

## Description publiée

Uses higher timeframe trend direction with lower timeframe precision entries.

- **Entrée annoncée** : Determine trend on Daily (above/below 50 EMA). On 15min, enter long at support levels or EMA pullbacks when Daily is bullish. Short on 15min resistance when Daily is bearish.
- **Sortie annoncée** : Target the next 15min resistance/support level. Stop at 1× ATR on 15min. Scale out at 1:1 and trail remainder.

## Données et formalisation

- **Indicateurs cités** : EMA 50 (Daily), Support/Resistance (15min), ATR
- **Horizons cités** : 15min, 1H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
