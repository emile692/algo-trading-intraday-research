# Value Area Fade

- **ID** : `value-area-fade`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `auction-market`
- **Famille de recherche (provisoire)** : `volume_auction`
- **Rôle envisagé (provisoire)** : `alpha_or_feature`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/value-area-fade) · [page MNQ](https://pinescriptforge.com/mnq/value-area-fade/backtest)

## Description publiée

Fades price at Value Area High/Low for mean reversion to the Point of Control.

- **Entrée annoncée** : Enter short at VAH with bearish rejection candle. Enter long at VAL with bullish rejection. Require at least 2 tests of VA boundary.
- **Sortie annoncée** : Target POC for full reversion. Partial exit at VA midpoint. Stop 2 ticks beyond VA boundary.

## Données et formalisation

- **Indicateurs cités** : Value Area High, Value Area Low, POC, Volume Profile
- **Horizons cités** : 5min, 15min, 30min
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Distribution volume-prix nécessaire pour reproduire un vrai profil de marché.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
