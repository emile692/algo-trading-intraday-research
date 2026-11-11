# Volume Profile POC

- **ID** : `volume-profile-poc`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `volume`
- **Famille de recherche (provisoire)** : `volume_auction`
- **Rôle envisagé (provisoire)** : `alpha_or_feature`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/volume-profile-poc) · [page MNQ](https://pinescriptforge.com/mnq/volume-profile-poc/backtest)

## Description publiée

Trades around the Point of Control using volume profile analysis.

- **Entrée annoncée** : Enter long at POC with bullish rejection candle. Enter breakout long above Value Area High (VAH) with volume surge.
- **Sortie annoncée** : Mean reversion: target VAH/VAL. Breakout: trail stop at POC. Risk 1× ATR per trade.

## Données et formalisation

- **Indicateurs cités** : Volume Profile, POC, VAH, VAL
- **Horizons cités** : 5min, 15min, 1H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Distribution volume-prix nécessaire pour reproduire un vrai profil de marché.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
