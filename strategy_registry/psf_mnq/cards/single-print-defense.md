# Single Print Zone Defense

- **ID** : `single-print-defense`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `auction-market`
- **Famille de recherche (provisoire)** : `volume_auction`
- **Rôle envisagé (provisoire)** : `alpha_or_feature`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/single-print-defense) · [page MNQ](https://pinescriptforge.com/mnq/single-print-defense/backtest)

## Description publiée

Trades bounces and breaks at single print zones from prior market profile distributions.

- **Entrée annoncée** : Enter long at single print support zone with bullish candle pattern. Short at single print resistance. Break through = continuation entry.
- **Sortie annoncée** : Bounce trade: target opposing single print zone. Break trade: target 1× zone width beyond break. Stop: 3 ticks through zone.

## Données et formalisation

- **Indicateurs cités** : Single Print Zones, TPO Profile, Volume, ATR
- **Horizons cités** : 15min, 30min
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Distribution volume-prix nécessaire pour reproduire un vrai profil de marché.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
