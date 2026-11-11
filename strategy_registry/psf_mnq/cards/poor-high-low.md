# Poor High/Low Retest

- **ID** : `poor-high-low`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `auction-market`
- **Famille de recherche (provisoire)** : `volume_auction`
- **Rôle envisagé (provisoire)** : `alpha_or_feature`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/poor-high-low) · [page MNQ](https://pinescriptforge.com/mnq/poor-high-low/backtest)

## Description publiée

Identifies 'poor' (unfinished) auction highs/lows that price returns to complete.

- **Entrée annoncée** : Enter long above a poor low being retested (expecting price to push through). Short below a poor high being retested.
- **Sortie annoncée** : Target the single print zone beyond the poor high/low. Stop at the 50% level between poor extreme and session POC.

## Données et formalisation

- **Indicateurs cités** : Poor High/Low, Single Prints, Excess Tails, Market Profile
- **Horizons cités** : 15min, 30min, 1H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Donnée à vérifier** : Calendrier de séance et horodatage New York à définir.
- **Donnée à vérifier** : Distribution volume-prix nécessaire pour reproduire un vrai profil de marché.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
