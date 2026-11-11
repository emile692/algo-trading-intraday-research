# Money Flow Index (MFI)

- **ID** : `money-flow-index`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `volume`
- **Famille de recherche (provisoire)** : `volume_auction`
- **Rôle envisagé (provisoire)** : `alpha_or_feature`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/money-flow-index) · [page MNQ](https://pinescriptforge.com/mnq/money-flow-index/backtest)

## Description publiée

Volume-weighted RSI for detecting overbought/oversold with volume confirmation.

- **Entrée annoncée** : Enter long when MFI drops below 20 (oversold with volume confirmation). Enter short when MFI rises above 80.
- **Sortie annoncée** : Exit long when MFI reaches 70. Exit short when MFI reaches 30. Stop at recent swing extreme.

## Données et formalisation

- **Indicateurs cités** : MFI, Volume, ATR
- **Horizons cités** : 1H, 4H, Daily
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
