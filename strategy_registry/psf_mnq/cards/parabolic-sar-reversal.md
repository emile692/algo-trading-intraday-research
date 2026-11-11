# Parabolic SAR Reversal

- **ID** : `parabolic-sar-reversal`
- **Univers** : `ranked_184`
- **Statut** : `Idea` — aucune validation indépendante
- **Catégorie du site** : `trend`
- **Famille de recherche (provisoire)** : `trend_momentum`
- **Rôle envisagé (provisoire)** : `alpha_or_filter`
- **Sources** : [fiche stratégie](https://pinescriptforge.com/strategy/parabolic-sar-reversal) · [page MNQ](https://pinescriptforge.com/mnq/parabolic-sar-reversal/backtest)

## Description publiée

Trades trend reversals using Parabolic SAR dot flips.

- **Entrée annoncée** : Enter long when SAR flips below price while price is above the 100 EMA. Enter short when SAR flips above price while below the 100 EMA. Use AF 0.02, max 0.20.
- **Sortie annoncée** : Exit on SAR flip against position. The SAR itself acts as a trailing stop.

## Données et formalisation

- **Indicateurs cités** : Parabolic SAR, EMA 100, ATR 14
- **Horizons cités** : 15min, 1H, 4H
- **Donnée à vérifier** : Barres OHLCV adaptées aux horizons indiqués, à confirmer.
- **Règles exactes** : à spécifier avant backtest (horodatage du signal, prix d'exécution, paramètres, stops, cas limites).

Les métriques du site ne sont pas reportées : leur présence dans le classement ne prouve aucun alpha exploitable.
