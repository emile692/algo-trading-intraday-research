# Registre commun des sleeves actives et séries forward

Date de gel : **9 octobre 2026**.

Ce dossier est le point d’entrée unique pour savoir ce qui doit tourner prospectivement. Il ne modifie pas les neuf sleeves actives : leur lock historique reste l’autorité canonique. Il ajoute les séries shadow/diagnostiques explicitement conservées après les campagnes récentes.

## Inventaire

| Groupe | Série | Instrument | Statut | Allocation |
|---|---|---|---|---:|
| Portefeuille actif | 9 sleeves equal-contract | MNQ / MGC / MBT | `LOCKED_ACTIVE_REFERENCE` | selon le lock actif |
| Gap reversion | `nq_opening_gap_reversion` | NQ | `SHADOW_FORWARD_ONLY` | 0 |
| Gap reversion | `mnq_opening_gap_reversion` | MNQ | `SHADOW_FORWARD_ONLY` | 0 |
| Volume climax | `mgc_volume_climax_confirmed_fade` | MGC | `FORWARD_MEASUREMENT_ONLY` | 0 |
| Price climax | `mgc_price_climax_confirmed_fade_control` | MGC | contrôle diagnostique, pas une sleeve indépendante | 0 |
| Gotobi | `6j_month_end_pre_fix_gotobi` | 6J | `KEEP_FORWARD_TEST_ONLY` | 0 |

`nq_opening_gap_reversion` et `mnq_opening_gap_reversion` représentent le même alpha économique. Ils doivent être suivis séparément pour la robustesse et l’exécution, mais jamais additionnés comme deux sources de diversification. Leurs anciens codes `A0_NQ` et `A0_MNQ` sont conservés comme alias d’audit dans les JSON.

`mgc_price_climax_confirmed_fade_control` sert uniquement à mesurer l’apport prospectif du filtre de volume. Il ne constitue pas une deuxième stratégie indépendante. Les anciens codes `B_canonical`, `B_no_volume` et `gotobi_6j_month_end` restent également disponibles comme alias.

## Fichiers canoniques

- [`registry.json`](registry.json) : inventaire machine-readable, statuts, chemins et empreintes.
- [`specs/equity_index_opening_gap_reversion.json`](specs/equity_index_opening_gap_reversion.json) : règles exactes des versions NQ/MNQ et protocole forward.
- [`specs/mgc_volume_climax_tracking.json`](specs/mgc_volume_climax_tracking.json) : B canonique et contrôle sans volume.
- [`specs/6j_month_end_pre_fix_gotobi.json`](specs/6j_month_end_pre_fix_gotobi.json) : règle Gotobi forward-only et limites de gouvernance.
- [`locked_nine_sleeve_config.json`](../../research/locked_nine_sleeve_portfolio/locked_nine_sleeve_config.json) : référence active des neuf sleeves.

## Règle d’ajout

Une nouvelle entrée nécessite une décision explicite, une règle intégralement définie, un statut non ambigu et un nouveau commit. Une watchlist ou un résultat rétrospectif positif n’entre pas automatiquement ici.

Le commit Git qui introduit ou modifie ce dossier est le hash de gel. Pour l’obtenir :

```powershell
git log -1 --format=%H -- strategy_registry/forward_monitoring
```
