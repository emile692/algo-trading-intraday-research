# Registre commun des sleeves actives et séries forward

Date de gel : **9 octobre 2026**.

Ce dossier est le point d’entrée unique pour savoir ce qui doit tourner prospectivement. Il ne modifie pas les neuf sleeves actives : leur lock historique reste l’autorité canonique. Il ajoute les séries shadow/diagnostiques explicitement conservées après les campagnes récentes.

## Inventaire

| Groupe | Série | Instrument | Statut | Allocation |
|---|---|---|---|---:|
| Portefeuille actif | 9 sleeves equal-contract | MNQ / MGC / MBT | `LOCKED_ACTIVE_REFERENCE` | selon le lock actif |
| A0 | `A0_NQ` | NQ | `SHADOW_FORWARD_ONLY` | 0 |
| A0 | `A0_MNQ` | MNQ | `SHADOW_FORWARD_ONLY` | 0 |
| B | `B_canonical` | MGC | `FORWARD_MEASUREMENT_ONLY` | 0 |
| B | `B_no_volume` | MGC | contrôle diagnostique, pas une sleeve | 0 |
| Gotobi | `gotobi_6j_month_end` | 6J | `KEEP_FORWARD_TEST_ONLY` | 0 |

`A0_NQ` et `A0_MNQ` représentent le même alpha économique. Ils doivent être suivis séparément pour la robustesse et l’exécution, mais jamais additionnés comme deux sources de diversification.

`B_no_volume` sert uniquement à mesurer l’apport prospectif du filtre de volume. Il ne constitue pas une deuxième stratégie B.

## Fichiers canoniques

- [`registry.json`](registry.json) : inventaire machine-readable, statuts, chemins et empreintes.
- [`specs/a0_gap_reversion_nq_mnq.json`](specs/a0_gap_reversion_nq_mnq.json) : règles exactes A0 NQ/MNQ et protocole forward.
- [`specs/mgc_volume_climax_tracking.json`](specs/mgc_volume_climax_tracking.json) : B canonique et contrôle sans volume.
- [`specs/gotobi_6j_month_end.json`](specs/gotobi_6j_month_end.json) : règle Gotobi forward-only et limites de gouvernance.
- [`locked_nine_sleeve_config.json`](../../research/locked_nine_sleeve_portfolio/locked_nine_sleeve_config.json) : référence active des neuf sleeves.

## Règle d’ajout

Une nouvelle entrée nécessite une décision explicite, une règle intégralement définie, un statut non ambigu et un nouveau commit. Une watchlist ou un résultat rétrospectif positif n’entre pas automatiquement ici.

Le commit Git qui introduit ou modifie ce dossier est le hash de gel. Pour l’obtenir :

```powershell
git log -1 --format=%H -- strategy_registry/forward_monitoring
```
