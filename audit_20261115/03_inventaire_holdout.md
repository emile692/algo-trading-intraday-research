# Phase 3 — Inventaire du holdout

*Lu le 2026-11-15. Seuls les timestamps ont été lus, depuis les statistiques des row groups parquet. Aucun prix, rendement ni indicateur n'a été calculé. Script : [phase3_inventory.py](phase3_inventory.py). Table : [03_inventory.csv](outputs/03_inventory.csv).*

## Dernier timestamp disponible par instrument

- **MNQ** : 2026-10-02 16:59 ET
- **MGC** : 2026-10-02 16:59 ET
- **MBT** : 2026-10-02 19:55 ET

## Fichiers

| Fichier | Instrument | Premier timestamp (UTC) | Dernier timestamp (ET) | Lignes | Méthode |
|---|---|---|---|---:|---|
| `data\downloaded\GC_v_0_to_v_9_1m_20150102_20230317.parquet` | GC | 2015-01-02 00:00 | **2023-03-16 19:59** | 5 137 924 | row-group statistics |
| `data\downloaded\MBT_v_0_1m_20210425_20261003.parquet` | MBT | 2021-05-02 22:01 | **2026-10-02 19:55** | 1 365 286 | row-group statistics |
| `data\downloaded\MGC_v_0_1m_20150101_20260923.parquet` | MGC | 2015-01-01 23:01 | **2026-09-22 19:59** | 3 289 297 | row-group statistics |
| `data\downloaded\MGC_v_0_1m_20260923_20261003_extension.parquet` | MGC | 2026-09-23 00:00 | **2026-10-02 16:59** | 10 920 | row-group statistics |
| `data\downloaded\MNQ_c_0_1m_20260320_20261003_extension.parquet` | MNQ | 2026-03-20 00:00 | **2026-10-02 16:59** | 191 547 | row-group statistics |
| `data\downloaded\MNQ_c_0_1m_20260321_094501.parquet` | MNQ | 2019-05-05 22:03 | **2026-03-20 09:29** | 2 401 697 | row-group statistics |
| `data\downloaded\NQ_c_0_1m_20260321_095036.parquet` | NQ | 2010-06-07 00:00 | **2026-03-20 09:29** | 5 241 154 | row-group statistics |
| `data\processed\parquet\GC_v_0_to_v_9_aggregate_volume_5m_20150102_20230317.parquet` | GC | 2015-01-02 00:00 | **2023-03-16 19:55** | 862 848 | row-group statistics |
| `data\processed\parquet\MBT_v_0_5m_20210425_20261003.parquet` | MBT | 2021-05-02 22:00 | **2026-10-02 19:55** | 358 980 | row-group statistics |
| `data\processed\parquet\MGC_c_0_1m_20260322_155729.parquet` | MGC | 2010-10-03 23:26 | **2026-03-20 16:59** | 1 648 698 | row-group statistics |
| `data\processed\parquet\MGC_c_0_5m_20260322_155729.parquet` | MGC | 2010-10-03 23:25 | **2026-03-20 16:55** | 461 313 | row-group statistics |
| `data\processed\parquet\MGC_v_0_1m_20150101_20260923.parquet` | MGC | 2015-01-01 23:01 | **2026-09-22 19:59** | 3 289 297 | row-group statistics |
| `data\processed\parquet\MGC_v_0_1m_20150101_20261003.parquet` | MGC | 2015-01-01 23:01 | **2026-10-02 16:59** | 3 300 217 | row-group statistics |
| `data\processed\parquet\MGC_v_0_5m_20150101_20260923.parquet` | MGC | 2015-01-01 23:00 | **2026-09-22 19:55** | 778 185 | row-group statistics |
| `data\processed\parquet\MNQ_c_0_1m_20190505_20261003.parquet` | MNQ | 2019-05-05 22:03 | **2026-10-02 16:59** | 2 592 447 | row-group statistics |
| `data\processed\parquet\MNQ_c_0_1m_20260321_094501.parquet` | MNQ | 2019-05-05 22:03 | **2026-03-20 09:29** | 2 401 697 | row-group statistics |
| `data\processed\parquet\MNQ_c_0_5m_20260321_094501.parquet` | MNQ | 2019-05-05 22:00 | **2026-03-20 09:25** | 482 559 | row-group statistics |
| `data\processed\parquet\NQ_c_0_1m_20260321_095036.parquet` | NQ | 2010-06-07 00:00 | **2026-03-20 09:29** | 5 241 154 | row-group statistics |
| `data\processed\parquet\NQ_c_0_5m_20260321_095036.parquet` | NQ | 2010-06-07 00:00 | **2026-03-20 09:25** | 1 097 001 | row-group statistics |

## Fenêtres déjà consultées

| Instrument | Fenêtre consultée | Trace dans le repo |
|---|---|---|
| MNQ | 2019-05-06 → 2026-03-19 | IS + validation réutilisée des sleeves MNQ (fiches 01–04) |
| MGC | 2019-05-06 → 2026-03-19 | IS + validation réutilisée des sleeves MGC (fiches 05–06) |
| MNQ | 2026-03-20 → 2026-10-02 | forward replay des votes gelés S02–S04 et de S01 `baseline_fixed` (research/frozen_vote_forward_replay_20261101) ; a motivé la sortie de S04 (ACTIVE_EIGHT_SLEEVES.md) ; déclarée EXT contaminée dans book_aggregation_risk_r1_20261112 |
| MGC | 2026-03-20 → 2026-10-02 | forward replay S05–S06 (même dossier) ; règle d'arrêt rejouée sur S05 (forward_context.json) |
| MBT | 2021-05-03 → 2026-10-02 | IS + legacy OOS des sleeves MBT (fiches 07–09) ; S09 nul sur son legacy OOS |

## Verdict

**Aucune donnée vierge n'existe sur disque pour les signaux tels qu'ils sont tradés.**

- MNQ et MGC du 2026-03-20 au dernier timestamp : rejoués le 2026-11-01 pour toutes les sleeves MNQ/MGC du book. C'est cette fenêtre qui a fait sortir S04. Elle n'est plus un holdout.
- MBT après le 2026-10-02 : aucun fichier ne dépasse le 2026-10-02 au soir (séance Globex qui ouvre le dimanche 2026-10-04 non comprise).
- Quelques minutes de Globex du vendredi 2026-10-02 au soir peuvent dépasser la dernière séance RTH. Elles n'appartiennent à aucune séance tradée par les sleeves.

**Cas particulier de S01 (ensemble ATR 18).** Le forward replay du 2026-11-01 a rejoué S01 `baseline_fixed`, pas l'ensemble ATR 18 de la fiche. Le S01 de référence n'a donc jamais été calculé sur 2026-03-20 → 2026-10-02. Cette fenêtre n'est pas vierge pour autant : les deux versions partagent l'entrée, la sortie et le sizing, et 100 % des trades ATR 18 de 2024-01 → 2026-03 figurent dans `baseline_fixed`. Le PnL de `baseline_fixed` sur cette fenêtre est connu. Au mieux, c'est un holdout partiellement contaminé, valable pour S01 seul.

**Conséquence pour la phase 4.** La seule validation propre est le suivi prospectif. Il démarre à la première séance après le 2026-11-03 pour le book à 8 sleeves (registre `forward_monitoring`), et ses données ne sont pas encore sur disque. Le pré-enregistrement de la phase 4a porte donc sur ce suivi. Il fixe les critères par horizon (20, 60, 120 et 250 séances), à évaluer quand les données arriveront. Un test one-shot de S01 ATR 18 sur 2026-03-20 → 2026-10-02 est proposé en option, étiqueté « partiellement contaminé ».
