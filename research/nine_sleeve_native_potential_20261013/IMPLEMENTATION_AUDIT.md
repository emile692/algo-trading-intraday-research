# Audit des implémentations et de l'historique de sélection

| ID | Sleeve | Entrée et sortie effectivement rejouées | Sizing source | Coûts nominaux | Période native | Statut sizing USD |
| --- | --- | --- | --- | --- | --- | --- |
| S01 | `mnq_orb_vwap_atr` | ORB/VWAP/ATR retenu, stop et target structurels, flat 16:00 NY | budget 250 USD, quantité entière 1–4 ; q1 reconstruit dans cette campagne | MNQ : 1 tick/côté + 1,25 USD/côté | 2019-05-06–2026-03-19 | oui ; prix prévu = close minute observable avant ordre |
| S02 | `mnq_volatility_breakout` | stop-entry gelé ; sortie effective time-stop 240 min | 1 MNQ | MNQ gelé | 2019-05-06–2026-03-19 | non : aucun stop de sortie effectif cohérent |
| S03 | `mnq_compression_breakout` | OCO stop-entry et stop/target baseline gelés | 1 MNQ | MNQ gelé | 2019-05-06–2026-03-19 | oui ; trigger et stop connus avant fill |
| S04 | `mnq_atr_trend_flip` | signal 15 min ; stop ATR(14) gelé de la dernière barre achevée | 1 MNQ | MNQ gelé | 2019-05-06–2026-03-19 | oui ; distance ATR causale, indépendante du fill |
| S05 | `mgc_volatility_breakout` | stop-entry et stop/target baseline gelés | 1 MGC | MGC : 1 tick/côté + 1,25 USD/côté | 2019-05-06–2026-03-19 | oui ; trigger et stop connus avant fill |
| S06 | `mgc_gap_continuation` | entrée next-open après confirmation, stop/target baseline gelés | 1 MGC | MGC gelé | 2019-05-06–2026-03-19 | oui ; prix prévu = close minute observable avant ordre |
| S07 | `mbt_rolling_range_breakout` | vote moyen des cellules, seuil ±0,5, exécution next-minute, flat 16:08 NY | 1 MBT exécutable | MBT : 1 tick/côté + 1,41 USD/côté | 2024-01-03–2026-10-02 | non : pas de stop initial unique de l'ensemble |
| S08 | `mbt_trend_pullback_ma_structure` | même protocole d'ensemble exécutable | 1 MBT exécutable | MBT gelé | 2024-01-03–2026-10-02 | non |
| S09 | `mbt_volatility_breakout` | même protocole d'ensemble exécutable | 1 MBT exécutable | MBT gelé | 2024-01-03–2026-10-02 | non |

Sources principales : `locked_four_sleeve_config.json`, ledgers S01, spécifications baseline gelées, `mgc_final_two_sleeve_lock_20260926`, et `mbt_three_sleeve_freeze_20261004/vote_ensemble/ensemble_manifest.json`.

Historique de sélection : MNQ et MGC ont été étudiés sur les périodes antérieures déjà documentées ; la validation MGC 2024+ est explicitement `adaptive_reused_not_pristine_oos`. Pour MBT, le split historique au 2025-05-29 a été inspecté pendant le choix des régions. Aucun de ces segments n'est requalifié ici en OOS indépendant.

La reproduction exacte des neuf sources est enregistrée dans `reference_reproduction.csv` (écart maximal par trade : 0 USD pour les neuf sleeves).
