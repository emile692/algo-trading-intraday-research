# Référence remplacée — conservée pour audit

Le portefeuille `eight_sleeve_equal_contract_reference_20261103` défini dans ce répertoire n'est plus la référence active depuis le **15 novembre 2026**.

Il est remplacé par `eight_sleeve_equal_contract_reference_20261115`, défini dans [`research/locked_eight_sleeve_portfolio_20261115/`](../locked_eight_sleeve_portfolio_20261115/locked_eight_sleeve_config.json). Ce lock corrige deux défauts de celui-ci :

- **S01.** Ce répertoire utilise pour `mnq_orb_vwap_atr` le ledger historique `baseline_fixed` (ATR 14 × q_high {90, 95}, 823 trades). La sleeve de référence est celle de la fiche 01 : le vote de 18 modèles ATR {14, 16, 18} × q_low {20, 25, 30} × q_high {80, 85}, 701 trades. Écart relevé par `audit_20261115`, phase 0, et tranché par l'utilisateur le 2026-11-15.
- **Calendrier.** Le PnL quotidien était aligné sur le calendrier MNQ, ce qui écarte 8 vendredis d'échéance trimestrielle tradés par MGC et MBT (−509,26 USD sur 2024-01-02 → 2026-03-19).

Les sept autres sleeves actives, S04 en shadow et la règle d'arrêt sont inchangés.

Les fichiers de ce répertoire, hachés dans `lock_manifest.json`, ne sont volontairement pas modifiés, y compris le champ `status` de `locked_eight_sleeve_config.json`. Le script `relock_eight_sleeves.py` sert de harnais au lock du 15 novembre ; ne pas le relancer pour produire une référence active.
