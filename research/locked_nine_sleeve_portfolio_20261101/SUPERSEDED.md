# Référence remplacée — conservée pour audit

Le portefeuille `nine_sleeve_equal_contract_reference_20261101` défini dans ce répertoire n'est plus la référence active depuis le **3 novembre 2026**.

Il est remplacé par `eight_sleeve_equal_contract_reference_20261103`, défini dans [`research/locked_eight_sleeve_portfolio_20261103/`](../locked_eight_sleeve_portfolio_20261103/locked_eight_sleeve_config.json) : mêmes sources gelées pour huit sleeves, `mnq_atr_trend_flip` passe en `SHADOW_FORWARD_ONLY`.

Les fichiers de ce répertoire, hachés dans `lock_manifest.json`, ne sont volontairement pas modifiés, y compris le champ `status` de `locked_nine_sleeve_config.json`. Le script `relock_nine_sleeves.py` sert de harnais au lock du 3 novembre ; ne pas le relancer pour produire une référence active.
