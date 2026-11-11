# Référence remplacée — conservée pour audit

Le portefeuille `nine_sleeve_equal_contract_reference_20261004` défini dans ce répertoire n'est plus la référence active depuis le **1er novembre 2026**.

Il est remplacé par `nine_sleeve_equal_contract_reference_20261101`, défini dans [`research/locked_nine_sleeve_portfolio_20261101/`](../locked_nine_sleeve_portfolio_20261101/locked_nine_sleeve_config.json) : même mapping `[1,1,1,1,1,1,1,1,1]`, mais les sleeves S02 à S09 y exécutent leurs ensembles de vote gelés.

Les fichiers de ce répertoire, hachés dans `lock_manifest.json`, ne sont volontairement pas modifiés, y compris le champ `status` de `locked_nine_sleeve_config.json` : ils documentent la décision du 4 octobre 2026 telle qu'elle a été prise. Ne pas relancer `freeze_reference.py` pour produire une référence active.
