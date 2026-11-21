# Référence remplacée — conservée pour audit

Le portefeuille `eight_sleeve_equal_contract_reference_20261115` défini dans ce répertoire n'est plus la référence active depuis le **21 novembre 2026**.

Il est remplacé par `eight_sleeve_equal_contract_reference_20261121`, défini dans [`research/locked_eight_sleeve_portfolio_20261121/`](../locked_eight_sleeve_portfolio_20261121/locked_eight_sleeve_config.json). Seule la **donnée MNQ** change : MNQ `c.0` est remplacé par MNQ `v.0` (le contrat le plus échangé, celui que trade la production), ajusté en arrière de l'écart de roll. Décision de l'utilisateur du 2026-11-21.

`c.0` posait trois problèmes :
- il gardait le contrat qui expire pendant la semaine du roll ;
- il n'avait aucune barre RTH les vendredis d'échéance ;
- il créait un faux gap (jusqu'à 276 points) dans l'ATR quotidien au changement de contrat.

Les sleeves S01–S04 sont reconstruites sur `v.0` avec les mêmes moteurs, paramètres, manifestes de vote et seuils IS de S01. Les sleeves MGC et MBT sont inchangées. Les seuils de la règle d'arrêt des sleeves MNQ et du book sont recalculés sur `v.0`.

Les fichiers de ce répertoire, hachés dans `lock_manifest.json`, ne sont volontairement pas modifiés, y compris le champ `status` de `locked_eight_sleeve_config.json`. Le script `relock_eight_sleeves_20261115.py` sert de harnais au lock du 21 novembre ; ne pas le relancer pour produire une référence active.
