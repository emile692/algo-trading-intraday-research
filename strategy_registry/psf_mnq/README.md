# Registre MNQ PineScriptForge — snapshot du 18 septembre 2026

Ce dossier est un **inventaire d'hypothèses** indépendant des campagnes de backtest du dépôt. Il contient [185 fiches](cards/) et un [index CSV](index.csv) filtrable. Chaque fiche donne le lien source, la description et les règles d'entrée/sortie annoncées par le site, les indicateurs et horizons cités, puis une première qualification de recherche. Les familles, rôles et besoins en données sont provisoires. Toutes les fiches restent au statut `Idea`.

La sélection courante est documentée dans le [triage des 185 fiches](TRIAGE.md),
le [tri par rôle architectural](ARCHITECTURE_ROLES.md) et la
[revue logique des 43 moteurs d'alpha provisoires](ALPHA_ENGINE_LOGICAL_AUDIT.md).
L'audit du cutoff ORB et la première baseline Gap Fill sur IS restent des
travaux antérieurs dont les conclusions sont conservées dans le
[registre des recherches non promues](../../docs/RESEARCH_FAILURE_REGISTER_2026.md) ;
leurs rapports générés n'ont pas été récupérés.

## Périmètre exact

- **184 fiches** (`ranked_184`) reproduisent le filtre d'affichage courant du classement MNQ à partir du module JavaScript public du site.
- **1 fiche** (`supplemental_page`) est [RSI Mean Reversion](cards/rsi-mean-reversion.md), dont la page MNQ est publiée mais qui est en dehors du filtre courant. Elle complète les 185 fiches demandées sans la présenter comme une ligne du classement.
- Le module source décrit **310 stratégies** au total. Quatre paires de fiches portent le même nom; elles sont signalées dans le [manifest](manifest.json) et dans les fiches concernées. Le nombre de mécanismes distincts reste à établir.

Les pages indexées par le moteur de recherche affichaient parfois 185 lignes, parfois 184. Ce snapshot ne prétend pas reconstruire une ancienne version exacte du classement. L'ordre de l'index suit le catalogue source, pas le score du site.

## Provenance et limites

Le [manifest](manifest.json) consigne l'URL et l'empreinte SHA-256 du module consulté. Les champs descriptifs proviennent de la liste de stratégies intégrée à ce module et sont attribués sur chaque fiche aux pages publiques correspondantes. Les descriptions ne constituent pas une spécification de backtest complète : il faudra fixer les paramètres manquants, la convention temporelle, les règles d'exécution et les cas limites avant d'implémenter une candidate.

Le filtre `>=300 trades` a été reproduit pour sélectionner les lignes affichées, mais le code du module calcule ce nombre par un générateur déterministe dépendant du nom de stratégie, du symbole et du premier horizon. Les performances affichées sur le site ne sont donc **pas importées** dans le registre et ne servent ni à classer ni à valider les idées. Cela ne permet pas de conclure sur les performances réelles d'une stratégie.

L'ORB MNQ retenu dans ce dépôt reste la seule référence sélectionnée dans notre recherche. Ses chiffres corrigés sont consignés dans le [mémo institutionnel](../../docs/mnq_orb_strategy_institutional_memo.md). La fiche [Opening Range Breakout (ORB)](cards/opening-range-breakout.md) décrit uniquement une idée externe et ne remplace pas cette référence.

## Utilisation

Ouvrir `index.csv` pour le registre source, `collaborative_universe.csv` pour
les 134 fiches retenues après les exclusions décidées ensemble,
`architecture_roles.csv` pour les cinq rôles, puis
`alpha_engine_logical_audit.csv` pour les décisions détaillées sur les 43
moteurs provisoires. Les fiches restent des hypothèses même si un prototype du
même nom a été exécuté.

Les 16 mécanismes représentatifs disposent ensuite de
[spécifications canoniques figées](FROZEN_BASELINES_REVIEW.md), listées dans
[`frozen_baselines_manifest.csv`](frozen_baselines_manifest.csv). Le dossier
[`frozen_baselines/`](frozen_baselines/) contient un JSON exécutable par
mécanisme et le benchmark séparé `MNQ_BUY_AND_HOLD`.

Pour reconstruire **ce snapshot** depuis le même module téléchargé :

```powershell
python -m pip install json5
python strategy_registry/psf_mnq/build_cards.py chemin/vers/index-DJyf-tLw.js
```

Le script vérifie qu'il retrouve 184 lignes du filtre et la page supplémentaire. Si le site change, il s'arrête pour imposer une nouvelle revue du périmètre. Aucun score de performance n'est sauvegardé.
