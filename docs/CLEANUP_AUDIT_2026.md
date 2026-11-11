# Audit du nettoyage 2026

Date de clôture : **26 septembre 2026**.

> Audit historique, remplacé pour l'état courant par
> [`CLEANUP_AUDIT_2026_10_04.md`](CLEANUP_AUDIT_2026_10_04.md). Les deux
> inventaires CSV cités ci-dessous n'ont pas été récupérés après l'incident du
> 4 octobre ; leurs totaux et leur rôle sont conservés ici à titre de trace.

Ce document relie la décision de clôture, l'inventaire des expériences et le
manifeste technique créé avant suppression. Il garantit qu'aucune campagne
testée n'est effacée sans laisser de trace écrite.

## Registre conservé

- `RESEARCH_CLOSURE_2026.md` décrit les hypothèses, principaux résultats et
  décisions pour MES/M2K, 6E, ZN, MCL ainsi que les pistes non retenues.
- `CLOSED_CAMPAIGN_INVENTORY_2026.csv` contenait une ligne par campagne,
  export, notebook ou fichier de support supprimé. Il conserve le chemin
  d'origine, le nombre de fichiers, la taille et les rapports qui existaient.
- `DELETED_FILE_MANIFEST_2026.csv` contenait une ligne pour chacun des 3 657
  fichiers supprimés, avec chemin d'origine, taille et empreinte SHA-256.
- `cleanup_manifest_metadata_2026.json` résume le périmètre et énumère les
  blocs explicitement préservés.
- `ACTIVE_SIX_SLEEVES.md` est la référence concise du portefeuille désormais
  suivi.

Les empreintes ne permettent pas de reconstruire les fichiers supprimés, mais
elles permettent d'identifier précisément ce qui a existé et de vérifier une
éventuelle copie retrouvée plus tard. Les résultats économiques essentiels
sont conservés dans le rapport narratif; les noms et périmètres de toutes les
expériences restent dans les deux manifestes CSV.

## Périmètre supprimé

- exports régénérables sous `data/exports/`;
- campagnes d'export supersédées, sauf l'ORB MNQ retenu;
- campagnes de recherche fermées MES/M2K, 6E, ZN, MCL et intermédiaires MGC;
- notebooks exécutés ou obsolètes, sauf les deux notebooks actifs;
- scripts devenus orphelins et le test de la campagne ciblée 6E/ZN supprimée.

Volume inventorié avant suppression : **2 795 114 545 octets**.

Vérification après nettoyage : **19 tests actifs réussis, 1 ignoré**
(`test_frozen_baseline_engine.py`, `test_backtester.py`,
`test_orb_strategy.py`). La suite historique globale n'est plus une gate :
elle référence encore des campagnes, notebooks et fixtures archivés.

## Périmètre préservé

Les données Databento téléchargées et les Parquet traités sont conservés. Les
artefacts, configurations, rapports et code nécessaires aux quatre sleeves MNQ
et aux deux sleeves MGC restent également dans le dépôt. Aucun résultat actif
n'a été choisi ou recalculé pendant le nettoyage.
