# Audit de nettoyage et récupération — 4–6 octobre 2026

## Résultat final

Le dépôt de travail a été reconstruit depuis le remote Git et les journaux locaux Codex après qu'une commande de suppression de caches a dépassé son périmètre le 4 octobre. Le répertoire vide issu de l'incident a été isolé hors du dépôt actif. Cette page remplace le compte rendu antérieur à l'incident.

Ont été récupérés et vérifiés :

- l'historique Git, les modifications locales de code et les scripts de recherche ;
- les neuf sleeves actives, leurs configurations et leurs sorties reproductibles ;
- les données MBT et MGC nécessaires à la reproduction ;
- le registre PineScriptForge et ses 185 fiches ;
- les neuf fiches signal homogènes et le registre des recherches non promues.

Ont été retirés du dépôt actif :

- environnements virtuels et caches Python, pytest, uv, matplotlib et notebooks ;
- douze répertoires d'exports `_tmp` régénérables, soit 263 fichiers ;
- anciens caches de recherche déjà marqués supprimés dans Git ;
- résultats de campagnes non temporaires, déplacés dans une archive externe vérifiable plutôt que supprimés.

L'archive est décrite dans [`RESULT_ARCHIVE_2026_10_06.md`](RESULT_ARCHIVE_2026_10_06.md).

## Restauration Databento

Deux acquisitions autorisées ont été effectuées, une seule requête payante par jeu :

| Jeu | Coût estimé | Fichier canonique | SHA-256 |
|---|---:|---|---|
| MBT 1 minute, 2021-04-25–2026-10-03 | 4,984364956617 USD | `data/downloaded/MBT_v_0_1m_20210425_20261003.parquet` | `0da668f29ba77c58168c5a0902b6f369d5f02f370bf59041ae851a7f4a14a683` |
| MGC 1 minute brut, 2015-01-01–2026-09-23 | 12,008514478803 USD | `data/downloaded/MGC_v_0_1m_20150101_20260923.parquet` | `80ea1f986f3b4a3a954c1b360e145609c805380d2954e739410cd588911b3875` |
| MGC normalisé America/New_York | inclus | `data/processed/parquet/MGC_v_0_1m_20150101_20260923.parquet` | `2bbc2d10d59409fd347779fc00a23633fa6ad6aec0cf4b52e7a6d7a278426e96` |

Total estimé : **16,99287943542 USD**. Databento a signalé quelques journées au statut `degraded`; cet avertissement fournisseur est conservé dans les métadonnées. Aucune clé API n'est stockée dans le dépôt ou dans les fichiers de métadonnées.

Les autres séries non suivies (`6E.v.0`, `6J.v.0`, `BTC.v.0`, `MCL.v.0`, `NG.v.0`, `ZF.v.0`, `ZN.v.0`) n'ont pas été retéléchargées : elles ne sont pas nécessaires à la référence active et occasionneraient une dépense supplémentaire.

## Contrôles de reproductibilité

- Trades MNQ ORB : hash historique reproduit `e4f5d74214127e4fa957bb40d035c08d9eb161796529b90a6f0a5322a997506c`.
- Lock money management : hash historique reproduit `7209d38b2b2e25cc5f9ffc523345a518e7bb728e5ae90889747dadc68ac3c419`.
- Manifeste MGC et deux spécifications : trois hashes historiques reproduits.
- Manifeste d'ensemble MBT et huit fichiers de trades : tous les hashes historiques reproduits.
- Référence nine-sleeve : configuration, PnL quotidien, manifeste de sleeves et métriques Topstep reproduits à l'octet près.
- Intégration six-sleeve : 564 séances, 1 481 trades, zéro violation de look-ahead signalée.

La gate ciblée finale produit :

```text
22 passed, 1 skipped in 1.32s
```

Les liens Markdown sont contrôlés séparément. Le livrable client contient exactement neuf fiches, chacune avec dix sections et un diagramme Mermaid.

## Documentation consolidée

- [`signal_sheets/`](signal_sheets/README.md) : neuf fiches signal client ;
- [`RESEARCH_FAILURE_REGISTER_2026.md`](RESEARCH_FAILURE_REGISTER_2026.md) : recherches rejetées, faibles ou stoppées ;
- [`ACTIVE_NINE_SLEEVES.md`](ACTIVE_NINE_SLEEVES.md) : référence opérationnelle ;
- [`REPOSITORY_MAP.md`](REPOSITORY_MAP.md) : carte du dépôt ;
- [`RESULT_ARCHIVE_2026_10_06.md`](RESULT_ARCHIVE_2026_10_06.md) : archive externe et procédure de contrôle.

## Environnement

L'environnement virtuel est régénérable et reste hors du dépôt :

```powershell
uv sync
```

Les règles `.gitignore` couvrent environnements, caches, temporaires, checkpoints et exports régénérables.
