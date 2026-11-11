# Données locales

Les Parquet ne sont pas destinés au suivi Git.

- `downloaded/` : téléchargements sources à conserver pour éviter de repayer les données ;
- `processed/parquet/` : versions normalisées utilisées par les campagnes ;
- `raw/` : petites sources auxiliaires ;
- `exports/` : sorties de campagnes ; seuls les sous-répertoires `_tmp` sont
  automatiquement supprimables.

## État après récupération

Présents depuis le remote : MNQ, NQ, MES, M2K et l'ancien MGC continu.

Restaurés avec autorisation Databento : `MBT.v.0` brut et `MGC.v.0` brut puis normalisé. Les hashes historiques du MBT et du MGC traité sont reproduits ; les ledgers des trois ensembles MBT et le lock nine-sleeve complet ont été régénérés.

À restaurer uniquement si une future campagne le justifie et après autorisation d'une nouvelle dépense : `6E.v.0`, `6J.v.0`, `BTC.v.0`, `MCL.v.0`, `NG.v.0`, `ZF.v.0` et `ZN.v.0`.

Les résultats non temporaires ont été déplacés dans l'archive externe documentée
par [`docs/RESULT_ARCHIVE_2026_10_06.md`](../docs/RESULT_ARCHIVE_2026_10_06.md).
`exports/` ne contient plus que son marqueur de répertoire.

Ne jamais stocker de clé API dans ce dossier ou dans Git.
