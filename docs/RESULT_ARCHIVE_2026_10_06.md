# Archive des résultats de recherche — 6 octobre 2026

Les sorties de campagnes non temporaires ont été déplacées hors du dépôt actif afin de l'alléger sans effacer la trace des recherches non abouties.

## Emplacement et intégrité

- Archive : `C:\Data_local\research-archives\algo-trading-intraday-research-archive-20261006`
- Charge utile : **2 466 fichiers**, **2 500 664 321 octets**
- Manifeste : `ARCHIVE_FILE_MANIFEST_2026_10_06.csv`
- SHA-256 du manifeste : `c31b9ec4376ef81b886aa9529081f84a856f33b05f59d9fcc2ad1c248bfb586f`

Le manifeste contient pour chacun des 2 466 fichiers de charge utile son chemin relatif, sa taille et son SHA-256. Il constitue la preuve d'intégrité et la table de restauration ; le `README.md` ajouté ensuite est une métadonnée hors manifeste.

## Politique

- Une recherche non promue n'est pas effacée : sa conclusion synthétique reste dans [`RESEARCH_FAILURE_REGISTER_2026.md`](RESEARCH_FAILURE_REGISTER_2026.md) et ses sorties détaillées restent dans l'archive.
- Les résultats actifs nécessaires à la référence nine-sleeve restent dans `research/`.
- Les caches et répertoires `_tmp` ne sont pas archivés, car ils sont régénérables et ne constituent pas une preuve de recherche.
- Restaurer un résultat consiste à recopier son chemin relatif depuis l'archive, puis à contrôler son SHA-256 contre le manifeste.

L'archive est externe au dépôt Git : la cloner ou synchroniser le dépôt ne la recrée pas automatiquement.
