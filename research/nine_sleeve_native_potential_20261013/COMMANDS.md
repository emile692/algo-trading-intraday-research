# Commandes reproductibles

Depuis la racine du repository, sous PowerShell :

```powershell
.\.venv\Scripts\python.exe -m py_compile research/nine_sleeve_native_potential_20261013/run_campaign.py
.\.venv\Scripts\python.exe research/nine_sleeve_native_potential_20261013/run_campaign.py
.\.venv\Scripts\python.exe research/nine_sleeve_native_potential_20261013/finalize_existing.py
```

Le second appel régénère les ledgers natifs MNQ/MGC, charge les ensembles exécutables MBT, recalcule toutes les variantes et remplace uniquement les livrables de cette campagne. Le troisième enrichit les audits de reproduction, pertes réalisées et plafonds de compte. Pour reconstruire seulement le rapport à partir des CSV déjà calculés :

```powershell
.\.venv\Scripts\python.exe research/nine_sleeve_native_potential_20261013/finalize_existing.py
```
