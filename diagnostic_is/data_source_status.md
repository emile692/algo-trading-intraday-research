# Statut des données externes requises

Ce statut précède les étapes 2 et 3. Aucune variable dépendant d'une source
manquante n'a été calculée.

| Donnée | Statut local | Source documentée possible | Couverture utile | Décision |
|---|---|---|---|---|
| VIX | `data/raw/vix-cboe.csv`, SHA256 `5c2f…bb32` | Cboe, `VIX_History.csv` | 1990-01-02 à 2026-10-07 | Intégrée et utilisée par A2 |
| VIX3M | `data/raw/vix3m-cboe.csv`, SHA256 `27e8…e23c` | Cboe, `VIX3M_History.csv` | 2009-09-18 à 2026-10-07 | Intégrée et utilisée par A2 |
| CPI et NFP | Calendrier local inutilisable: seulement 10 événements de 2026 | Archives annuelles officielles BLS | 2010-2024 pour A; 2015-2023 pour B | Construction documentée possible; A1/B1 bloquées jusque-là |
| FOMC | Absent sur l'IS | Calendriers et communiqués historiques de la Federal Reserve | 2010-2024 et 2015-2023 | Construction documentée possible; A1/B1 bloquées jusque-là |
| Autres annonces de 08:30 ET | Absentes | Nécessiteraient plusieurs calendriers officiels BLS/BEA/Census | À auditer source par source | Non testées pour l'instant |
| Volume GC | Absent | Databento `GLBX.MDP3`, parent `GC` par échéance ou continu `GC.v.0` | Dataset disponible depuis juin 2010, couvrant tout B | Potentiellement payant; B0b bloquée sans estimation et autorisation |

Sources:

- Cboe historical indices: https://www.cboe.com/tradable_products/vix/vix_historical_data/
- VIX3M CSV Cboe: https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX3M_History.csv
- BLS annual release calendars: https://www.bls.gov/schedule/2010/home.htm
- Federal Reserve FOMC calendars: https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
- Databento continuous futures: https://databento.com/docs/examples/symbology/continuous
- Databento dataset coverage: https://databento.com/docs/knowledge-base/datasets

