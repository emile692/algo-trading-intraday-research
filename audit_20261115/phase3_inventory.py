"""Phase 3 — holdout inventory without reading any value.

For each MNQ / MGC / MBT (and parent NQ / GC) minute file on disk, only the timestamp range is read, from the
parquet row-group statistics of the timestamp column (fallback: the timestamp column alone, never a price
column).  The consulted windows come from the repository's own documentation, cited per row.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .common import AUDIT, OUT, ROOT, write_json

SEARCH = [ROOT / "data" / "downloaded", ROOT / "data" / "processed" / "parquet", ROOT / "data" / "raw",
          Path(r"C:\Data_local\market_data_archive\databento")]
PATTERN = ("MNQ", "MGC", "MBT", "NQ_", "GC_")
TS_COLS = ("timestamp", "ts_event")

# Windows already read, with the document that records the reading.
CONSULTED = [
    ("MNQ", "2019-05-06", "2026-03-19", "IS + validation réutilisée des sleeves MNQ (fiches 01–04)"),
    ("MGC", "2019-05-06", "2026-03-19", "IS + validation réutilisée des sleeves MGC (fiches 05–06)"),
    ("MNQ", "2026-03-20", "2026-10-02", "forward replay des votes gelés S02–S04 et de S01 `baseline_fixed` "
     "(research/frozen_vote_forward_replay_20261101) ; a motivé la sortie de S04 (ACTIVE_EIGHT_SLEEVES.md) ; "
     "déclarée EXT contaminée dans book_aggregation_risk_r1_20261112"),
    ("MGC", "2026-03-20", "2026-10-02", "forward replay S05–S06 (même dossier) ; règle d'arrêt rejouée sur S05 "
     "(forward_context.json)"),
    ("MBT", "2021-05-03", "2026-10-02", "IS + legacy OOS des sleeves MBT (fiches 07–09) ; S09 nul sur son legacy OOS"),
]


def ts_range(path: Path) -> dict:
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    col = next((c for c in TS_COLS if c in names), None)
    if col is None:
        return {"error": "no timestamp column"}
    i = names.index(col)
    tz = str(pf.schema_arrow.field(col).type.tz)
    lo = hi = None
    method = "row-group statistics"
    for g in range(pf.metadata.num_row_groups):
        st = pf.metadata.row_group(g).column(i).statistics
        if st is None or not st.has_min_max:
            method = None
            break
        lo = st.min if lo is None else min(lo, st.min)
        hi = st.max if hi is None else max(hi, st.max)
    if method is None:  # read the timestamp column only
        t = pf.read(columns=[col]).column(col).to_pandas()
        lo, hi, method = t.min(), t.max(), "timestamp column only"
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    if lo.tzinfo is None:
        lo, hi = lo.tz_localize("UTC"), hi.tz_localize("UTC")
    return {"min_utc": str(lo.tz_convert("UTC")), "max_utc": str(hi.tz_convert("UTC")),
            "max_et": str(hi.tz_convert("America/New_York")), "rows": pf.metadata.num_rows, "method": method,
            "column": col, "column_tz": tz}


def main() -> dict:
    rows = []
    for base in SEARCH:
        if not base.exists():
            continue
        for p in sorted(base.rglob("*.parquet")):
            if not p.name.upper().startswith(PATTERN):
                continue
            r = {"file": str(p.relative_to(ROOT)) if ROOT in p.parents else str(p), **ts_range(p)}
            r["instrument"] = next(s.rstrip("_") for s in PATTERN if p.name.upper().startswith(s))
            rows.append(r)
    inv = pd.DataFrame(rows)
    inv.to_csv(OUT / "03_inventory.csv", index=False)
    last = {}
    for inst in ("MNQ", "MGC", "MBT"):
        sub = inv[inv["instrument"] == inst]
        last[inst] = max(sub["max_et"]) if len(sub) else None
    write_json(OUT / "03_inventory.json", {"files": rows, "last_timestamp_et": last})
    render(inv, last)
    return last


def render(inv: pd.DataFrame, last: dict) -> None:
    t = ["| Fichier | Instrument | Premier timestamp (UTC) | Dernier timestamp (ET) | Lignes | Méthode |",
         "|---|---|---|---|---:|---|"]
    for r in inv.itertuples(index=False):
        t.append(f"| `{r.file}` | {r.instrument} | {r.min_utc[:16]} | **{r.max_et[:16]}** | {r.rows:,} | {r.method} |".replace(",", " "))
    c = ["| Instrument | Fenêtre consultée | Trace dans le repo |", "|---|---|---|"]
    for inst, a, b, src in CONSULTED:
        c.append(f"| {inst} | {a} → {b} | {src} |")
    lines = [
        "# Phase 3 — Inventaire du holdout",
        "",
        "*Lu le 2026-11-15. Seuls les timestamps ont été lus, depuis les statistiques des row groups parquet. Aucun "
        "prix, rendement ni indicateur n'a été calculé. Script : [phase3_inventory.py](phase3_inventory.py). Table : "
        "[03_inventory.csv](outputs/03_inventory.csv).*",
        "",
        "## Dernier timestamp disponible par instrument",
        "",
        f"- **MNQ** : {last['MNQ'][:16]} ET",
        f"- **MGC** : {last['MGC'][:16]} ET",
        f"- **MBT** : {last['MBT'][:16]} ET",
        "",
        "## Fichiers",
        "",
        *t,
        "",
        "## Fenêtres déjà consultées",
        "",
        *c,
        "",
        "## Verdict",
        "",
        "**Aucune donnée vierge n'existe sur disque pour les signaux tels qu'ils sont tradés.**",
        "",
        "- MNQ et MGC du 2026-03-20 au dernier timestamp : rejoués le 2026-11-01 pour toutes les sleeves MNQ/MGC "
        "du book. C'est cette fenêtre qui a fait sortir S04. Elle n'est plus un holdout.",
        "- MBT après le 2026-10-02 : aucun fichier ne dépasse le 2026-10-02 au soir (séance Globex qui ouvre le "
        "dimanche 2026-10-04 non comprise).",
        "- Quelques minutes de Globex du vendredi 2026-10-02 au soir peuvent dépasser la dernière séance RTH. Elles "
        "n'appartiennent à aucune séance tradée par les sleeves.",
        "",
        "**Cas particulier de S01 (ensemble ATR 18).** Le forward replay du 2026-11-01 a rejoué S01 `baseline_fixed`, "
        "pas l'ensemble ATR 18 de la fiche. Le S01 de référence n'a donc jamais été calculé sur 2026-03-20 → "
        "2026-10-02. Cette fenêtre n'est pas vierge pour autant : les deux versions partagent l'entrée, la sortie et "
        "le sizing, et 100 % des trades ATR 18 de 2024-01 → 2026-03 figurent dans `baseline_fixed`. Le PnL de "
        "`baseline_fixed` sur cette fenêtre est connu. Au mieux, c'est un holdout partiellement contaminé, valable "
        "pour S01 seul.",
        "",
        "**Conséquence pour la phase 4.** La seule validation propre est le suivi prospectif. Il démarre à la première "
        "séance après le 2026-11-03 pour le book à 8 sleeves (registre `forward_monitoring`), et ses données ne sont "
        "pas encore sur disque. Le pré-enregistrement de la phase 4a porte donc sur ce suivi. Il fixe les critères "
        "par horizon (20, 60, 120 et 250 séances), à évaluer quand les données arriveront. Un test one-shot de S01 "
        "ATR 18 sur 2026-03-20 → 2026-10-02 est proposé en option, étiqueté « partiellement contaminé ».",
    ]
    (AUDIT / "03_inventaire_holdout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    print(main())
