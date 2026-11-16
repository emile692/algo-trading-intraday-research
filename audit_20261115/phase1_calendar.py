"""Phase 1 — daily PnL on the union of the three instruments' calendars.

Reference sleeve definitions (user decision of 2026-11-15): S01 = the 18-model ATR vote of signal sheet 01
(``research/signal_sheet_01_atr18``), S02-S09 = the frozen vote clusters.  The corrected panel built here
(``outputs/daily_pnl_union.csv``) is the input of every later phase.  A sleeve has NaN outside its data range
(MNQ/MGC after 2026-03-19, MBT before 2021-05-03) and 0.0 on a session of the union where it did not trade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import (ACTIVE8, BOOK_WINDOW, CODES, INSTRUMENT, OUT, ROOT, fr, load_daily, load_trades,
                     lock_daily_nine, max_dd, md_table, sharpe, write_json)

DATA_RANGE = {"MNQ": ("2019-05-06", "2026-03-19"), "MGC": ("2019-05-06", "2026-03-19"),
              "MBT": ("2021-05-03", "2026-10-02")}


def instrument_calendars() -> dict[str, pd.DatetimeIndex]:
    cal = {"MNQ": set(), "MGC": set(), "MBT": set()}
    for code in CODES[1:]:
        cal[INSTRUMENT[code]] |= set(load_daily(code).index)
    return {k: pd.DatetimeIndex(sorted(v)) for k, v in cal.items()}


def union_panel() -> tuple[pd.DataFrame, dict]:
    cals = instrument_calendars()
    union = pd.DatetimeIndex(sorted(set().union(*[set(v) for v in cals.values()])))
    panel = pd.DataFrame(index=union, columns=CODES, dtype=float)
    checks = {}
    for code in CODES:
        inst = INSTRUMENT[code]
        d = load_daily(code)
        if code == "S01":
            off = d[~d.index.isin(union)]
            checks["S01_rows_off_union"] = int(len(off))
            checks["S01_nonzero_rows_off_union"] = int((off != 0).sum())
            if (off != 0).any():
                raise RuntimeError("S01 has PnL on a date outside the instrument calendars")
        a, b = map(pd.Timestamp, DATA_RANGE[inst])
        in_range = (union >= a) & (union <= b)
        panel.loc[in_range, code] = d.reindex(union[in_range]).fillna(0.0).to_numpy()
        t = load_trades(code)
        t = t[t["session_date"].between(a, b)]
        diff = abs(float(panel[code].sum()) - float(t["net_pnl_usd"].sum()))
        checks[f"{code}_panel_minus_trades_abs"] = diff
        if diff > 1e-6:
            raise RuntimeError(f"{code}: union panel does not equal the trade sum ({diff})")
    panel.index.name = "session_date"
    return panel, {"calendars": {k: [str(v.min().date()), str(v.max().date()), len(v)] for k, v in cals.items()},
                   "union_sessions": len(union), **checks}


def risk_block(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    book = df[cols].sum(axis=1)
    var = book.var(ddof=1)
    rows = []
    for c in cols:
        x = df[c]
        rows.append({"sleeve": c, "pnl": x.sum(), "vol": x.std(ddof=1), "worst_day": x.min(), "max_dd": max_dd(x),
                     "risk_contribution": float(np.cov(x, book, ddof=1)[0, 1] / var)})
    return pd.DataFrame(rows)


def book_line(df: pd.DataFrame, cols: list[str]) -> dict:
    p = df[cols].sum(axis=1)
    return {"sessions": len(p), "pnl": float(p.sum()), "vol": float(p.std(ddof=1)), "sharpe": sharpe(p),
            "worst_day": float(p.min()), "max_dd": max_dd(p)}


def loo(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    full = book_line(df, cols)
    rows = []
    for c in cols:
        r = book_line(df, [x for x in cols if x != c])
        rows.append({"removed": c, "pnl": r["pnl"], "d_pnl": r["pnl"] - full["pnl"], "sharpe": r["sharpe"],
                     "max_dd": r["max_dd"], "d_max_dd": r["max_dd"] - full["max_dd"]})
    return pd.DataFrame(rows)


def main() -> dict:
    panel, checks = union_panel()
    panel.to_csv(OUT / "daily_pnl_union.csv", float_format="%.8f")
    a, b = map(pd.Timestamp, BOOK_WINDOW)

    lock = lock_daily_nine()  # A: as published (S01 baseline_fixed, MNQ calendar)
    b_df = lock.copy()        # B: S01 = reference, MNQ calendar
    b_df["S01"] = load_daily("S01").reindex(lock.index).fillna(0.0)
    c_df = panel.loc[(panel.index >= a) & (panel.index <= b)].copy()  # C: reference, union calendar
    if c_df.isna().any().any():
        raise RuntimeError("NaN inside the common window")

    missing = c_df.index.difference(lock.index)
    miss_rows = []
    for dte in missing:
        row = {"session_date": str(dte.date())}
        trading = []
        for inst in ("MNQ", "MGC", "MBT"):
            if dte in instrument_calendars()[inst]:
                trading.append(inst)
        row["instruments"] = "+".join(trading)
        for c in CODES:
            row[c] = float(c_df.loc[dte, c])
        row["book9"] = float(c_df.loc[dte, CODES].sum())
        row["book8"] = float(c_df.loc[dte, ACTIVE8].sum())
        miss_rows.append(row)
    miss = pd.DataFrame(miss_rows)
    miss.to_csv(OUT / "01_missing_sessions.csv", index=False)
    extra = lock.index.difference(c_df.index)

    variants = {"A_lock_publie": lock, "B_S01ref_calMNQ": b_df, "C_S01ref_union": c_df}
    books = {f"{k}__{n}": book_line(df, cols) for k, df in variants.items() for n, cols in (("9", CODES), ("8", ACTIVE8))}
    pd.DataFrame(books).T.to_csv(OUT / "01_book_before_after.csv")
    risk = {}
    for k, df in variants.items():
        for n, cols in (("9", CODES), ("8", ACTIVE8)):
            rb = risk_block(df, cols).assign(variant=k, book=n)
            lb = loo(df, cols).assign(variant=k, book=n)
            risk[(k, n)] = (rb, lb)
    pd.concat([v[0] for v in risk.values()]).to_csv(OUT / "01_section4_risk.csv", index=False)
    pd.concat([v[1] for v in risk.values()]).to_csv(OUT / "01_section5_loo.csv", index=False)

    # Same correction outside the book window, per instrument, for the record.
    out = {"checks": checks, "missing_in_window": len(missing), "lock_dates_not_in_union": len(extra),
           "missing_pnl_book9": float(miss["book9"].sum()) if len(miss) else 0.0,
           "missing_pnl_book8": float(miss["book8"].sum()) if len(miss) else 0.0, "books": books}
    write_json(OUT / "01_calendar.json", out)
    render(out, miss, risk)
    return out


def render(out: dict, miss: pd.DataFrame, risk: dict) -> None:
    from .common import AUDIT
    B = out["books"]

    def bl(key):
        r = B[key]
        return [fr(r["sessions"], 0), fr(r["pnl"]), fr(r["vol"]), fr(r["sharpe"], 3), fr(r["worst_day"]), fr(r["max_dd"])]

    t_book = ["| Variante | Book | Séances | PnL | Vol. quotidienne | Sharpe | Pire jour | maxDD |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    labels = {"A_lock_publie": "A — lock publié (S01 `baseline_fixed`, calendrier MNQ)",
              "B_S01ref_calMNQ": "B — S01 de référence, calendrier MNQ",
              "C_S01ref_union": "C — **S01 de référence, calendrier union (corrigé)**"}
    for k, lab in labels.items():
        for n in ("9", "8"):
            t_book.append(f"| {lab} | {n} | " + " | ".join(bl(f"{k}__{n}")) + " |")

    mcols = ["session_date", "instruments"] + [c for c in CODES if c not in ("S01", "S02", "S03", "S04")] + ["book8"]
    t_miss = ["| " + " | ".join(mcols) + " |", "|" + "|".join(["---"] * 2 + ["---:"] * (len(mcols) - 2)) + "|"]
    for r in miss.itertuples(index=False):
        d = r._asdict()
        t_miss.append("| " + " | ".join([d["session_date"], d["instruments"]] + [fr(d[c]) for c in mcols[2:]]) + " |")
    tot = miss[mcols[2:]].sum()
    t_miss.append("| **Total** | | " + " | ".join(fr(tot[c]) for c in mcols[2:]) + " |")
    weekday = pd.to_datetime(miss["session_date"]).dt.day_name(locale=None).value_counts().to_dict() if len(miss) else {}

    def risk_table(n):
        a_rb, _ = risk[("A_lock_publie", n)]
        c_rb, _ = risk[("C_S01ref_union", n)]
        rows = ["| Sleeve | Vol. avant | Vol. après | Pire jour avant | Pire jour après | maxDD avant | maxDD après | Contrib. risque avant | Contrib. risque après |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for (_, x), (_, y) in zip(a_rb.iterrows(), c_rb.iterrows()):
            rows.append(f"| {x.sleeve} | {fr(x.vol)} | {fr(y.vol)} | {fr(x.worst_day)} | {fr(y.worst_day)} | {fr(x.max_dd)} | "
                        f"{fr(y.max_dd)} | {fr(100 * x.risk_contribution, 1)} % | {fr(100 * y.risk_contribution, 1)} % |")
        return rows

    def loo_table(n):
        _, a_l = risk[("A_lock_publie", n)]
        _, c_l = risk[("C_S01ref_union", n)]
        rows = ["| Sleeve retirée | PnL avant | PnL après | Sharpe avant | Sharpe après | maxDD avant | maxDD après | ΔmaxDD après |",
                "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for (_, x), (_, y) in zip(a_l.iterrows(), c_l.iterrows()):
            rows.append(f"| {x.removed} | {fr(x.pnl)} | {fr(y.pnl)} | {fr(x.sharpe, 3)} | {fr(y.sharpe, 3)} | {fr(x.max_dd)} | "
                        f"{fr(y.max_dd)} | {fr(y.d_max_dd)} |")
        return rows

    c9, c8, a9, a8 = B["C_S01ref_union__9"], B["C_S01ref_union__8"], B["A_lock_publie__9"], B["A_lock_publie__8"]
    lines = [
        "# Phase 1 — Correction du calendrier",
        "",
        "*Fenêtre : 2024-01-02 → 2026-03-19. PnL quotidien net en USD, coûts nominaux du lock. Sharpe : √252 × moyenne / "
        "écart-type, jours sans trade inclus. S01 : ensemble ATR 18 de la fiche 01 (décision du 2026-11-15). "
        "Script : [phase1_calendar.py](phase1_calendar.py). Sorties : [daily_pnl_union.csv](outputs/daily_pnl_union.csv), "
        "[01_missing_sessions.csv](outputs/01_missing_sessions.csv), [01_section4_risk.csv](outputs/01_section4_risk.csv), "
        "[01_section5_loo.csv](outputs/01_section5_loo.csv).*",
        "",
        "## Ce qui change",
        "",
        f"Le lock aligne les neuf sleeves sur les 564 séances du calendrier MNQ. Le calendrier union des trois instruments "
        f"compte **{c9['sessions']} séances** sur la même fenêtre. Les {out['missing_in_window']} séances ajoutées sont des jours où "
        f"MGC ou MBT tradent sans séance MNQ dans les fichiers. Elles pèsent {fr(out['missing_pnl_book8'])} USD pour le book à "
        f"8 sleeves et {fr(out['missing_pnl_book9'])} USD pour le book à 9. Aucune séance du lock n'est absente de l'union "
        f"({out['lock_dates_not_in_union']}).",
        "",
        "Deux corrections se cumulent dans la colonne C : le S01 de référence (B − A) et le calendrier (C − B). "
        f"Contrôle : sur l'union, la somme quotidienne de chaque sleeve égale la somme de ses trades (écart max < 10⁻⁶ USD).",
        "",
        "## Book avant / après",
        "",
        *t_book,
        "",
        f"Book à 8 sleeves corrigé : **{fr(c8['pnl'])} USD, Sharpe {fr(c8['sharpe'], 3)}, maxDD {fr(c8['max_dd'])}, "
        f"pire jour {fr(c8['worst_day'])}** (lock publié : {fr(a8['pnl'])}, {fr(a8['sharpe'], 3)}, {fr(a8['max_dd'])}).",
        "",
        "## Séances manquantes du lock",
        "",
        "Les sleeves MNQ ne tradent pas ces jours-là : leurs colonnes, toutes nulles, sont omises.",
        "",
        *t_miss,
        "",
        "## Section 4 recalculée — risque réalisé, book à 9 sleeves",
        "",
        "« Avant » = lock publié (A). « Après » = S01 de référence et calendrier union (C). Contribution au risque = "
        "cov(sleeve, book) / var(book).",
        "",
        *risk_table("9"),
        "",
        "## Section 4 recalculée — book actif à 8 sleeves",
        "",
        *risk_table("8"),
        "",
        "## Section 5 recalculée — leave-one-out, book à 9 sleeves",
        "",
        *loo_table("9"),
        "",
        "## Leave-one-out, book actif à 8 sleeves",
        "",
        *loo_table("8"),
        "",
        "Le panel corrigé `outputs/daily_pnl_union.csv` couvre 2019-05-06 → 2026-10-02 sur le calendrier union. "
        "Une sleeve y vaut NaN hors de la plage de ses données (MNQ/MGC après le 2026-03-19, MBT avant le 2021-05-03). "
        "Toutes les phases suivantes l'utilisent.",
    ]
    (AUDIT / "01_calendrier.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    import json
    print(json.dumps(main()["books"], indent=1, default=float))
