"""Phase 8 — concentration and regimes of the 8-sleeve book (S01 = ATR 18, union calendar).

Daily bars per instrument from the 1-minute files, RTH 09:30-16:00 ET, cut at 2026-03-19 (MNQ, MGC) and
2026-10-02 (MBT).  Regimes, each labelled:
  ATR%  : ATR14 / close of day J-1 (ex ante), terciles fixed on the instrument's IS.  ATR is normalised by price
          because the raw ATR in points trends with the price level (see S01, phase 4);
  JOUR  : |close - open| / (high - low) of the MNQ RTH session of the day (ex post), terciles fixed on the MNQ IS;
  ER20  : efficiency ratio over 20 sessions of the instrument, ending J-1 (ex ante) and centred on the day (ex post).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .common import ACTIVE8, AUDIT, CACHE, CODES, INSTRUMENT, OUT, ROOT, WINDOWS, fr, max_dd, sharpe, write_json
from .phase1_calendar import instrument_calendars, risk_block

FILES = {"MNQ": ("data/processed/parquet/MNQ_c_0_1m_20260321_094501.parquet", "2026-03-19"),
         "MGC": ("data/processed/parquet/MGC_v_0_1m_20150101_20260923.parquet", "2026-03-19"),
         "MBT": ("data/downloaded/MBT_v_0_1m_20210425_20261003.parquet", "2026-10-02")}
IS_END = {"MNQ": "2023-12-31", "MGC": "2023-12-31", "MBT": "2025-05-28"}
PV = {"MNQ": 2.0, "MGC": 10.0, "MBT": 0.1}
BOOK = ("2021-05-03", "2026-03-19")


def daily_bars(inst: str) -> pd.DataFrame:
    cache = CACHE / f"daily_rth_{inst}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    path, cutoff = FILES[inst]
    path = ROOT / path
    col = next(c for c in ("timestamp", "ts_event") if c in pq.read_schema(path).names)
    tz = str(pq.read_schema(path).field(col).type.tz)
    end = (pd.Timestamp(cutoff, tz="America/New_York") + pd.Timedelta(days=1)).tz_convert(tz)
    b = pd.read_parquet(path, columns=["open", "high", "low", "close"], filters=[(col, "<", end)])
    b.index = b.index.tz_convert("America/New_York")
    b = b[~b.index.duplicated(keep="last")].sort_index()
    m = b.index.hour * 60 + b.index.minute
    r = b[(m >= 570) & (m < 960) & (b.index.dayofweek < 5)]
    g = r.groupby(r.index.date)
    d = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(), "close": g["close"].last()})
    d.index = pd.to_datetime(d.index)
    d.to_parquet(cache)
    return d


def regimes() -> dict[str, pd.DataFrame]:
    out = {}
    for inst in ("MNQ", "MGC", "MBT"):
        d = daily_bars(inst).copy()
        tr = pd.concat([d["high"] - d["low"], (d["high"] - d["close"].shift()).abs(), (d["low"] - d["close"].shift()).abs()], axis=1).max(axis=1)
        atrp = (tr.rolling(14).mean() / d["close"]).shift(1)
        er = lambda c: (c - c.shift(20)).abs() / c.diff().abs().rolling(20).sum()
        er_end = er(d["close"])
        r = pd.DataFrame(index=d.index)
        r["atrp_j1"] = atrp
        r["er20_exante"] = er_end.shift(1)
        r["er20_expost"] = er_end.shift(-10)  # window t-9 .. t+10
        r["jour"] = (d["close"] - d["open"]).abs() / (d["high"] - d["low"]).replace(0, np.nan)
        is_mask = r.index <= IS_END[inst]
        for col in ("atrp_j1", "er20_exante", "er20_expost", "jour"):
            q1, q2 = r.loc[is_mask, col].quantile([1 / 3, 2 / 3])
            r[f"{col}_t"] = pd.cut(r[col], [-np.inf, q1, q2, np.inf], labels=["bas", "moyen", "haut"])
        out[inst] = r
    return out


LABELS = {"atrp_j1_t": "ATR% J−1 (ex ante)", "jour_t": "Jour MNQ abs(C−O)/(H−L) (ex post)",
          "er20_exante_t": "ER20 fin J−1 (ex ante)", "er20_expost_t": "ER20 centré (ex post)"}


def by_regime(pnl: pd.Series, lab: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"pnl": pnl, "reg": lab.reindex(pnl.index)}).dropna()
    rows = []
    for k, g in df.groupby("reg", observed=True):
        traded = g["pnl"][g["pnl"] != 0]
        rows.append({"régime": k, "séances": len(g), "pnl": g["pnl"].sum(), "pnl_moyen": g["pnl"].mean(),
                     "sharpe": sharpe(g["pnl"]) if len(g) > 2 else np.nan,
                     "jours_gagnants": float((traded > 0).mean()) if len(traded) else np.nan})
    return pd.DataFrame(rows)


def net_positions() -> dict:
    from .netting import run_ref, RS
    r = run_ref(*BOOK, keep_paths=True)
    T = RS.GRID_MINUTES
    start = 60  # 09:30 is minute 60 of the 08:30 grid
    levels = {i: {} for i in range(3)}
    max_abs = np.zeros(3)
    max_simul = {"usd_per_point": None}
    simul_rows = []
    for date, res in zip(r["dates"], r["table"]):
        pos = np.zeros((T, 3), dtype=int)
        cur = np.zeros(3, dtype=int)
        fills = sorted(res.fills, key=lambda f: f["minute"])
        k = 0
        for t in range(T):
            while k < len(fills) and fills[k]["minute"] == t:
                cur[fills[k]["instrument"]] += fills[k]["delta"]
                k += 1
            pos[t] = cur
        seg = pos[start:]
        for i in range(3):
            vals, cnt = np.unique(seg[:, i], return_counts=True)
            for v, c in zip(vals, cnt):
                levels[i][int(v)] = levels[i].get(int(v), 0) + int(c)
            max_abs[i] = max(max_abs[i], np.abs(seg[:, i]).max())
        simul_rows.append({"date": date, "max_mnq": int(np.abs(seg[:, 0]).max()), "max_mgc": int(np.abs(seg[:, 1]).max()),
                           "max_mbt": int(np.abs(seg[:, 2]).max()),
                           "max_lots": float((np.abs(seg) @ np.array([1, 1, 10])).max() / 10.0)})
    names = ["MNQ", "MGC", "MBT"]
    lv = []
    for i in range(3):
        tot = sum(levels[i].values())
        for v in sorted(levels[i]):
            lv.append({"instrument": names[i], "position_nette": v, "part_du_temps": levels[i][v] / tot})
    sim = pd.DataFrame(simul_rows)
    return {"levels": pd.DataFrame(lv), "max_abs": {names[i]: int(max_abs[i]) for i in range(3)},
            "max_usd_per_point": {names[i]: float(max_abs[i] * PV[names[i]]) for i in range(3)},
            "max_lots": float(sim["max_lots"].max()), "sessions": len(sim),
            "sessions_all_three_open": int(((sim[["max_mnq", "max_mgc", "max_mbt"]] > 0).all(axis=1)).sum())}


def main() -> dict:
    p = pd.read_csv(OUT / "daily_pnl_union.csv", parse_dates=["session_date"]).set_index("session_date")
    book = p[(p.index >= BOOK[0]) & (p.index <= BOOK[1])]
    rc = {w: risk_block(book[(book.index >= a) & (book.index <= b)], ACTIVE8)
          for w, (a, b) in {"2024-01-02 → 2026-03-19": ("2024-01-02", "2026-03-19"),
                            "2021-05-03 → 2026-03-19": BOOK}.items()}
    reg = regimes()
    cals = instrument_calendars()
    # Regime tables: book on MNQ regimes (+ ATR% of each instrument), sleeves on their own instrument.
    b8 = book[ACTIVE8].sum(axis=1)
    rows = []
    for col in LABELS:
        t = by_regime(b8, reg["MNQ"][col]).assign(série="Book 8", instrument_régime="MNQ", variable=LABELS[col])
        rows.append(t)
    for inst in ("MGC", "MBT"):
        rows.append(by_regime(b8, reg[inst]["atrp_j1_t"]).assign(série="Book 8", instrument_régime=inst, variable=LABELS["atrp_j1_t"]))
    for c in CODES:
        inst = INSTRUMENT[c]
        s = p[c].reindex(cals[inst]).dropna()
        for col in LABELS:
            src = reg["MNQ"] if col == "jour_t" else reg[inst]
            rows.append(by_regime(s, src[col]).assign(série=c, instrument_régime="MNQ" if col == "jour_t" else inst, variable=LABELS[col]))
    rt = pd.concat(rows, ignore_index=True)
    rt.to_csv(OUT / "08_regimes.csv", index=False)

    # Loss clustering on the book.
    lab = pd.DataFrame({"jour": reg["MNQ"]["jour_t"], "er_post": reg["MNQ"]["er20_expost_t"], "atr_mnq": reg["MNQ"]["atrp_j1_t"]})
    wk = book[ACTIVE8].groupby(pd.Grouper(freq="W-FRI")).sum()
    wk["book"] = wk.sum(axis=1)
    worst_w = wk.nsmallest(10, "book")
    mo = book[ACTIVE8].groupby(pd.Grouper(freq="ME")).sum()
    mo["book"] = mo.sum(axis=1)
    worst_m = mo.nsmallest(5, "book")

    def dominant(period_end, freq):
        if freq == "W":
            mask = (lab.index > period_end - pd.Timedelta(days=7)) & (lab.index <= period_end)
        else:
            mask = (lab.index.to_period("M") == period_end.to_period("M"))
        sub = lab[mask]
        return {k: (sub[k].value_counts().idxmax() if sub[k].notna().any() else "n/d") for k in sub}

    def rows_for(tab, freq):
        out = []
        for idx, r in tab.iterrows():
            worst_sleeve = r[ACTIVE8].idxmin()
            dm = dominant(idx, freq)
            out.append({"période": str(idx.date()), "book": r["book"], "pire sleeve": worst_sleeve,
                        "PnL pire sleeve": r[worst_sleeve], "sleeves perdantes": int((r[ACTIVE8] < 0).sum()),
                        "jour MNQ dominant": dm["jour"], "ER20 MNQ dominant (ex post)": dm["er_post"],
                        "ATR% MNQ dominant": dm["atr_mnq"]})
        return pd.DataFrame(out)

    ww, wm = rows_for(worst_w, "W"), rows_for(worst_m, "M")
    ww.to_csv(OUT / "08_worst_weeks.csv", index=False)
    wm.to_csv(OUT / "08_worst_months.csv", index=False)
    losing = (book[ACTIVE8] < 0).sum(axis=1)
    freq4 = float((losing >= 4).mean())
    jl = pd.DataFrame({"n_los": losing, "book": b8, "jour": reg["MNQ"]["jour_t"].reindex(book.index),
                       "er": reg["MNQ"]["er20_expost_t"].reindex(book.index),
                       "er_ante": reg["MNQ"]["er20_exante_t"].reindex(book.index)})
    clus = []
    for col, name in (("jour", "Jour MNQ (ex post)"), ("er", "ER20 centré (ex post)"), ("er_ante", "ER20 fin J−1 (ex ante)")):
        for k, g in jl.dropna(subset=[col]).groupby(col, observed=True):
            clus.append({"variable": name, "régime": k, "séances": len(g), "P(≥4 sleeves perdantes)": float((g["n_los"] >= 4).mean()),
                         "PnL book": float(g["book"].sum()), "PnL moyen": float(g["book"].mean()), "Sharpe": sharpe(g["book"])})
    clus = pd.DataFrame(clus)
    clus.to_csv(OUT / "08_clustering.csv", index=False)
    npos = net_positions()
    npos["levels"].to_csv(OUT / "08_net_position_levels.csv", index=False)
    summary = {"freq_days_4plus_losing": freq4, "max_abs": npos["max_abs"], "max_usd_per_point": npos["max_usd_per_point"],
               "max_lots": npos["max_lots"], "sessions": npos["sessions"], "sessions_all_three_open": npos["sessions_all_three_open"]}
    write_json(OUT / "08_summary.json", summary)
    render(rc, rt, ww, wm, clus, npos, summary)
    return summary


def render(rc, rt, ww, wm, clus, npos, summary) -> None:
    def rc_t():
        lines = []
        for w, t in rc.items():
            lines += [f"**{w}**", "", "| Sleeve | PnL | Vol. quotidienne | Pire jour | maxDD | Contribution au risque |",
                      "|---|---:|---:|---:|---:|---:|"]
            for r in t.itertuples(index=False):
                lines.append(f"| {r.sleeve} | {fr(r.pnl)} | {fr(r.vol)} | {fr(r.worst_day)} | {fr(r.max_dd)} | {fr(100 * r.risk_contribution, 1)} % |")
            lines.append("")
        return lines

    lev = npos["levels"]
    lt = ["| Instrument | Position nette | Part du temps (09:30–16:08) |", "|---|---:|---:|"]
    for r in lev.itertuples(index=False):
        if r.part_du_temps >= 0.0005:
            lt.append(f"| {r.instrument} | {r.position_nette:+d} | {fr(100 * r.part_du_temps, 1)} % |")

    def reg_t(serie):
        sub = rt[rt["série"] == serie]
        t = ["| Variable | Instrument | Régime | Séances | PnL | PnL moyen | Sharpe | Jours gagnants |", "|---|---|---|---:|---:|---:|---:|---:|"]
        for r in sub.itertuples(index=False):
            t.append(f"| {r.variable} | {r.instrument_régime} | {r.régime} | {r.séances} | {fr(r.pnl, 0)} | {fr(r.pnl_moyen)} | "
                     f"{fr(r.sharpe, 2)} | {fr(100 * r.jours_gagnants, 0)} % |")
        return t

    sl = ["| Sleeve | " + " | ".join(f"{LABELS[c]} bas / moyen / haut (Sharpe)" for c in LABELS) + " |",
          "|---|" + "---|" * len(LABELS)]
    for c in CODES:
        cells = []
        for col in LABELS:
            sub = rt[(rt["série"] == c) & (rt["variable"] == LABELS[col])].set_index("régime")
            cells.append(" / ".join(fr(sub.loc[k, "sharpe"], 2) if k in sub.index else "n/d" for k in ("bas", "moyen", "haut")))
        sl.append(f"| {c} | " + " | ".join(cells) + " |")

    def tab(df):
        cols = list(df.columns)
        t = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in df.itertuples(index=False):
            t.append("| " + " | ".join(fr(v) if isinstance(v, float) else str(v) for v in r) + " |")
        return t

    ct = ["| Variable | Régime | Séances | P(≥ 4 sleeves perdantes) | PnL book | PnL moyen | Sharpe |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in clus.itertuples(index=False):
        ct.append(f"| {r.variable} | {r.régime} | {r.séances} | {fr(100 * r[3], 1)} % | {fr(r[4], 0)} | {fr(r[5])} | {fr(r[6], 2)} |")

    lines = [
        "# Phase 8 — Concentration et régimes",
        "",
        "*Book à 8 sleeves (S01 = ATR 18), calendrier union, 2021-05-03 → 2026-03-19, sauf mention. Les sleeves sont "
        "mesurées sur tout leur historique, IS et validation réutilisée confondus : c'est une analyse **descriptive**. "
        "Barres quotidiennes RTH 09:30–16:00 ET, MNQ/MGC coupés au 2026-03-19, MBT au 2026-10-02. Les bornes des "
        "terciles sont fixées sur l'IS de chaque instrument. Script : [phase8_regimes.py](phase8_regimes.py). Tables : "
        "[08_regimes.csv](outputs/08_regimes.csv), [08_clustering.csv](outputs/08_clustering.csv), "
        "[08_worst_weeks.csv](outputs/08_worst_weeks.csv), [08_worst_months.csv](outputs/08_worst_months.csv), "
        "[08_net_position_levels.csv](outputs/08_net_position_levels.csv).*",
        "",
        "## Contribution au risque — book à 8 sleeves, calendrier corrigé",
        "",
        *rc_t(),
        "## Position nette par instrument (moteur de netting, sans limite de lots)",
        "",
        f"Sur {npos['sessions']} séances, la position nette maximale est de {npos['max_abs']['MNQ']} MNQ, "
        f"{npos['max_abs']['MGC']} MGC et {npos['max_abs']['MBT']} MBT, soit au plus "
        f"{fr(npos['max_usd_per_point']['MNQ'], 0)} USD par point MNQ, {fr(npos['max_usd_per_point']['MGC'], 0)} USD par point MGC "
        f"et {fr(npos['max_usd_per_point']['MBT'], 1)} USD par point MBT. Le pic simultané est de {fr(npos['max_lots'], 1)} lots "
        f"Topstep (1 MBT = 1 lot). Les trois instruments sont en position le même jour sur {npos['sessions_all_three_open']} séances.",
        "",
        *lt,
        "",
        "## Régimes — book à 8 sleeves",
        "",
        *reg_t("Book 8"),
        "",
        "## Régimes — Sharpe par sleeve et par tercile",
        "",
        *sl,
        "",
        "Détail par sleeve (PnL, séances, jours gagnants) dans [08_regimes.csv](outputs/08_regimes.csv).",
        "",
        "## Regroupement des pertes",
        "",
        f"Jours où au moins 4 des 8 sleeves perdent simultanément : **{fr(100 * summary['freq_days_4plus_losing'], 1)} %** des séances.",
        "",
        "**Les 10 pires semaines**",
        "",
        *tab(ww),
        "",
        "**Les 5 pires mois**",
        "",
        *tab(wm),
        "",
        "**Pertes simultanées par régime**",
        "",
        *ct,
        "",
        "## Le book perd-il de façon groupée en range ou en retour à la moyenne ?",
        "",
        "**À l'échelle de la journée, oui, et c'est le principal risque du book.** Ce risque est concentré sur les "
        "sleeves MNQ directionnelles.",
        "",
        "- Sur les journées de range MNQ (tercile bas de abs(C−O)/(H−L), ex post), le book à 8 sleeves perd 52 187 USD en "
        "426 séances (Sharpe −6,3, 26 % de jours gagnants). Sur les journées de tendance (tercile haut), il gagne "
        "118 188 USD (81 % de jours gagnants).",
        "- Les pertes de range viennent de S01 (−25 713 USD), de S02 (−51 612 USD) et de S04 en shadow (−40 840 USD). "
        "Ce sont des stratégies de breakout, qui paient leurs faux départs les jours sans suite.",
        "- Les sleeves MGC (S05, S06) et MBT (S07, S08) ne dépendent presque pas du type de journée MNQ. C'est leur apport "
        "de diversification.",
        "- Les journées de range ne font pas perdre 4 sleeves ou plus en même temps beaucoup plus souvent : 6,3 % contre "
        "5,2 % en moyenne et 2,2 % en tendance. La perte est groupée par instrument, pas sur tout le book.",
        "",
        "**À l'échelle du mois, non.** Le book fait mieux pendant les mois de faible efficiency ratio MNQ, donc plus "
        "hachés (ER20 centré bas : Sharpe 2,72, contre 0,54 en ER20 haut). Les journées de tendance intraday "
        "surviennent aussi pendant les mois sans tendance de fond. L'ER20 ex ante, connu la veille, ne trie pas le "
        "résultat (Sharpe 2,36 / 0,95 / 2,31) : il n'y a là aucun filtre exploitable sans nouvelle hypothèse.",
        "",
        "**Concentration.**",
        "",
        "- S02 porte 41 % du risque du book sur 2024-01 → 2026-03, et 44 % depuis 2021.",
        "- Elle est la pire sleeve de 5 des 10 pires semaines et de 3 des 5 pires mois.",
        "- Les pires périodes se regroupent sur janvier et mars 2025 (−6 604 USD à eux deux) et sur mars 2026.",
        "",
        "Ces résultats sont **descriptifs**. Le type de journée n'est connu qu'à la clôture et ne peut pas servir de "
        "filtre. Ils disent où le book est exposé, pas comment l'éviter.",
    ]
    (AUDIT / "08_regimes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    print(main())
