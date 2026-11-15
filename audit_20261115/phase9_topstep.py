"""Phase 9 — the reference book under the Topstep lot limit (aggregated execution, S01 = ATR 18).

Sessions: union calendar, IS 2021-05-03 -> 2023-12-29 (selection), validation 2024-01-02 -> 2026-03-19 (reported).
Tiers: fixed 2, 3 and 5 lots (20 / 30 / 50 tenths) and dynamic XFA tier read on the previous end-of-session balance
of a sequential XFA starting at 0 (< 1 500 USD: 2 lots, 1 500-2 000: 3 lots, > 2 000: 5 lots); 1 MBT = 1 lot,
micro = 0.1 lot.  Variants (sim_variants.py): i current, ii no late catch-up, iii IS-defined priority, iv no MBT
below 3 lots.  Gate: variant i == src/portfolio/aggregator.py on every IS session and cap.
Complement: Combine -> XFA -> payout lifecycle (src/analytics/topstep_lifecycle) on the frozen study's bootstrap
settings (8 000 paths, 126 traded sessions, blocks 3/5/10/20, seed 20261112).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import AUDIT, OUT, fr, max_dd, sharpe, write_json
from .netting import RS, b0_sizes, bars, specs, trades_ref, union_dates
from .sim_variants import VariantSpec, simulate_session_v
from src.analytics.topstep_lifecycle.simulator import (CombineRules, Fees, SessionTable, XfaRules, block_bootstrap_paths,
                                                       simulate_lifecycle, summarize)
from src.portfolio.aggregator import RiskLimits, simulate_session

WIN = {"IS": ("2021-05-03", "2023-12-29"), "validation": ("2024-01-02", "2026-03-19")}
CAPS = (20, 30, 50, None)
VARIANTS = ("i", "ii", "iii", "iv")
SLEEVES = RS.SLEEVES  # S01 S02 S03 S05 S06 S07 S08 S09


def sleeve_params(t: pd.DataFrame) -> dict:
    """Last entry minute (observed on IS, +0) and IS risk-adjusted priority; all from the IS window only."""
    is_t = t[(t.session_date >= RS.d(WIN["IS"][0])) & (t.session_date <= RS.d(WIN["IS"][1]))]
    last = np.array([int(is_t[is_t.s_idx == s]["m_entry"].max()) for s in range(len(SLEEVES))])
    daily = is_t.groupby(["session_date", "s_idx"])["net_native"].sum().unstack(fill_value=0.0)
    dates = union_dates(*WIN["IS"])
    daily = daily.reindex(pd.Index(dates)).fillna(0.0).reindex(columns=range(len(SLEEVES)), fill_value=0.0)
    book = daily.sum(axis=1)
    rc = np.array([np.cov(daily[s], book, ddof=1)[0, 1] / book.var(ddof=1) for s in range(len(SLEEVES))])
    pnl = daily.sum().to_numpy()
    units_native = np.array([int(round(is_t[is_t.s_idx == s]["qty_native"].mean() * (10 if SLEEVES[s] in ("S07", "S08", "S09") else 1)))
                             for s in range(len(SLEEVES))])
    units_native = np.maximum(units_native, 1)
    score = np.where(rc > 0, pnl / (rc * units_native), np.inf)  # PnL per unit of risk contribution and per lot-tenth
    priority = np.argsort(np.argsort(-score))  # 0 = highest priority
    reserve = np.array([4 if s == "S01" else (10 if s in ("S07", "S08", "S09") else 1) for s in SLEEVES])
    return {"last_entry_minute": last, "priority": priority, "reserve_units": reserve, "rc_is": rc, "pnl_is": pnl,
            "score": score}


def run_all_sessions(dates, cube, events, variant: VariantSpec, cap):
    lim = RiskLimits(lot_cap_units=cap)
    res, logs = [], []
    for x in dates:
        r, lg = simulate_session_v(cube.session(x), events.get(x, RS.EMPTY_EVENTS), RS.SLEEVE_INSTRUMENT, specs(), lim, variant)
        res.append(r)
        logs.append(lg)
    return res, logs


def gate(dates, cube, events) -> dict:
    out = {}
    for cap in CAPS:
        lim = RiskLimits(lot_cap_units=cap)
        ref = np.array([simulate_session(cube.session(x), events.get(x, RS.EMPTY_EVENTS), RS.SLEEVE_INSTRUMENT, specs(), lim).final
                        for x in dates])
        mine, _ = run_all_sessions(dates, cube, events, VariantSpec("i"), cap)
        gap = float(np.abs(ref - np.array([r.final for r in mine])).max())
        out[str(cap)] = gap
    out["passed"] = all(v <= 1e-9 for k, v in out.items() if k != "passed")
    return out


def dynamic(results_by_cap: dict) -> tuple[np.ndarray, np.ndarray]:
    """Sequential XFA balance from 0: cap of session k from the balance after session k-1."""
    n = len(results_by_cap[20])
    pnl, caps = np.zeros(n), np.zeros(n, dtype=int)
    bal = 0.0
    for k in range(n):
        cap = 20 if bal < 1500 else (30 if bal <= 2000 else 50)
        caps[k] = cap
        pnl[k] = results_by_cap[cap][k].final
        bal += pnl[k]
    return pnl, caps


def lost_by_sleeve(res_cap, logs, res_free) -> dict:
    """Per-instrument PnL lost to the cap, split across that instrument's sleeves pro rata to refused contracts."""
    lost = {s: 0.0 for s in SLEEVES}
    refused = {s: 0 for s in SLEEVES}
    for rc, lg, rf in zip(res_cap, logs, res_free):
        diff = rf.instrument_final - rc.instrument_final
        for i in range(3):
            owners = [s for s in range(len(SLEEVES)) if RS.SLEEVE_INSTRUMENT[s] == i]
            cnt = {s: lg.refused_by_sleeve.get(s, 0) for s in owners}
            tot = sum(cnt.values())
            for s in owners:
                refused[SLEEVES[s]] += cnt[s]
                if tot:
                    lost[SLEEVES[s]] += diff[i] * cnt[s] / tot
                elif abs(diff[i]) > 1e-9:
                    lost[SLEEVES[s]] += diff[i] / len(owners)
    return {"lost": lost, "refused": refused}


def stats(pnl: np.ndarray) -> dict:
    s = pd.Series(pnl)
    return {"pnl": float(s.sum()), "sharpe": sharpe(s), "max_dd": max_dd(s), "worst_day": float(s.min())}


def main() -> dict:
    t = trades_ref()
    all_dates = union_dates(WIN["IS"][0], WIN["validation"][1])
    cube = RS.BarCube(bars("2026-03-19"), all_dates)
    events = RS.build_events(t, b0_sizes(t))
    is_dates = [x for x in all_dates if x <= RS.d(WIN["IS"][1])]
    g = gate(is_dates, cube, events)
    if not g["passed"]:
        write_json(OUT / "09_gate.json", g)
        raise SystemExit(f"variant simulator does not reproduce the aggregator: {g}")
    sp = sleeve_params(t)
    rows, lost_rows, life_rows = [], [], []
    tables_by_variant: dict = {}
    for vn in VARIANTS:
        v = VariantSpec(vn, last_entry_minute=sp["last_entry_minute"], priority=sp["priority"], reserve_units=sp["reserve_units"])
        res, logs = {}, {}
        for cap in CAPS:
            res[cap], logs[cap] = run_all_sessions(all_dates, cube, events, v, cap)
        for wname, (a, b) in WIN.items():
            idx = [k for k, x in enumerate(all_dates) if RS.d(a) <= x <= RS.d(b)]
            sub = {cap: [res[cap][k] for k in idx] for cap in CAPS}
            sublog = {cap: [logs[cap][k] for k in idx] for cap in CAPS}
            free = np.array([r.final for r in sub[None]])
            for mode in ("2 lots", "3 lots", "5 lots", "XFA dynamique"):
                if mode == "XFA dynamique":
                    pnl, caps = dynamic(sub)
                    red = sum(1 for k, c in enumerate(caps) if sublog[c][k].sessions_reduced)
                    lb_res = [sub[c][k] for k, c in enumerate(caps)]
                    lb_log = [sublog[c][k] for k, c in enumerate(caps)]
                    share = {c: float(np.mean(caps == c)) for c in (20, 30, 50)}
                else:
                    cap = {"2 lots": 20, "3 lots": 30, "5 lots": 50}[mode]
                    pnl = np.array([r.final for r in sub[cap]])
                    red = sum(1 for lg in sublog[cap] if lg.sessions_reduced)
                    lb_res, lb_log, share = sub[cap], sublog[cap], None
                lb = lost_by_sleeve(lb_res, lb_log, sub[None])
                rows.append({"variante": vn, "fenêtre": wname, "palier": mode, "séances": len(idx), **stats(pnl),
                             "séances réduites": red, "PnL perdu vs sans limite": float(free.sum() - pnl.sum()),
                             "part 2/3/5 lots": share})
                for s in SLEEVES:
                    lost_rows.append({"variante": vn, "fenêtre": wname, "palier": mode, "sleeve": s,
                                      "contrats refusés": lb["refused"][s], "PnL perdu (attribué)": lb["lost"][s]})
            if True:
                tables = {cap: SessionTable.from_results(sub[cap]) for cap in (20, 30, 50)}
                paths = block_bootstrap_paths(len(idx), 8000, 126, (3, 5, 10, 20), 20261112)
                out = simulate_lifecycle(paths, tables, CombineRules(), XfaRules(), Fees(), profit_lock=False, lock_margin=50.0)
                life_rows.append({"variante": vn, "fenêtre": wname, **summarize(out)})
        print(vn, "done", flush=True)
    tab = pd.DataFrame(rows)
    lost = pd.DataFrame(lost_rows)
    life = pd.DataFrame(life_rows)
    tab.to_csv(OUT / "09_topstep.csv", index=False)
    lost.to_csv(OUT / "09_lost_by_sleeve.csv", index=False)
    life.to_csv(OUT / "09_lifecycle.csv", index=False)
    write_json(OUT / "09_gate.json", g)
    write_json(OUT / "09_params.json", {"sleeves": SLEEVES, **{k: np.asarray(v).tolist() for k, v in sp.items()}})
    render(tab, lost, life, g, sp)
    return {"gate": g}


def render(tab, lost, life, g, sp) -> None:
    def minute_str(m):
        h, mm = divmod(8 * 60 + 30 + int(m), 60)
        return f"{h:02d}:{mm:02d}"

    head = tab[(tab["variante"] == "i") & (tab["palier"] == "XFA dynamique")]
    free = tab[(tab["variante"] == "i") & (tab["palier"] == "5 lots")]
    h_t = ["| Fenêtre | Palier | PnL | Sharpe | maxDD | Pire jour | Séances réduites | PnL perdu vs sans limite |",
           "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in pd.concat([head, free]).itertuples(index=False):
        h_t.append(f"| {r.fenêtre} | {r.palier} | {fr(r.pnl, 0)} | {fr(r.sharpe, 3)} | {fr(r.max_dd, 0)} | {fr(r.worst_day, 0)} | "
                   f"{r[8]} / {r.séances} | {fr(r[9], 0)} |")
    v_t = ["| Variante | Fenêtre | Palier | PnL | Sharpe | maxDD | Pire jour | Séances réduites | PnL perdu |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in tab.itertuples(index=False):
        v_t.append(f"| {r.variante} | {r.fenêtre} | {r.palier} | {fr(r.pnl, 0)} | {fr(r.sharpe, 3)} | {fr(r.max_dd, 0)} | "
                   f"{fr(r.worst_day, 0)} | {r[8]} | {fr(r[9], 0)} |")
    lo = lost[(lost["variante"] == "i") & (lost["palier"] == "XFA dynamique")]
    l_t = ["| Fenêtre | " + " | ".join(SLEEVES) + " |", "|---|" + "---:|" * len(SLEEVES)]
    for w, gdf in lo.groupby("fenêtre", sort=False):
        gdf = gdf.set_index("sleeve")
        l_t.append(f"| {w} — PnL perdu | " + " | ".join(fr(gdf.loc[s, 'PnL perdu (attribué)'], 0) for s in SLEEVES) + " |")
        l_t.append(f"| {w} — contrats refusés | " + " | ".join(str(int(gdf.loc[s, 'contrats refusés'])) for s in SLEEVES) + " |")
    lf = ["| Variante | Fenêtre | Cash net espéré | Cash net médian | P(payout) | Séances médianes au 1er payout | P(XFA perdu) | P(PASS) |",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in life.itertuples(index=False):
        lf.append(f"| {r.variante} | {r.fenêtre} | {fr(r.expected_net_cash, 0)} | {fr(r.median_net_cash, 0)} | "
                  f"{fr(100 * r.p_any_payout, 1)} % | {fr(r.median_sessions_to_first_payout, 0)} | {fr(100 * r.p_xfa_lost, 1)} % | "
                  f"{fr(100 * r.p_pass, 1)} % |")
    pr = ["| Sleeve | Dernière entrée observée (IS) | Contribution au risque IS | PnL IS | Priorité (iii), 0 = haute |",
          "|---|---|---:|---:|---:|"]
    for k, s in enumerate(SLEEVES):
        pr.append(f"| {s} | {minute_str(sp['last_entry_minute'][k])} | {fr(100 * sp['rc_is'][k], 1)} % | {fr(sp['pnl_is'][k], 0)} | {int(sp['priority'][k])} |")
    lines = [
        "# Phase 9 — Book sous contrainte Topstep",
        "",
        "*Exécution agrégée par instrument, book à 8 sleeves, S01 = ATR 18, calendrier union. IS 2021-05-03 → 2023-12-29 "
        "(choix), validation 2024-01-02 → 2026-03-19 (reportée). 1 MBT = 1 lot, un micro = 0,1 lot. Une entrée qui dépasse "
        "la limite est réduite ; une position ouverte n'est jamais coupée. Script : [phase9_topstep.py](phase9_topstep.py), "
        "simulateur [sim_variants.py](sim_variants.py). Tables : [09_topstep.csv](outputs/09_topstep.csv), "
        "[09_lost_by_sleeve.csv](outputs/09_lost_by_sleeve.csv), [09_lifecycle.csv](outputs/09_lifecycle.csv).*",
        "",
        "**Gate** : avec la variante (i), le simulateur reproduit `src/portfolio/aggregator.py` sur les 4 configurations de "
        f"limite (2, 3, 5 lots et sans limite), séance par séance, sur l'IS. Écart max : {max(v for k, v in g.items() if k != 'passed'):.1e} USD.",
        "",
        "## Chiffres de tête — règle actuelle (i), palier XFA dynamique",
        "",
        "Le palier dynamique suit le solde d'un XFA qui part de 0 : 2 lots sous 1 500 USD, 3 lots jusqu'à 2 000, puis 5. "
        "La ligne « 5 lots » correspond au Combine 50K. Elle ne contraint presque jamais le book, dont le pic est de 3,3 lots.",
        "",
        *h_t,
        "",
        "## PnL perdu à cause de la limite, par sleeve (variante i, palier dynamique)",
        "",
        "Pour chaque instrument, la perte est exacte : PnL sans limite moins PnL avec limite. Sa répartition entre les "
        "sleeves de l'instrument se fait au prorata des contrats refusés à l'entrée.",
        "",
        *l_t,
        "",
        "## Variantes d'exécution (signaux inchangés)",
        "",
        "- **(i)** Règle actuelle : premier arrivé, premier servi, avec rattrapage.",
        "- **(ii)** Pas de rattrapage après la dernière heure d'entrée de la ou des sleeves sous-servies de l'instrument.",
        "- **(iii)** Priorité fixée sur l'IS : une entrée ne peut pas consommer la place réservée aux sleeves à plat de "
        "priorité plus haute. Le score est le PnL IS par unité de contribution au risque et par dixième de lot.",
        "- **(iv)** Aucune sleeve MBT tant que le palier est inférieur à 3 lots.",
        "",
        *pr,
        "",
        *v_t,
        "",
        "## Cycle Combine → XFA → payout (complément)",
        "",
        "Les réglages sont ceux de l'étude d'agrégation : 8 000 chemins de 126 séances tradées, blocs de 3, 5, 10 et 20, "
        "seed 20261112, frais standard, palier XFA relu chaque séance. Les chemins sont tirés dans chaque fenêtre séparément.",
        "",
        *lf,
        "",
        "## Lecture",
        "",
        "1. **Chiffre de tête sous contrainte** (règle actuelle, palier XFA dynamique) : 29 142 USD sur l'IS 2021-05 → 2023-12 "
        "(Sharpe 2,32, maxDD −2 567 USD) et 29 270 USD sur la validation 2024-01 → 2026-03 (Sharpe 1,65, maxDD −5 863 USD, "
        "pire jour −1 656 USD). Sans limite, la validation fait 30 886 USD : la limite coûte 1 616 USD sur la validation, "
        "et en fait gagner 574 sur l'IS, où les entrées refusées étaient perdantes en moyenne.",
        "2. **La limite ne mord vraiment qu'au palier de 2 lots.** Un XFA qui démarre à 0 passe vite au-dessus de 2 000 USD "
        "dans l'historique, d'où seulement 21 (IS) et 41 (validation) séances réduites en dynamique. À 2 lots fixes, 196 et "
        "279 séances sont réduites. MBT, à 1 lot par contrat, est la cause : S08 a le plus de contrats refusés.",
        "3. **Choix sur l'IS, avec pour critère le cash net espéré du cycle complet (objectif payout)** :",
        "   - (ii) fait +7 USD par rapport à (i), ce qui est indiscernable du bruit ;",
        "   - (iii) fait −11 USD ;",
        "   - (iv) fait −80 USD, et perd 3 152 USD de PnL historique à 2 lots.",
        "",
        "   **Aucune variante ne bat nettement la règle actuelle : (i) reste la règle.** (iv) est rejetée. Sur la "
        "validation, simplement reportée, (iii) fait un peu mieux (1 030 contre 972 USD) ; ce n'est pas un motif de "
        "changement, puisque l'IS ne la favorisait pas.",
        "4. **Constat d'exécution.** S02 a des entrées observées jusqu'à 16:02 sur l'IS, soit 6 minutes avant le flatten "
        "de 16:08. Son vote peut se retourner après la dernière entrée des cellules, quand une cellule sort.",
        "5. **Le cycle reste défavorable à la médiane.** Le cash net médian est négatif (−50 USD sur l'IS) et l'XFA est "
        "perdu dans 65 à 74 % des chemins, quelle que soit la variante. Le cash espéré positif vient d'une minorité de "
        "chemins à plusieurs payouts.",
        "",
        "La répartition du PnL perdu entre sleeves d'un même instrument est une approximation, au prorata des contrats "
        "refusés. Ses signes peuvent être contre-intuitifs quand les entrées refusées étaient perdantes.",
    ]
    (AUDIT / "09_topstep.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def rerender() -> None:
    import json
    tab = pd.read_csv(OUT / "09_topstep.csv")
    lost = pd.read_csv(OUT / "09_lost_by_sleeve.csv")
    life = pd.read_csv(OUT / "09_lifecycle.csv")
    g = json.loads((OUT / "09_gate.json").read_text(encoding="utf-8"))
    sp = json.loads((OUT / "09_params.json").read_text(encoding="utf-8"))
    render(tab, lost, life, g, {k: np.asarray(v) for k, v in sp.items() if k != "sleeves"})


if __name__ == "__main__":
    import sys
    if "--render" in sys.argv:
        rerender()
    else:
        print(main())
