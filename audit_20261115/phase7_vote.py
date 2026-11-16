"""Phase 7 — diagnostic of the thresholded vote (EXPLORATOIRE).  Nothing here changes the reference.

A vectorised re-implementation of scripts/rebuild_signal_sheet_02_cluster.py::aggregate_vote with an injectable
target function q = f(score).  Gate: with the reference target (+1 if score >= 0.5, -1 if score <= -0.5, else 0),
next-minute-open execution and nominal costs, it must reproduce every sleeve's frozen daily PnL (|gap| <= 1e-6 USD)
before any variant is computed.

Compared on IS / reused validation / total, S02-S09:
  (a) the reference vote;
  (b) the mean of the cells' own daily net PnL (theoretical, not executable);
  (c) q = round_half_away(K x score), K in {2, 3}, MNQ and MGC sleeves only, costs on every quantity change.
S09 legacy OOS: bridge from (a) to (b) — costs, one-minute delay, thresholding (never-triggered days, late entries,
early exits, exposure), residual (cell-level execution and costs).
"""

from __future__ import annotations

import json
import math
import sys
from typing import Callable

import numpy as np
import pandas as pd

from .common import AUDIT, CACHE, CODES, INSTRUMENT, OUT, ROOT, WINDOWS, fr, max_dd, sharpe, write_json

SLEEVES = CODES[1:]  # S02..S09
KS = (2, 3)
SCOPES = ("is", "val", "full")


# ----------------------------------------------------------------------------- loading
def load(code: str):
    """prepared data, cell positions, execution parameters, cell count, threshold."""
    cache = CACHE / f"vote_input_{code}.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    key = code.lower()
    if code == "S02":
        import rebuild_signal_sheet_02_cluster as r
        specs, _, _ = r.load_frozen_specs(r.FROZEN_DIR, r.FROZEN_MANIFEST)
        spec = next(s for s in specs if s["mechanism_id"] == "volatility_breakout")
        prepared = r.prepare_data(r.DATA_PATH, start_date=r.IS_START, end_date=r.OOS_END)
    else:
        import freeze_signal_sheet_cluster_vote as f
        prepared, spec, _ = f.prepare_campaign(key, f.CAMPAIGNS[key])
    pos = pd.read_csv(ROOT / "research" / f"signal_sheet_{code[1:]}_cluster" / "cell_positions.csv")
    ex = spec["canonical_parameters"]["execution"]
    minute = prepared.minute[["open", "close", "session_date"]].copy()
    data = {"minute": minute, "sessions": prepared.sessions[["session_date", "hard_flatten"]].copy(), "positions": pos,
            "tick": float(ex["tick_size"]), "pv": float(ex["point_value_usd"]), "comm": float(ex["commission_per_side_usd"]),
            "slip": int(ex["slippage_ticks_per_side"]), "n": int(pos["cell_id"].nunique())}
    pd.to_pickle(data, cache)
    return data


def ref_target(score: np.ndarray) -> np.ndarray:
    return np.where(score >= 0.5 - 1e-12, 1, np.where(score <= -0.5 + 1e-12, -1, 0))


def k_target(K: int) -> Callable[[np.ndarray], np.ndarray]:
    return lambda s: (np.sign(s) * np.floor(np.abs(K * s) + 0.5 + 1e-12)).astype(int)


def linear_target(score: np.ndarray) -> np.ndarray:
    return score.astype(float)


# ----------------------------------------------------------------------------- simulator
def simulate(d: dict, target: Callable, delay: bool = True, costs: bool = True, keep: bool = False) -> pd.DataFrame:
    """Daily gross / costs / net / contracts of the aggregate driven by f(score)."""
    minute, sessions, pos = d["minute"], d["sessions"], d["positions"]
    tick, pv, n = d["tick"], d["pv"], d["n"]
    slip_px = d["slip"] * tick if costs else 0.0
    comm = d["comm"] if costs else 0.0
    ns = lambda s: pd.to_datetime(s, utc=True).astype("int64").to_numpy()
    ev = pd.DataFrame({
        "day": np.r_[pos["session_date"].to_numpy(), pos["session_date"].to_numpy()],
        "t": np.r_[ns(pos["entry_time"]), ns(pos["exit_time"])],
        "delta": np.r_[pos["position"].to_numpy(float), -pos["position"].to_numpy(float)],
    })
    ev_by_day = {k: g for k, g in ev.groupby("day")}
    hf = dict(zip(sessions["session_date"].astype(str), pd.to_datetime(sessions["hard_flatten"], utc=True).astype("int64")))
    rows, attrib = [], []
    for day, frame in minute.groupby("session_date", sort=True):
        dkey = str(day)
        ts = frame.index.tz_convert("UTC").asi8
        op, cl = frame["open"].to_numpy(float), frame["close"].to_numpy(float)
        T = len(ts)
        h = hf[dkey]
        exec_q = np.full(T, np.nan)  # quantity to hold from this bar's execution price
        exec_px = op.copy()
        g = ev_by_day.get(dkey)
        if g is not None and len(g):
            gg = g.groupby("t")["delta"].sum().sort_index()
            times = gg.index.to_numpy()
            score = np.cumsum(gg.to_numpy()) / n
            q = target(score)
            for e, qq in zip(times, q):
                if e >= h:
                    continue
                if delay:
                    j = int(np.searchsorted(ts, e, side="right"))
                    if j < T and ts[j] <= h:
                        exec_q[j] = qq
                else:
                    j = int(np.searchsorted(ts, e, side="left"))
                    if j < T and ts[j] == e:
                        exec_q[j] = qq
                        exec_px[j] = cl[j]  # immediate fill at the signal bar's close (theoretical)
        jf = int(np.searchsorted(ts, h, side="left"))
        held = np.zeros(T)
        cur = 0.0
        cash = gross = cost = 0.0
        contracts = 0.0
        fills = []
        for k in range(T):
            if k == jf:
                new, px = 0.0, op[k]
            elif k < jf and not np.isnan(exec_q[k]):
                new, px = float(exec_q[k]), exec_px[k]
            else:
                new, px = cur, None
            if new != cur:
                dq = new - cur
                gross -= dq * px * pv
                cost += abs(dq) * (comm + slip_px * pv)
                contracts += abs(dq)
                fills.append((k, dq, px))
                cur = new
            held[k] = cur
            if k == jf:
                break
        if cur != 0.0:  # no bar at or after hard flatten: close at the last bar's close
            dq = -cur
            gross -= dq * cl[-1] * pv
            cost += abs(dq) * (comm + slip_px * pv)
            contracts += abs(dq)
            cur = 0.0
        net = gross - cost
        rows.append({"session_date": pd.Timestamp(day), "gross": gross, "costs": cost, "net": net, "contracts": contracts,
                     "active": bool(np.any(held != 0))})
        if keep:
            attrib.append((pd.Timestamp(day), ts, op, cl, held, fills, jf))
    out = pd.DataFrame(rows).set_index("session_date")
    if keep:
        out.attrs["paths"] = attrib
    return out


def gate(code: str, d: dict) -> dict:
    sim = simulate(d, ref_target)
    fro = pd.read_csv(ROOT / "research" / f"signal_sheet_{code[1:]}_cluster" / "daily_results.csv",
                      parse_dates=["session_date"]).set_index("session_date")["daily_pnl_usd"]
    gap = (sim["net"].reindex(fro.index).fillna(0.0) - fro).abs()
    return {"max_abs_gap": float(gap.max()), "days": int(len(fro)), "passed": bool(gap.max() <= 1e-6), "sim": sim}


# ----------------------------------------------------------------------------- cells mean
def cells_mean(code: str, d: dict) -> pd.Series:
    if code == "S02":
        from .phase7_s02_cells import main as s02
        cells = s02()
        ids = sorted(d["positions"]["cell_id"].unique())
        if set(cells.columns) != set(ids):
            raise RuntimeError("S02 cell ids differ from the cluster")
        return cells[ids].mean(axis=1)
    key = code.lower()
    grid = pd.read_csv(ROOT / "research" / "signal_sheet_parameter_refinement" / key / "grid_metrics.csv").drop_duplicates("cell_id")
    import freeze_signal_sheet_cluster_vote as f
    region = f.LOCK_NEIGHBOURHOOD[key]
    names = list(region)
    mask = np.ones(len(grid), bool)
    for nme in names:
        mask &= grid[nme].astype(float).isin([float(v) for v in region[nme]]).to_numpy()
    ids = grid.loc[mask, "cell_id"].tolist()
    if len(ids) != d["n"]:
        raise RuntimeError(f"{code}: {len(ids)} grid cells matched for {d['n']} cluster cells")
    cdp = pd.read_parquet(ROOT / "research" / "signal_sheet_parameter_refinement" / key / "cell_daily_pnl.parquet")
    cdp.index = pd.to_datetime(cdp.index)
    return cdp[ids].mean(axis=1)


def scope_of(code: str) -> dict:
    return WINDOWS["MBT"] if INSTRUMENT[code] == "MBT" else WINDOWS["MNQ_MGC"]


def metrics(s: pd.Series, a: str, b: str) -> dict:
    x = s[(s.index >= a) & (s.index <= b)]
    return {"pnl": float(x.sum()), "sharpe": sharpe(x), "max_dd": max_dd(x)}


# ----------------------------------------------------------------------------- S09 bridge
def minute_attribution(d: dict, a: str, b: str) -> dict:
    """Linear (q = score) no-cost no-delay exposure vs the vote, split by the vote's state within the day."""
    lin = simulate(d, linear_target, delay=False, costs=False, keep=True)
    vote = simulate(d, ref_target, delay=False, costs=False, keep=True)
    parts = {"jamais_declenche": 0.0, "avant_entree": 0.0, "apres_sortie_anticipee": 0.0, "pendant_vote_exposition": 0.0}
    lp = {p[0]: p for p in lin.attrs["paths"]}
    for day, ts, op, cl, held_v, fills_v, jf in vote.attrs["paths"]:
        if not (pd.Timestamp(a) <= day <= pd.Timestamp(b)):
            continue
        _, _, _, _, held_l, fills_l, _ = lp[day]
        pv = d["pv"]
        # bar-by-bar PnL of each exposure: position held after bar k's fill, marked to bar k's close,
        # plus the fill-to-close move on the bar where it changed.
        def bar_pnl(held, fills):
            pnl = np.zeros(len(held))
            prev_close = None
            prev_q = 0.0
            fill_at = {k: (dq, px) for k, dq, px in fills}
            for k in range(len(held)):
                if k > jf:
                    break
                if prev_close is not None:
                    pnl[k] += prev_q * (op[k] - prev_close) * pv
                if k in fill_at:
                    dq, px = fill_at[k]
                    pnl[k] += prev_q * (px - op[k]) * pv + held[k] * (cl[k] - px) * pv
                else:
                    pnl[k] += held[k] * (cl[k] - op[k]) * pv
                prev_close, prev_q = cl[k], held[k]
            return pnl
        pl, pv_ = bar_pnl(held_l, fills_l), bar_pnl(held_v, fills_v)
        active = held_v != 0
        if not active.any():
            parts["jamais_declenche"] += float(pl.sum())
            continue
        first, last = int(np.argmax(active)), int(len(active) - 1 - np.argmax(active[::-1]))
        k_idx = np.arange(len(active))
        exit_bar = min(last + 1, jf if jf < len(active) else len(active) - 1)
        before = k_idx < first
        after = (k_idx > exit_bar) & (k_idx <= jf)
        mid = (k_idx >= first) & (k_idx <= exit_bar)
        parts["avant_entree"] += float(pl[before].sum())
        parts["apres_sortie_anticipee"] += float(pl[after].sum())
        parts["pendant_vote_exposition"] += float((pl[mid] - pv_[mid]).sum())
    return {"lin_total": float(lin["net"][(lin.index >= a) & (lin.index <= b)].sum()), **parts,
            "vote_nodelay_gross": float(vote["net"][(vote.index >= a) & (vote.index <= b)].sum())}


def s09_bridge(d: dict, cm: pd.Series, a_ref: pd.Series) -> dict:
    a, b = WINDOWS["MBT"]["val"]
    w = lambda s: float(s[(s.index >= a) & (s.index <= b)].sum())
    gross_delay = simulate(d, ref_target, delay=True, costs=False)["net"]
    att = minute_attribution(d, a, b)
    A, Bm = w(a_ref), w(cm)
    G = w(gross_delay)
    out = {"window": [a, b], "a_vote_net": A, "b_cells_mean": Bm, "gap_b_minus_a": Bm - A,
           "1_couts": G - A, "2_decalage_une_minute": att["vote_nodelay_gross"] - G,
           "3_seuil_jours_jamais_declenches": att["jamais_declenche"], "3_seuil_avant_entree": att["avant_entree"],
           "3_seuil_sorties_anticipees": att["apres_sortie_anticipee"],
           "3_seuil_exposition_pendant_le_vote": att["pendant_vote_exposition"],
           "lineaire_sans_cout_sans_decalage": att["lin_total"]}
    out["4_residu_execution_et_couts_des_cellules"] = Bm - att["lin_total"]
    parts3 = sum(out[k] for k in out if k.startswith("3_"))
    out["controle_additivite_3"] = float(att["lin_total"] - att["vote_nodelay_gross"] - parts3)
    # days where the majority of cells win but the vote stays flat
    return out


def majority_flat_days(code: str, d: dict, a_sim: pd.DataFrame, a: str, b: str) -> dict:
    if code == "S02":
        from .phase7_s02_cells import main as s02
        cells = s02()
    else:
        key = code.lower()
        import freeze_signal_sheet_cluster_vote as f
        grid = pd.read_csv(ROOT / "research" / "signal_sheet_parameter_refinement" / key / "grid_metrics.csv").drop_duplicates("cell_id")
        region = f.LOCK_NEIGHBOURHOOD[key]
        mask = np.ones(len(grid), bool)
        for nme in region:
            mask &= grid[nme].astype(float).isin([float(v) for v in region[nme]]).to_numpy()
        cells = pd.read_parquet(ROOT / "research" / "signal_sheet_parameter_refinement" / key / "cell_daily_pnl.parquet")
        cells.index = pd.to_datetime(cells.index)
        cells = cells[grid.loc[mask, "cell_id"].tolist()]
    cells = cells[(cells.index >= a) & (cells.index <= b)]
    win_share = (cells > 0).sum(axis=1) / cells.shape[1]
    flat = ~a_sim["active"].reindex(cells.index).fillna(False).astype(bool)
    sel = (win_share > 0.5) & flat
    return {"days": int(sel.sum()), "cells_mean_pnl_on_those_days": float(cells.mean(axis=1)[sel].sum()),
            "days_vote_flat_total": int(flat.sum())}


# ----------------------------------------------------------------------------- main
def main() -> dict:
    rows, gates, bridge = [], {}, {}
    for code in SLEEVES:
        d = load(code)
        g = gate(code, d)
        gates[code] = {k: v for k, v in g.items() if k != "sim"}
        if not g["passed"]:
            write_json(OUT / "07_gates.json", gates)
            raise SystemExit(f"{code}: vote re-implementation does not reproduce the frozen daily PnL ({g['max_abs_gap']})")
        a_net = g["sim"]["net"]
        cm = cells_mean(code, d)
        variants = {"(a) vote seuillé": a_net, "(b) moyenne des cellules": cm}
        contracts = {"(a) vote seuillé": g["sim"]["contracts"]}
        if INSTRUMENT[code] in ("MNQ", "MGC"):
            for K in KS:
                s = simulate(d, k_target(K))
                variants[f"(c) K={K}"] = s["net"]
                contracts[f"(c) K={K}"] = s["contracts"]
        w = scope_of(code)
        for name, s in variants.items():
            for sc in SCOPES:
                m = metrics(s, *w[sc])
                c = contracts.get(name)
                rows.append({"sleeve": code, "variante": name, "fenêtre": sc, **m,
                             "contrats": float(c[(c.index >= w[sc][0]) & (c.index <= w[sc][1])].sum()) if c is not None else np.nan})
        if code == "S09":
            bridge = s09_bridge(d, cm, a_net)
            bridge["majorite_gagnante_vote_flat"] = majority_flat_days(code, d, g["sim"], *WINDOWS["MBT"]["val"])
        print(code, "ok", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "07_vote.csv", index=False)
    write_json(OUT / "07_gates.json", gates)
    write_json(OUT / "07_s09_bridge.json", bridge)
    render(res, gates, bridge)
    return {"gates": gates}


def render(res: pd.DataFrame, gates: dict, br: dict) -> None:
    sc_fr = {"is": "IS", "val": "validation", "full": "total"}
    t = ["| Sleeve | Variante | PnL IS | Sharpe IS | PnL validation | Sharpe validation | PnL total | Sharpe total | maxDD IS | Contrats IS |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (code, var), g in res.groupby(["sleeve", "variante"], sort=False):
        g = g.set_index("fenêtre")
        t.append(f"| {code} | {var} | {fr(g.loc['is', 'pnl'], 0)} | {fr(g.loc['is', 'sharpe'], 3)} | {fr(g.loc['val', 'pnl'], 0)} | "
                 f"{fr(g.loc['val', 'sharpe'], 3)} | {fr(g.loc['full', 'pnl'], 0)} | {fr(g.loc['full', 'sharpe'], 3)} | "
                 f"{fr(g.loc['is', 'max_dd'], 0)} | {fr(g.loc['is', 'contrats'], 0) if not math.isnan(g.loc['is', 'contrats']) else '—'} |")
    gt = ["| Sleeve | Séances | Écart max vs PnL gelé | Gate |", "|---|---:|---:|---|"]
    for k, v in gates.items():
        gt.append(f"| {k} | {v['days']} | {v['max_abs_gap']:.2e} USD | {'réussi' if v['passed'] else 'ÉCHOUÉ'} |")
    b = br
    m = b["majorite_gagnante_vote_flat"]
    bt = ["| Étape | USD |", "|---|---:|",
          f"| (a) vote seuillé, net | {fr(b['a_vote_net'])} |",
          f"| + 1. coûts du vote | {fr(b['1_couts'])} |",
          f"| + 2. décalage d'une minute (open suivant → close de la barre du signal) | {fr(b['2_decalage_une_minute'])} |",
          f"| + 3a. jours où le vote ne se déclenche jamais | {fr(b['3_seuil_jours_jamais_declenches'])} |",
          f"| + 3b. minutes avant l'entrée du vote | {fr(b['3_seuil_avant_entree'])} |",
          f"| + 3c. minutes après une sortie anticipée (score repassé sous ±0,50) | {fr(b['3_seuil_sorties_anticipees'])} |",
          f"| + 3d. exposition linéaire − exposition du vote, de la première entrée à la dernière sortie | {fr(b['3_seuil_exposition_pendant_le_vote'])} |",
          f"| = exposition linéaire (q = score), sans coût ni décalage | {fr(b['lineaire_sans_cout_sans_decalage'])} |",
          f"| + 4. résidu : exécution propre des cellules (stops intrabarre) et leurs coûts | {fr(b['4_residu_execution_et_couts_des_cellules'])} |",
          f"| = (b) moyenne des PnL des cellules | {fr(b['b_cells_mean'])} |"]
    lines = [
        "# Phase 7 — Diagnostic du vote seuillé (EXPLORATOIRE)",
        "",
        "*Aucune de ces variantes ne remplace la référence. Les conclusions s'appuient sur l'IS ; la validation et le "
        "total sont seulement reportés. Script : [phase7_vote.py](phase7_vote.py). Tables : [07_vote.csv](outputs/07_vote.csv), "
        "[07_s09_bridge.json](outputs/07_s09_bridge.json). PnL net en USD, coûts nominaux, Sharpe √252 sur le calendrier "
        "de la sleeve.*",
        "",
        "## Gate de reproduction",
        "",
        "Le vote est ré-implémenté de façon vectorisée, avec une fonction cible injectable. Avec la cible de référence, "
        "il doit redonner le PnL quotidien gelé de chaque sleeve avant toute variante.",
        "",
        *gt,
        "",
        "## Variantes",
        "",
        "- **(a)** Vote seuillé de la référence : +1 si score ≥ 0,50, −1 si score ≤ −0,50, 0 sinon. Exécution à l'open de la minute suivante.",
        "- **(b)** Moyenne des PnL quotidiens nets des cellules, chacune avec son propre stop et ses propres coûts. "
        "C'est une référence théorique : la répliquer demanderait des fractions de contrat.",
        "- **(c)** q = arrondi(K × score) contrats, l'arrondi s'éloignant de zéro, pour K = 2 et 3, sur MNQ et MGC "
        "seulement. Coûts comptés par contrat à chaque changement de quantité.",
        "",
        *t,
        "",
        "## S09 — de (a) à (b) sur le legacy OOS (2025-05-29 → 2026-10-02)",
        "",
        "Décomposition additive. Les étapes 3a à 3d comparent, minute par minute, une exposition linéaire (q = score, "
        "fractionnaire) au vote, tous deux sans coût ni décalage.",
        "",
        *bt,
        "",
        f"Jours où plus de la moitié des cellules gagnent alors que le vote reste flat toute la séance : **{m['days']}** "
        f"(sur {m['days_vote_flat_total']} jours de vote flat), pour {fr(m['cells_mean_pnl_on_those_days'])} USD de PnL "
        "moyen des cellules.",
        "",
        "## Lecture",
        "",
        "Conclusions fondées sur l'IS uniquement.",
        "",
        "1. **Le vote seuillé ne détruit pas de valeur sur l'IS.** Il fait mieux que la moyenne de ses cellules (b) pour "
        "7 sleeves sur 8, en PnL (S02 10 980 contre 8 893 USD, S08 2 535 contre 1 909, S09 4 989 contre 4 376) et le plus "
        "souvent en Sharpe. Seule S04 fait moins bien que ses cellules (4 692 contre 5 096 USD). L'idée d'un vote qui "
        "« laisse de l'argent sur la table » n'est pas confirmée sur la fenêtre de sélection.",
        "2. **La taille proportionnelle au score (c) n'améliore pas le rendement ajusté du risque sur l'IS.** Le PnL "
        "augmente avec K, mais le drawdown et les coûts augmentent davantage :",
        "   - S02 : Sharpe 0,86 → 0,80, avec un maxDD IS qui passe de −4 790 à −8 750 USD à K = 2 ;",
        "   - S03 et S04 : le Sharpe est divisé par 1,3 à 2,5 ;",
        "   - S06 est stable ;",
        "   - seule S05 progresse un peu (0,52 → 0,56–0,58), puis s'effondre sur la validation (0,39 → 0,10–0,15).",
        "",
        "   **Aucune variante (c) n'est retenue.** Pour un compte Topstep, elle multiplierait aussi les lots MNQ/MGC "
        "consommés.",
        "3. **S09 sur son legacy OOS.** L'écart de 993 USD entre (b) et (a) se décompose ainsi :",
        "   - les **coûts du vote** : 844 USD. Le vote change souvent de cible pour un gain par trade faible ;",
        "   - le **décalage d'une minute** : 88 USD ;",
        "   - le **seuil**, net : +505 USD. Le vote entre tard (+949 USD captés par l'exposition linéaire avant son entrée) "
        "et prend une exposition pleine (+1 038 USD de mieux pour l'exposition fractionnaire pendant le vote). Mais il "
        "évite les jours faibles : ne jamais se déclencher fait économiser 1 466 USD. Les sorties anticipées ne coûtent "
        "presque rien (−17 USD) ;",
        "   - l'exécution propre des cellules (stops intrabarre) et leurs coûts : −444 USD.",
        "",
        "   Aucun jour n'a vu la majorité des cellules gagner pendant que le vote restait flat. Le vote nul de S09 sur son "
        "OOS s'explique d'abord par ses **coûts de rotation**, puis par le retard d'entrée. Ce n'est pas un défaut "
        "d'agrégation qu'un autre seuil corrigerait proprement.",
        "",
        "Toute modification du vote serait une nouvelle hypothèse, à pré-enregistrer et à juger sur des données neuves.",
    ]
    (AUDIT / "07_vote.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def rerender() -> None:
    res = pd.read_csv(OUT / "07_vote.csv")
    gates = json.loads((OUT / "07_gates.json").read_text(encoding="utf-8"))
    br = json.loads((OUT / "07_s09_bridge.json").read_text(encoding="utf-8"))
    render(res, gates, br)


if __name__ == "__main__":
    if "--render" in sys.argv:
        rerender()
    else:
        main()
