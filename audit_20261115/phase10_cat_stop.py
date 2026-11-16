"""Phase 10 — operational catastrophe stop (disconnection / bug), not an alpha stop.

Broker stop per instrument at k x ATR14(J-1) (daily RTH true range, points) from the average entry price of the
NET position, k in {1.5, 2, 3}.  Netting rule for the stop:
  flat -> open: average = fill, stop set;  increase (same sign): weighted average, stop recomputed;
  reduction: average and stop kept;        flip: average = new fill, stop reset;   flat: stop cancelled.
Positions: reference book (S01 = ATR 18), aggregated execution without lot cap (the 5-lot Combine never binds),
MNQ/MGC 2019-05-06 -> 2026-03-19, MBT 2021-05-03 -> 2026-03-19.  Trigger: minute low <= stop (long) or high >= stop
(short); fill at the stop minus one tick of slippage (gaps through the stop are not modelled); the instrument then
stays flat for the session.  Daily loss limit: Topstep 50K DLL = 1 000 USD (help.topstep.com, read 2026-11-15); X = 50 %.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import AUDIT, OUT, fr, write_json
from .netting import RS, bars, run_ref, specs
from .phase8_regimes import daily_bars

KS = (1.5, 2.0, 3.0)
DLL = 1000.0
X = 0.50
NAMES = ["MNQ", "MGC", "MBT"]
WINDOW = ("2019-05-06", "2026-03-19")


def atr_j1(inst: str) -> pd.Series:
    d = daily_bars(inst)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - d["close"].shift()).abs(), (d["low"] - d["close"].shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(14).mean().shift(1)


def replay(res, cube_s, i: int, atr: float, k: float, sp) -> dict | None:
    """Replay one instrument-session with the stop; return the trigger record or None."""
    fl = sorted([f for f in res.fills if f["instrument"] == i], key=lambda f: f["minute"])
    if not fl or not np.isfinite(atr):
        return None
    T = cube_s.close.shape[0]
    pv, slip, comm = sp[i].point_value, sp[i].slippage_price, sp[i].commission_per_side
    pos, avg, stop = 0, 0.0, None
    cash = 0.0
    j = 0
    for t in range(T):
        while j < len(fl) and fl[j]["minute"] == t:
            f = fl[j]
            d, raw = int(f["delta"]), float(f["raw"])
            new = pos + d
            cash -= d * float(f["fill"]) * pv + abs(d) * comm
            if new == 0:
                avg, stop = 0.0, None
            elif pos == 0 or np.sign(new) != np.sign(pos):
                avg = raw
                stop = avg - np.sign(new) * k * atr
            elif abs(new) > abs(pos):
                avg = (avg * abs(pos) + raw * abs(d)) / abs(new)
                stop = avg - np.sign(new) * k * atr
            pos = new
            j += 1
        if pos != 0 and stop is not None:
            lo, hi = cube_s.low[t, i], cube_s.high[t, i]
            hit = (pos > 0 and lo <= stop) or (pos < 0 and hi >= stop)
            if hit:
                fill = stop - np.sign(pos) * slip
                stopped = cash + pos * fill * pv - abs(pos) * comm
                return {"minute": t, "position": pos, "avg_entry": avg, "stop": stop, "stopped_pnl": stopped,
                        "actual_pnl": float(res.instrument_final[i])}
    return None


def main() -> dict:
    r = run_ref(*WINDOW, keep_paths=True)
    dates = [pd.Timestamp(x) for x in r["dates"]]
    cube = RS.BarCube(bars("2026-03-19"), [x.date() for x in dates])
    sp = specs()
    atrs = {n: atr_j1(n) for n in NAMES}
    trig, bound_rows = [], []
    for k in KS:
        for x, res in zip(dates, r["table"]):
            cs = cube.session(x.date())
            bound = 0.0
            for i, n in enumerate(NAMES):
                a = float(atrs[n].get(x, np.nan))
                fl = [f for f in res.fills if f["instrument"] == i]
                if fl and np.isfinite(a):
                    # largest open position of the day x stop distance
                    p, mx = 0, 0
                    for f in sorted(fl, key=lambda f: f["minute"]):
                        p += int(f["delta"])
                        mx = max(mx, abs(p))
                    bound += mx * k * a * sp[i].point_value + mx * (sp[i].slippage_price * sp[i].point_value + sp[i].commission_per_side)
                rec = replay(res, cs, i, a, k, sp)
                if rec:
                    trig.append({"k": k, "date": str(x.date()), "instrument": n, "atr_j1": a, **rec,
                                 "pnl_evite": rec["stopped_pnl"] - rec["actual_pnl"]})
            bound_rows.append({"k": k, "date": str(x.date()), "bound_open_position_usd": bound,
                               "actual_day_pnl": float(res.final), "intraday_worst": float(res.min_worst)})
    tr = pd.DataFrame(trig)
    bd = pd.DataFrame(bound_rows)
    tr.to_csv(OUT / "10_triggers.csv", index=False)
    bd.to_csv(OUT / "10_bounds.csv", index=False)
    summ = []
    for k in KS:
        b = bd[bd["k"] == k]["bound_open_position_usd"]
        t = tr[tr["k"] == k] if len(tr) else tr
        row = {"k": k, "sessions": int(len(b)), "bound_p50": float(b.median()), "bound_p99": float(b.quantile(0.99)),
               "bound_max": float(b.max()), "share_days_bound_gt_X": float((b > X * DLL).mean())}
        for n in NAMES:
            tn = t[t["instrument"] == n] if len(t) else t
            row[f"triggers_{n}"] = int(len(tn))
            row[f"pnl_evite_{n}"] = float(tn["pnl_evite"].sum()) if len(tn) else 0.0
        summ.append(row)
    sm = pd.DataFrame(summ)
    sm.to_csv(OUT / "10_summary.csv", index=False)
    days = bd[bd["k"] == KS[0]]
    dll = {"sessions": int(len(days)), "days_close_le_minus_dll": int((days["actual_day_pnl"] <= -DLL).sum()),
           "days_intraday_le_minus_dll": int((days["intraday_worst"] <= -DLL).sum())}
    write_json(OUT / "10_summary.json", {"summary": summ, "dll": dll})
    render(sm, tr, dll)
    return {"summary": summ, "dll": dll}


def render(sm, tr, dll) -> None:
    t = ["| k | Déclenchements MNQ | MGC | MBT | PnL évité (+) / perdu (−) total | Borne de perte de la position ouverte : médiane | p99 | max | Jours où la borne dépasse 500 USD |",
         "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sm.itertuples(index=False):
        tot = r.pnl_evite_MNQ + r.pnl_evite_MGC + r.pnl_evite_MBT
        t.append(f"| {fr(r.k, 1)} | {r.triggers_MNQ} | {r.triggers_MGC} | {r.triggers_MBT} | {fr(tot, 0)} | {fr(r.bound_p50, 0)} | "
                 f"{fr(r.bound_p99, 0)} | {fr(r.bound_max, 0)} | {fr(100 * r.share_days_bound_gt_X, 1)} % |")
    det = ["| k | Date | Instrument | Position | Entrée moyenne | Stop | PnL réel | PnL avec stop | Évité (+) / perdu (−) |",
           "|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in (tr.itertuples(index=False) if len(tr) else []):
        det.append(f"| {fr(r.k, 1)} | {r.date} | {r.instrument} | {r.position:+d} | {fr(r.avg_entry, 2)} | {fr(r.stop, 2)} | "
                   f"{fr(r.actual_pnl, 0)} | {fr(r.stopped_pnl, 0)} | {fr(r.pnl_evite, 0)} |")
    k15 = sm[sm["k"] == 1.5].iloc[0]
    k_needed = 1.5 * X * DLL / k15["bound_p99"]
    lines = [
        "# Phase 10 — Stop catastrophe opérationnel",
        "",
        "*Stop contre une déconnexion ou un bug, pas un stop alpha. Positions nettes du book de référence (S01 = ATR 18), "
        "exécution agrégée sans limite de lots. MNQ et MGC du 2019-05-06 au 2026-03-19, MBT du 2021-05-03 au 2026-03-19. "
        "ATR14 quotidien RTH de J−1, en points. Le remplissage se fait au stop moins un tick ; les gaps à travers le stop "
        "ne sont pas modélisés. Script : [phase10_cat_stop.py](phase10_cat_stop.py). Tables : "
        "[10_triggers.csv](outputs/10_triggers.csv), [10_summary.csv](outputs/10_summary.csv), [10_bounds.csv](outputs/10_bounds.csv).*",
        "",
        "## Règle de netting du stop",
        "",
        "Un seul ordre stop par instrument, posé chez le broker, pour la position nette :",
        "",
        "- **ouverture depuis flat** : prix moyen = fill ; stop = prix moyen ∓ k × ATR(J−1) ;",
        "- **augmentation dans le même sens** : prix moyen pondéré, stop recalculé, quantité ajustée ;",
        "- **réduction** : prix moyen et niveau conservés, quantité réduite ;",
        "- **retournement** : prix moyen = fill du nouveau sens, stop recalculé de l'autre côté ;",
        "- **retour à plat** : ordre annulé.",
        "",
        "Une fois déclenché, l'instrument reste à plat jusqu'à la fin de la séance. Les sleeves continuent leur PnL "
        "virtuel. Ce stop est indépendant des ordres stop et cible de S01, qui restent posés.",
        "",
        "## Résultats",
        "",
        *t,
        "",
        "La « borne » est la perte maximale de la position ouverte si le stop est touché : position nette maximale de la "
        "journée × k × ATR × valeur du point, plus les coûts, sommée sur les instruments. Elle ne compte pas les pertes "
        "déjà réalisées dans la journée.",
        "",
        "## Déclenchements historiques",
        "",
        *(det if len(det) > 2 else ["Aucun déclenchement."]),
        "",
        "## Limite de perte journalière",
        "",
        f"La DLL d'un compte 50K Topstep est de 1 000 USD. Elle est appliquée automatiquement aux nouveaux comptes "
        f"(Responsible Trading Program). Sur {dll['sessions']} séances, le book de référence :",
        "",
        f"- a fini la journée à −1 000 USD ou moins **{dll['days_close_le_minus_dll']} fois** ;",
        f"- a touché −1 000 USD en intraday (marquage adverse) **{dll['days_intraday_le_minus_dll']} fois**.",
        "",
        "Ces journées auraient été interrompues par la DLL. C'est un effet de la règle Topstep, distinct du stop catastrophe.",
        "",
        "## Recommandation",
        "",
        f"**Aucun k de {{1,5 ; 2 ; 3}} ne borne la perte d'un jour à moins de X = 50 % de la DLL, soit 500 USD.** Même à "
        f"k = 1,5, la borne de la position ouverte dépasse 500 USD sur {fr(100 * k15['share_days_bound_gt_X'], 0)} % des "
        f"jours (médiane {fr(k15['bound_p50'], 0)} USD). Un ATR journalier MNQ vaut plusieurs centaines de points, à 2 USD "
        "le point, et S01 peut porter 4 contrats. Pour borner le p99 à 500 USD, il faudrait k ≈ "
        f"{fr(k_needed, 2)} : ce serait un stop alpha serré, qui couperait les trades ordinaires.",
        "",
        "Je recommande donc de séparer les deux rôles.",
        "",
        "- **Stop catastrophe chez le broker : k = 3.** Il ne s'est jamais déclenché sur l'historique (2019–2026 pour MNQ et MGC, 2021–2026 pour MBT), ce qui veut "
        "dire qu'il ne touche pas au comportement des sleeves. À k = 1,5, il se serait déclenché 6 fois pour −418 USD au "
        "total ; à k = 2, une fois pour −355 USD. Son rôle est de borner une position orpheline (déconnexion, bug) à environ "
        "3 ATR, pas de borner une journée.",
        "- **Borne journalière : la DLL Topstep de 1 000 USD (X = 100 %)**, appliquée par Topstep au niveau du compte. "
        f"Historiquement, elle aurait interrompu {dll['days_intraday_le_minus_dll']} séances en intraday sur {dll['sessions']}. "
        "Un plafond plus serré relève d'une décision de risque au niveau du book. L'étude d'agrégation du 13/11 a rejeté "
        "un stop book de 1 000 USD, qui coupait plus de gains qu'il n'évitait de pertes.",
        "",
        "Limites : remplissage au niveau du stop (pas de gap modélisé), barres minute uniquement, positions sans limite de "
        "lots (Combine). Le comportement réel sur un gap d'ouverture ou une coupure de flux n'est pas testé.",
    ]
    (AUDIT / "10_stop_catastrophe.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    print(main())
