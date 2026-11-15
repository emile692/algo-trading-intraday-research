"""Phase 5 — stop rules per sleeve and for the book: thresholds, false alarms, detection delays.

History: frozen daily PnL of each sleeve on its instrument calendar (MNQ/MGC 2019-05-06 -> 2026-03-19, MBT
2021-05-03 -> 2026-10-02, as the registered R0 thresholds); Book 8 on the union calendar 2021-05-03 -> 2026-03-19.
Bootstrap: stationary, mean block 20, seed 20261115 + 5.

Rules, evaluated each session from the start of monitoring (h = sessions since start):
  R0  rolling 126-session PnL < registered threshold (minimum of the same sum over the frozen history);
  R1  max drawdown since start < p5(h) of the bootstrap max-drawdown distribution at h (i.e. beyond its p95 severity);
  R2  same with p1(h) (p99 severity);
  R3  rolling 120-session PnL < p5 of the bootstrap 120-session PnL.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import ACTIVE8, AUDIT, CODES, INSTRUMENT, OUT, ROOT, SEED, fr, write_json
from .phase1_calendar import instrument_calendars
from .stats import stationary_indices

BLOCK, N_REF, N_SIM, LEN = 20, 10_000, 4_000, 500
HORIZONS = (20, 60, 120, 250)
HIST_END = {"MNQ": "2026-03-19", "MGC": "2026-03-19", "MBT": "2026-10-02"}
HIST_START = {"MNQ": "2019-05-06", "MGC": "2019-05-06", "MBT": "2021-05-03"}
RULES = ("R0", "R1", "R2", "R3", "R0∪R2")  # R0∪R2 = shadow trigger of the two-level scheme (b); added 2026-11-15
REINSTATE_N = (126, 252)


def histories() -> dict[str, pd.Series]:
    p = pd.read_csv(OUT / "daily_pnl_union.csv", parse_dates=["session_date"]).set_index("session_date")
    cals = instrument_calendars()
    out = {}
    for c in CODES:
        inst = INSTRUMENT[c]
        s = p[c].reindex(cals[inst])
        out[c] = s[(s.index >= HIST_START[inst]) & (s.index <= HIST_END[inst])]
    b = p[(p.index >= "2021-05-03") & (p.index <= "2026-03-19")]
    out["Book 8"] = b[ACTIVE8].sum(axis=1)
    for k, v in out.items():
        if v.isna().any():
            raise RuntimeError(k)
    return out


def r0_thresholds() -> dict[str, float]:
    t = pd.read_csv(ROOT / "research" / "locked_eight_sleeve_portfolio_20261115" / "sleeve_stop_rule_thresholds.csv")
    from .common import CODE_OF
    out = {CODE_OF[r.sleeve]: float(r.threshold_usd) for r in t.itertuples()}
    return out


def running_maxdd(m: np.ndarray) -> np.ndarray:
    eq = np.cumsum(m, axis=-1)
    peak = np.maximum.accumulate(np.concatenate([np.zeros(eq.shape[:-1] + (1,)), eq], axis=-1), axis=-1)[..., 1:]
    return np.minimum.accumulate(eq - peak, axis=-1)


def rolling_sum(m: np.ndarray, w: int) -> np.ndarray:
    """Rolling sum along the last axis; NaN before w observations."""
    c = np.cumsum(m, axis=-1)
    out = np.full(m.shape, np.nan)
    out[..., w - 1] = c[..., w - 1]
    out[..., w:] = c[..., w:] - c[..., :-w]
    return out


def first_trigger(flags: np.ndarray) -> np.ndarray:
    """Index of the first True along the last axis, -1 if none."""
    any_ = flags.any(axis=-1)
    return np.where(any_, flags.argmax(axis=-1), -1)


def rule_flags(m: np.ndarray, th: dict) -> dict[str, np.ndarray]:
    dd = running_maxdd(m)
    T = m.shape[-1]
    out = {}
    r126 = rolling_sum(m, 126)
    out["R0"] = np.nan_to_num(r126, nan=np.inf) < th["R0"] if th["R0"] is not None else np.zeros(m.shape, bool)
    out["R1"] = dd < th["dd_p05"][:T]
    out["R2"] = dd < th["dd_p01"][:T]
    r120 = rolling_sum(m, 120)
    out["R3"] = np.nan_to_num(r120, nan=np.inf) < th["pnl120_p05"]
    out["R0∪R2"] = out["R0"] | out["R2"]
    return out


def historical_episodes(x: np.ndarray, th: dict) -> dict[str, dict]:
    """Rules rolled over the full history with fixed-length windows (250 for R1/R2)."""
    T = len(x)
    eq = np.r_[0.0, np.cumsum(x)]
    res = {}
    # R0 and R3 on the whole history; R1/R2 on trailing 250-session windows.
    r126 = pd.Series(x).rolling(126).sum().to_numpy()
    r120 = pd.Series(x).rolling(120).sum().to_numpy()
    trail = np.full(T, np.nan)
    for t in range(249, T):
        w = x[t - 249:t + 1]
        e = np.cumsum(w)
        trail[t] = (e - np.maximum.accumulate(np.r_[0.0, e])[1:]).min()
    flags = {"R0": np.nan_to_num(r126, nan=np.inf) < (th["R0"] if th["R0"] is not None else -np.inf),
             "R1": np.nan_to_num(trail, nan=0.0) < th["dd_p05"][249],
             "R2": np.nan_to_num(trail, nan=0.0) < th["dd_p01"][249],
             "R3": np.nan_to_num(r120, nan=np.inf) < th["pnl120_p05"]}
    flags["R0∪R2"] = flags["R0"] | flags["R2"]
    for rule, f in flags.items():
        starts = np.flatnonzero(f & ~np.r_[False, f[:-1]])
        pos120 = [bool(eq[min(t + 1 + 120, T)] - eq[t + 1] > 0) for t in starts if t + 120 < T]
        pos250 = [bool(eq[min(t + 1 + 250, T)] - eq[t + 1] > 0) for t in starts if t + 250 < T]
        res[rule] = {"episodes": int(len(starts)), "sessions_in_alarm": int(f.sum()),
                     "first_dates_idx": starts.tolist(),
                     "followed_by_positive_120": int(sum(pos120)), "evaluable_120": len(pos120),
                     "followed_by_positive_250": int(sum(pos250)), "evaluable_250": len(pos250)}
    return res


def main() -> dict:
    rng = np.random.default_rng(SEED + 5)
    hist = histories()
    r0 = r0_thresholds()
    rows, dist_rows, hist_rows, reint_rows = [], [], [], []
    detail = {}
    curves = {}
    for name, s in hist.items():
        x = s.to_numpy(float)
        sd = x.std(ddof=1)
        idx = stationary_indices(len(x), N_REF, BLOCK, rng, length=LEN)
        ref = x[idx]
        dd = running_maxdd(ref)
        cum = np.cumsum(ref, axis=1)
        th = {"R0": r0.get(name), "dd_p05": np.percentile(dd, 5, axis=0), "dd_p01": np.percentile(dd, 1, axis=0),
              "pnl120_p05": float(np.percentile(cum[:, 119], 5)), "cum_p05": np.percentile(cum, 5, axis=0)}
        for h in HORIZONS:
            dist_rows.append({"série": name, "H": h, "pnl_p05": np.percentile(cum[:, h - 1], 5),
                              "pnl_p50": np.percentile(cum[:, h - 1], 50), "maxdd_p05": th["dd_p05"][h - 1],
                              "maxdd_p01": th["dd_p01"][h - 1], "maxdd_p50": np.percentile(dd[:, h - 1], 50)})
        # Forward behaviour: healthy (historical Sharpe), true Sharpe 0 and -0.5.
        scen = {"historique": x, "SR 0": x - x.mean(), "SR −0,5": x - x.mean() - 0.5 * sd / np.sqrt(252.0)}
        for label, base in scen.items():
            sim = base[stationary_indices(len(base), N_SIM, BLOCK, rng, length=LEN)]
            fl = rule_flags(sim, th)
            for rule in RULES:
                ft = first_trigger(fl[rule])
                det250 = float(((ft >= 0) & (ft < 250)).mean())
                det500 = float((ft >= 0).mean())
                hit = ft[ft >= 0] + 1
                rows.append({"série": name, "scénario": label, "règle": rule, "P(alarme ≤ 250)": det250,
                             "P(alarme ≤ 500)": det500,
                             "délai médian": float(np.median(hit)) if len(hit) else np.nan,
                             "délai moyen": float(np.mean(hit)) if len(hit) else np.nan})
            cum_s = np.cumsum(sim, axis=1)
            for N in REINSTATE_N:
                ok = (cum_s[:, :N] >= th["cum_p05"][:N]).all(axis=1) & (cum_s[:, N - 1] >= 0)
                reint_rows.append({"série": name, "scénario": label, "N": N, "P(réintégration)": float(ok.mean())})
        he = historical_episodes(x, th)
        for rule, v in he.items():
            first = [str(s.index[i].date()) for i in v["first_dates_idx"]][:6]
            hist_rows.append({"série": name, "règle": rule, "épisodes": v["episodes"], "séances en alarme": v["sessions_in_alarm"],
                              "suivis de PnL > 0 à +120": f"{v['followed_by_positive_120']}/{v['evaluable_120']}",
                              "suivis de PnL > 0 à +250": f"{v['followed_by_positive_250']}/{v['evaluable_250']}",
                              "premiers déclenchements": ", ".join(first)})
        curves[name] = th["dd_p01"]
        detail[name] = {"history": [str(s.index.min().date()), str(s.index.max().date()), len(s)],
                        "R0": th["R0"], "R1_dd_p05_250": float(th["dd_p05"][249]), "R2_dd_p01_250": float(th["dd_p01"][249]),
                        "R3_pnl120_p05": th["pnl120_p05"]}
    det = pd.DataFrame(rows)
    dist = pd.DataFrame(dist_rows)
    hst = pd.DataFrame(hist_rows)
    rei = pd.DataFrame(reint_rows)
    det.to_csv(OUT / "05_detection.csv", index=False)
    dist.to_csv(OUT / "05_distributions.csv", index=False)
    hst.to_csv(OUT / "05_historical_alarms.csv", index=False)
    rei.to_csv(OUT / "05_reinstatement.csv", index=False)
    write_json(OUT / "05_thresholds.json", detail)
    # R2 threshold curve: p1 of the bootstrap max drawdown after h sessions of tracking, h = 1..500 (rule of 2026-11-15).
    pd.DataFrame({"h": np.arange(1, LEN + 1), **{k: np.round(v, 2) for k, v in curves.items()}}).to_csv(
        OUT / "05_r2_threshold_curve.csv", index=False)
    write_weekly_template()
    render(det, dist, hst, rei, detail)
    return {"series": len(hist)}


def write_weekly_template() -> None:
    cols = ["semaine_iso", "sleeve", "statut", "pnl_semaine_usd", "pnl_cumule_depuis_20261103_usd", "seances_depuis_debut",
            "pnl_roulant_126_usd", "seuil_R0_usd", "ecart_R0_usd", "maxdd_depuis_debut_usd", "seuil_dd_h_usd",
            "pnl_roulant_120_usd", "seuil_R3_usd", "alarme", "commentaire"]
    pd.DataFrame(columns=cols).to_csv(AUDIT / "05_suivi_hebdo_gabarit.csv", index=False)


def render(det, dist, hst, rei, detail) -> None:
    def thr_table():
        t = ["| Série | Historique | R0 : seuil 126 s. | R1 : maxDD p5 à 250 s. | R2 : maxDD p1 à 250 s. | R3 : p5 PnL 120 s. |",
             "|---|---|---:|---:|---:|---:|"]
        for k, v in detail.items():
            t.append(f"| {k} | {v['history'][0]} → {v['history'][1]} ({v['history'][2]}) | "
                     f"{fr(v['R0']) if v['R0'] is not None else 'n/d'} | {fr(v['R1_dd_p05_250'])} | {fr(v['R2_dd_p01_250'])} | {fr(v['R3_pnl120_p05'])} |")
        return t

    def det_table():
        t = ["| Série | Règle | Fausse alarme ≤ 250 s. (sleeve saine) | Détection ≤ 250 s., SR vrai 0 | délai médian | "
             "Détection ≤ 250 s., SR vrai −0,5 | délai médian |", "|---|---|---:|---:|---:|---:|---:|"]
        for (name, rule), g in det.groupby(["série", "règle"], sort=False):
            g = g.set_index("scénario")
            h, z, n = g.loc["historique"], g.loc["SR 0"], g.loc["SR −0,5"]
            t.append(f"| {name} | {rule} | {fr(100 * h['P(alarme ≤ 250)'], 1)} % | {fr(100 * z['P(alarme ≤ 250)'], 1)} % | "
                     f"{fr(z['délai médian'], 0)} | {fr(100 * n['P(alarme ≤ 250)'], 1)} % | {fr(n['délai médian'], 0)} |")
        return t

    def summary_table():
        sl = det[det["série"] != "Book 8"]
        g = sl.groupby(["règle", "scénario"])["P(alarme ≤ 250)"].median().unstack()
        d = sl.groupby(["règle", "scénario"])["délai médian"].median().unstack()
        t = ["| Règle | Fausse alarme médiane ≤ 250 s. | Détection médiane ≤ 250 s., SR 0 | Détection médiane ≤ 250 s., SR −0,5 | "
             "Délai médian, SR −0,5 |", "|---|---:|---:|---:|---:|"]
        for r in RULES:
            t.append(f"| {r} | {fr(100 * g.loc[r, 'historique'], 1)} % | {fr(100 * g.loc[r, 'SR 0'], 1)} % | "
                     f"{fr(100 * g.loc[r, 'SR −0,5'], 1)} % | {fr(d.loc[r, 'SR −0,5'], 0)} séances |")
        return t

    hist_t = ["| Série | Règle | Épisodes | Séances en alarme | Suivis de PnL > 0 à +120 | à +250 | Premiers déclenchements |",
              "|---|---|---:|---:|---|---|---|"]
    for r in hst.itertuples(index=False):
        if r.règle == "R0":
            continue
        hist_t.append(f"| {r.série} | {r.règle} | {r.épisodes} | {r[3]} | {r[4]} | {r[5]} | {r[6]} |")
    rei_t = ["| Série | N | P(réintégration), sleeve saine | SR vrai 0 | SR vrai −0,5 |", "|---|---:|---:|---:|---:|"]
    for (name, N), g in rei.groupby(["série", "N"], sort=False):
        g = g.set_index("scénario")["P(réintégration)"]
        rei_t.append(f"| {name} | {N} | {fr(100 * g['historique'], 1)} % | {fr(100 * g['SR 0'], 1)} % | {fr(100 * g['SR −0,5'], 1)} % |")

    lines = [
        "# Phase 5 — Règles d'arrêt",
        "",
        "*Historique gelé de chaque sleeve, sur le calendrier de son instrument : MNQ/MGC du 2019-05-06 au 2026-03-19, "
        "MBT du 2021-05-03 au 2026-10-02. Book 8 sur le calendrier union, du 2021-05-03 au 2026-03-19. S01 = ATR 18. "
        "Bootstrap stationnaire, bloc moyen de 20 séances : 10 000 chemins de 500 séances pour les seuils, 4 000 par "
        "scénario pour les délais ; seed 20261120. Script : [phase5_stop_rules.py](phase5_stop_rules.py). Tables : "
        "[05_detection.csv](outputs/05_detection.csv), [05_historical_alarms.csv](outputs/05_historical_alarms.csv), "
        "[05_distributions.csv](outputs/05_distributions.csv), [05_reinstatement.csv](outputs/05_reinstatement.csv).*",
        "",
        "## Point de départ : une règle existe déjà",
        "",
        "Contrairement à ce que dit la mission, une règle d'arrêt est enregistrée depuis le 2026-11-03 "
        "(`ACTIVE_EIGHT_SLEEVES.md`). C'est **R0** : la somme des 126 dernières séances doit rester au-dessus du minimum "
        "de cette même somme sur l'historique gelé. Par construction, R0 ne s'est jamais déclenchée sur l'historique. "
        "Ses fausses alarmes et ses délais ne se mesurent donc qu'en simulation.",
        "",
        "## Règles comparées",
        "",
        "Chaque règle est évaluée à chaque séance, depuis le début du suivi (h séances écoulées).",
        "",
        "- **R0**, la règle en vigueur : PnL sur 126 séances inférieur au minimum historique de cette somme.",
        "- **R1** : drawdown maximal depuis le début inférieur au p5(h) du maxDD bootstrap, soit une sévérité au-delà du p95.",
        "- **R2** : même chose avec le p1(h), soit une sévérité au-delà du p99.",
        "- **R3** : PnL roulant sur 120 séances inférieur au p5 du PnL bootstrap à 120 séances.",
        "",
        "## Seuils",
        "",
        *thr_table(),
        "",
        "Les distributions à 20, 60, 120 et 250 séances (p5, p50 du PnL ; p50, p5, p1 du maxDD) sont dans "
        "[05_distributions.csv](outputs/05_distributions.csv).",
        "",
        "## Compromis fausses alarmes / délai — synthèse (médiane des 9 sleeves)",
        "",
        "- **Fausse alarme** : probabilité qu'une sleeve dont le vrai Sharpe est celui de son historique soit coupée "
        "dans ses 250 premières séances de suivi.",
        "- **Détection** : même probabilité quand le PnL quotidien est décalé pour un Sharpe vrai de 0, puis de −0,5 "
        "(volatilité inchangée).",
        "- **Délai** : nombre de séances avant la première alarme, parmi les chemins qui en déclenchent une sur 500 séances.",
        "",
        *summary_table(),
        "",
        "## Détail par série",
        "",
        *det_table(),
        "",
        "## Application glissante sur l'historique (R1 à R3)",
        "",
        "R1 et R2 sont appliquées ici au maxDD de chaque fenêtre glissante de 250 séances, R3 au PnL glissant sur "
        "120 séances. Une sleeve « restée positive ensuite » est une sleeve dont le PnL des 120 (ou 250) séances "
        "suivant le déclenchement est positif : c'est une fausse alarme ex post. Ces seuils sont calibrés sur le même "
        "historique, ce qui sous-estime les déclenchements.",
        "",
        *hist_t,
        "",
        "## Réintégration proposée",
        "",
        "Une sleeve coupée passe en shadow. Elle revient après N séances de shadow si son PnL cumulé en shadow n'est "
        "jamais passé sous le p5(h) et qu'il est ≥ 0 à N. Table : probabilité de réintégration.",
        "",
        *rei_t,
        "",
        "## Format de suivi hebdomadaire",
        "",
        "Gabarit : [05_suivi_hebdo_gabarit.csv](05_suivi_hebdo_gabarit.csv). Une ligne par sleeve et par semaine ISO, plus "
        "une ligne « Book 8 ». Elle reprend : le PnL de la semaine, le cumul depuis le 2026-11-03, le roulant sur 126 "
        "séances et son écart au seuil R0, le maxDD depuis le début et le seuil R1/R2 à h, le roulant sur 120 séances "
        "et le seuil R3, l'alarme et un commentaire. Toute alarme déclenche une revue, jamais une action automatique.",
        "",
        "## Lecture et choix qui vous revient",
        "",
        "1. **Aucune règle ne détecte vite sans fausses alarmes nombreuses.** C'est la conséquence directe de la phase 2 : "
        "les sleeves ont des Sharpe de 0,4 à 1,3, donc 250 séances ne suffisent pas à distinguer une sleeve saine "
        "d'une sleeve à Sharpe nul.",
        "2. **R0, la règle en vigueur, est lente sans être sûre.** Elle coupe à tort environ 17 % des sleeves saines en "
        "250 séances (médiane), ne détecte un Sharpe vrai nul que 31 % du temps, et met environ 200 séances quand elle "
        "le fait. Son seuil « pire que tout l'historique » n'empêche pas les fausses alarmes, parce que l'historique "
        "contient moins de fenêtres indépendantes que le futur n'en tire.",
        "3. **R2 (drawdown au-delà du p99) est la seule règle à faible taux de fausses alarmes** (environ 7 %). Elle "
        "rate en revanche la plupart des dégradations : 14 % de détection à SR 0, 23 % à SR −0,5.",
        "4. **R1 et R3 détectent plus, mais coupent à tort une sleeve saine sur quatre environ.** R1 est la plus rapide "
        "(environ 80 séances), R3 la plus sensible (66 % de détection à SR −0,5).",
        "5. **Au niveau du book, la détection est bien meilleure**, parce que son Sharpe est plus élevé : R3 détecte 87 % "
        "des books à Sharpe vrai nul en 250 séances, pour 25 % de fausses alarmes, et R2 en détecte 41 % pour 7 %.",
        "6. **La réintégration par le shadow discrimine mal.** Après 252 séances, une sleeve saine n'est réintégrée "
        "que 48 à 65 % du temps, et une sleeve à Sharpe nul l'est encore 20 à 38 % du temps.",
        "7. **Sur l'historique**, R1 aurait signalé **S09** dès le 2025-08-13, au début de son legacy OOS (144 séances "
        "en alarme), et R3 signale **S08** en juin-août 2026.",
        "",
        "8. **Combiner R0 et R2 pour le passage en shadow coûte cher en fausses alarmes.** Avec la règle « R0 ou R2 » "
        "(ligne R0∪R2, ajoutée le 2026-11-15 sur les mêmes tirages), une sleeve saine est coupée à tort dans 21 % des cas "
        "(médiane), contre 7 % pour R2 seule. La détection monte à 36 % à SR 0 et à 59 % à SR −0,5. **Correction** : "
        "la première version de cette page annonçait environ 7 % pour un passage en shadow sur « R2 ou R0 ». Ce chiffre "
        "ne valait que pour R2 seule.",
        "",
        "**Proposition, à décider par vous**, à deux niveaux, uniforme pour toutes les sleeves. Version corrigée le "
        "2026-11-15 : R0 devient une alerte.",
        "",
        "- **alerte « revue »** si R3 (PnL sur 120 séances < p5) ou R0 (règle déjà enregistrée) se déclenche, sans "
        "action automatique ;",
        "- **passage en shadow** si R2 se déclenche (drawdown depuis le début du suivi au-delà du p99 à h) ;",
        "- **réintégration** après 252 séances de shadow, par une décision versionnée, sur les critères de réintégration "
        "ci-dessus ;",
        "- **au niveau du book**, R3 en alerte et R2 en revue générale.",
        "",
        "Le compromis assumé : l'action automatique coupe peu de sleeves saines (environ 7 %), au prix d'une détection "
        "lente. Les alertes R3 et R0 servent à regarder plus tôt. Garder R0 comme déclencheur du shadow ferait passer "
        "les fausses coupures à 21 %.",
        "",
        "Limites : les seuils sont estimés sur l'historique déjà vu, qui inclut la validation réutilisée. Le bootstrap "
        "suppose que la distribution passée reste valable. Pour S01, dont la fréquence de trades s'est effondrée en "
        "2026 (phase 4), la distribution historique surestime la volatilité actuelle : une dégradation de S01 serait "
        "encore plus lente à détecter.",
    ]
    (AUDIT / "05_regles_arret.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
