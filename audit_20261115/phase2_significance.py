"""Phase 2 — statistical significance of each sleeve and of the 8- and 9-sleeve books.

Calendars: a sleeve is measured on the sessions of its own instrument (union calendar restricted to the days its
instrument has a session file); the books on the union calendar of phase 1.  Bootstrap: Politis-Romano stationary
bootstrap, mean blocks 5 and 20, 10 000 draws, seed 20261115 (one index matrix per window length and block, shared by
all series of that window so that book and sleeves are resampled jointly).
"""

from __future__ import annotations

import subprocess
from io import StringIO

import numpy as np
import pandas as pd

from .common import ACTIVE8, AUDIT, CACHE, CODES, INSTRUMENT, OUT, ROOT, SEED, WINDOWS, fr, write_json
from .phase1_calendar import DATA_RANGE, instrument_calendars
from .stats import boot_stats, dsr, expected_max_sharpe, lo_sharpe, psr, stationary_indices

N_BOOT = 10_000
BLOCKS = (5, 20)
BOOK_WINDOWS = {"is": ("2021-05-03", "2023-12-31"), "val": ("2024-01-02", "2026-03-19"),
                "full": ("2021-05-03", "2026-03-19")}
SCOPE_FR = {"is": "IS", "val": "validation réutilisée", "full": "total"}


def panel() -> pd.DataFrame:
    return pd.read_csv(OUT / "daily_pnl_union.csv", parse_dates=["session_date"]).set_index("session_date")


def sleeve_series(p: pd.DataFrame, code: str, a: str, b: str) -> pd.Series:
    cal = instrument_calendars()[INSTRUMENT[code]]
    s = p[code].reindex(cal)
    s = s[(s.index >= pd.Timestamp(a)) & (s.index <= pd.Timestamp(b))]
    if s.isna().any():
        raise RuntimeError(f"{code}: NaN in {a}..{b}")
    return s


def book_series(p: pd.DataFrame, cols: list[str], a: str, b: str) -> pd.Series:
    s = p.loc[(p.index >= pd.Timestamp(a)) & (p.index <= pd.Timestamp(b)), cols]
    if s.isna().any().any():
        raise RuntimeError(f"book NaN in {a}..{b}")
    return s.sum(axis=1)


# ----------------------------------------------------------------------------- trials for the DSR
def _git_csv(rev_path: str) -> pd.DataFrame:
    raw = subprocess.run(["git", "show", rev_path], cwd=ROOT, check=True, capture_output=True).stdout.decode("utf-8")
    return pd.read_csv(StringIO(raw))


def trials() -> dict[str, dict]:
    """IS Sharpe of every documented trial, and the low / high N hypotheses."""
    ref = ROOT / "research" / "signal_sheet_parameter_refinement"
    out = {}
    cube01 = CACHE / "s01_cube_metrics_is.csv"
    if not cube01.exists():
        _git_csv("72ded70^:data/exports/atr_local_robustness_20260320_194355/tables/metrics_is.csv").to_csv(cube01, index=False)
    c01 = pd.read_csv(cube01)
    out["S01"] = {"sr": c01["is_sharpe_ratio"].to_numpy(), "n_low": len(c01), "n_high": len(c01) + 224 + 16,
                  "source": "cube ATR × q_low × q_high, 160 cellules (git 72ded70^, data/exports/atr_local_robustness_20260320_194355)",
                  "high_note": "+ 224 variantes des campagnes ORB archivées (metrics_by_scope.csv) + 16 variantes des deux campagnes atr_ensemble du 2026-03-20",
                  "convention_note": "Sharpe du cube calculé dans l'ancien moteur ORB (avant la correction du cutoff du 18/09) ; sa dispersion est utilisée telle quelle"}
    c02 = pd.read_csv(ROOT / "research" / "signal_sheet_02_cube" / "cube_metrics.csv")
    out["S02"] = {"sr": c02["is_sharpe_annualized"].to_numpy(), "n_low": len(c02), "n_high": len(c02) + 20,
                  "source": "cube S02, 770 cellules (research/signal_sheet_02_cube)",
                  "high_note": "+ 20 overlays de sortie du money_management_lab",
                  "convention_note": "Sharpe du cube en rendements d'un compte de 50K (même convention que la fiche 02)"}
    mbt = pd.read_csv(ROOT / "research" / "mbt_three_sleeve_freeze_20261004" / "cell_metrics.csv")
    mbt_cells = mbt.groupby("mechanism_id")["cell_id"].nunique().to_dict()
    extra = {"S03": (0, ""), "S04": (20, "+ 20 overlays de sortie du money_management_lab"),
             "S05": (162, "+ 162 cellules de mgc_volatility_breakout_refinement_v0_20260925"),
             "S06": (0, ""),
             "S07": (mbt_cells.get("rolling_range_breakout", 0), "+ cellules de mbt_three_sleeve_freeze_20261004"),
             "S08": (mbt_cells.get("trend_pullback_ma_structure", 0), "+ cellules de mbt_three_sleeve_freeze_20261004"),
             "S09": (mbt_cells.get("volatility_breakout", 0), "+ cellules de mbt_three_sleeve_freeze_20261004")}
    for code in CODES[2:]:
        g = pd.read_csv(ref / code.lower() / "grid_metrics.csv")
        g = g[g["scope"] == "is"]
        out[code] = {"sr": g["sharpe_annualized"].to_numpy(), "n_low": len(g), "n_high": len(g) + extra[code][0],
                     "source": f"grille de raffinement {code.lower()}, {len(g)} cellules (research/signal_sheet_parameter_refinement/{code.lower()})",
                     "high_note": extra[code][1] or "aucune campagne antérieure retrouvée : N haut = N bas",
                     "convention_note": "Sharpe en USD (metric_block)"}
    return out


# ----------------------------------------------------------------------------- main
def classify(r: dict) -> str:
    if r["sharpe"] < 0:
        return "négative"
    if r["t_hac"] >= 1.96 and all(r[f"ci_lo_b{b}"] > 0 for b in BLOCKS):
        return "distinguable de zéro"
    return "indistinguable"


def analyse(series: dict[tuple[str, str], pd.Series], rng_seed: int) -> list[dict]:
    """series keyed by (label, scope); identical lengths share bootstrap indices."""
    rows = []
    cache: dict[tuple[int, int], np.ndarray] = {}
    rng = np.random.default_rng(rng_seed)
    for (label, scope), s in series.items():
        x = s.to_numpy(float)
        r = {"série": label, "fenêtre": scope, "début": str(s.index.min().date()), "fin": str(s.index.max().date()),
             "pnl": float(x.sum()), **lo_sharpe(x), "psr0": psr(x, 0.0)}
        for b in BLOCKS:
            key = (len(x), b)
            if key not in cache:
                cache[key] = stationary_indices(len(x), N_BOOT, b, rng)
            bs = boot_stats(x, cache[key])
            r[f"ci_lo_b{b}"], r[f"ci_hi_b{b}"] = np.percentile(bs["sharpe"], [2.5, 97.5])
        r["classe"] = classify(r)
        rows.append(r)
    return rows


def main() -> dict:
    p = panel()
    series: dict[tuple[str, str], pd.Series] = {}
    for code in CODES:
        w = WINDOWS["S01"] if code == "S01" else (WINDOWS["MBT"] if INSTRUMENT[code] == "MBT" else WINDOWS["MNQ_MGC"])
        for scope in ("is", "val", "full"):
            a, b = w[scope]
            b = min(pd.Timestamp(b), pd.Timestamp(DATA_RANGE[INSTRUMENT[code]][1])).strftime("%Y-%m-%d")
            series[(code, scope)] = sleeve_series(p, code, a, b)
    for name, cols in (("Book 8", ACTIVE8), ("Book 9", CODES)):
        for scope, (a, b) in BOOK_WINDOWS.items():
            series[(name, scope)] = book_series(p, cols, a, b)
    rows = analyse(series, SEED)
    res = pd.DataFrame(rows)

    tr = trials()
    drows = []
    for code in CODES:
        t = tr[code]
        v = float(np.var(t["sr"], ddof=1))
        for scope in ("is", "full"):
            x = series[(code, scope)].to_numpy(float)
            for hyp, n in (("bas", t["n_low"]), ("haut", t["n_high"]), ("stress", 1000)):
                d, sr0 = dsr(x, n, v)
                drows.append({"sleeve": code, "fenêtre": scope, "hypothèse": hyp, "N": n, "var_sr_essais": v,
                              "sd_sr_essais": v ** 0.5, "sr0_annualisé": sr0,
                              "sharpe": float(res[(res["série"] == code) & (res["fenêtre"] == scope)]["sharpe"].iloc[0]),
                              "dsr": d})
    dtab = pd.DataFrame(drows)
    res.to_csv(OUT / "02_significance.csv", index=False)
    dtab.to_csv(OUT / "02_deflated_sharpe.csv", index=False)
    meta = {code: {k: v for k, v in tr[code].items() if k != "sr"} | {"n_cells_with_sr": int(len(tr[code]["sr"]))}
            for code in CODES}
    write_json(OUT / "02_trials.json", meta)
    render(res, dtab, meta)
    return {"rows": len(res)}


def render(res: pd.DataFrame, dtab: pd.DataFrame, meta: dict) -> None:
    head = ["| Série | Fenêtre | Dates | Séances | PnL | Sharpe | SE Lo (HAC) | t | IC 95 % bloc 5 | IC 95 % bloc 20 | PSR(0) | Classe |",
            "|---|---|---|---:|---:|---:|---:|---:|---|---|---:|---|"]
    body = []
    for r in res.itertuples(index=False):
        body.append(f"| {r.série} | {SCOPE_FR[r.fenêtre]} | {r.début} → {r.fin} | {r.T} | {fr(r.pnl)} | {fr(r.sharpe, 3)} | "
                    f"{fr(r.se_hac, 3)} | {fr(r.t_hac, 2)} | [{fr(r.ci_lo_b5, 2)} ; {fr(r.ci_hi_b5, 2)}] | "
                    f"[{fr(r.ci_lo_b20, 2)} ; {fr(r.ci_hi_b20, 2)}] | {fr(r.psr0, 3)} | {r.classe} |")
    summ = ["| Sleeve | IS | Validation réutilisée | Total |", "|---|---|---|---|"]
    for code in CODES + ["Book 8", "Book 9"]:
        c = {r.fenêtre: r.classe for r in res[res["série"] == code].itertuples()}
        summ.append(f"| {code} | {c['is']} | {c['val']} | **{c['full']}** |")
    dsr_t = ["| Sleeve | Fenêtre | Sharpe | σ des Sharpe d'essais | N bas | SR₀ bas | DSR bas | N haut | SR₀ haut | DSR haut | DSR N = 1000 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for (code, scope), g in dtab.groupby(["sleeve", "fenêtre"], sort=False):
        g = g.set_index("hypothèse")
        dsr_t.append(f"| {code} | {SCOPE_FR[scope]} | {fr(g.loc['bas', 'sharpe'], 3)} | {fr(g.loc['bas', 'sd_sr_essais'], 3)} | "
                     f"{int(g.loc['bas', 'N'])} | {fr(g.loc['bas', 'sr0_annualisé'], 3)} | {fr(g.loc['bas', 'dsr'], 3)} | "
                     f"{int(g.loc['haut', 'N'])} | {fr(g.loc['haut', 'sr0_annualisé'], 3)} | {fr(g.loc['haut', 'dsr'], 3)} | "
                     f"{fr(g.loc['stress', 'dsr'], 3)} |")
    src = [f"- **{c}** : {meta[c]['source']} ; N haut : {meta[c]['high_note']}. {meta[c]['convention_note']}." for c in CODES]
    lines = [
        "# Phase 2 — Significativité statistique",
        "",
        "*Données : panel corrigé de la phase 1. S01 = ensemble ATR 18. PnL quotidien net en USD, coûts nominaux. "
        "Sharpe = √252 × moyenne / écart-type, jours sans trade inclus. Calendrier : séances de l'instrument pour une "
        "sleeve, calendrier union pour un book. Script : [phase2_significance.py](phase2_significance.py), "
        "[stats.py](stats.py). Tables : [02_significance.csv](outputs/02_significance.csv), "
        "[02_deflated_sharpe.csv](outputs/02_deflated_sharpe.csv).*",
        "",
        "## Méthodes",
        "",
        "- **Erreur type de Lo (2002)** : estimateur GMM du Sharpe, matrice de covariance de long terme de Newey-West "
        "(noyau de Bartlett, retard ⌊4 (T/100)^(2/9)⌋, soit 7 ou 8 jours). Elle tient compte de l'autocorrélation et "
        "des queues épaisses. t = Sharpe / SE.",
        "- **Bootstrap stationnaire de Politis-Romano** : blocs de longueur géométrique, moyennes 5 et 20 séances, "
        "10 000 tirages, seed 20261115. IC à 95 % par percentiles.",
        "- **PSR(0)** : probabilité que le vrai Sharpe soit positif, avec correction de skew et de kurtosis "
        "(Bailey et López de Prado 2012).",
        "- **DSR** : PSR contre SR₀, le Sharpe maximal attendu parmi N essais sans talent, compte tenu de la "
        "dispersion des Sharpe IS entre essais (Bailey et López de Prado 2014). Il est calculé sur l'IS, la fenêtre "
        "de sélection, et sur le total.",
        "- **Book** : IS 2021-05-03 → 2023-12-31, validation 2024-01-02 → 2026-03-19, total 2021-05-03 → 2026-03-19, "
        "soit les fenêtres où les trois instruments ont des données. Pour les sleeves MBT, 2024-01 → 2025-05 fait "
        "partie de leur IS : la « validation » du book est donc mixte.",
        "",
        "**Règle de classement, fixée avant le calcul** : « négative » si le Sharpe est < 0 ; « distinguable de zéro » "
        "si t ≥ 1,96 et que les bornes basses des deux IC bootstrap sont > 0 ; « indistinguable » sinon.",
        "",
        "## Classement",
        "",
        *summ,
        "",
        "## Détail",
        "",
        *head,
        *body,
        "",
        "## Deflated Sharpe Ratio",
        "",
        "Un DSR > 0,95 signifie que le Sharpe observé reste significatif après prise en compte du nombre d'essais. "
        "La colonne N = 1000 est un stress arbitraire, pour les essais non documentés.",
        "",
        *dsr_t,
        "",
        "Sources des essais :",
        "",
        *src,
        "",
        "## Lecture",
        "",
        "1. **Une seule sleeve est distinguable de zéro sur chaque fenêtre prise isolément : S01.** Son t de Lo vaut "
        "3,09 sur l'IS et 2,28 sur la validation. Son DSR vaut 0,91 à 0,94 sur l'IS selon N, et passe 0,95 sur le "
        "total. C'est la seule sleeve dont le DSR approche le seuil.",
        "2. **S02 et S09 ne sont « distinguables » que sur le total.**",
        "   - S02 n'atteint t ≥ 1,96 ni sur l'IS ni sur la validation. Après déflation par les 770 cellules du cube, son "
        "DSR tombe à 0,39–0,42 : son Sharpe IS de 0,86 est inférieur au maximum attendu par hasard parmi 770 essais "
        "(SR₀ ≈ 0,95).",
        "   - S09 doit tout à son IS (Sharpe 1,31). Il est nul sur la validation (−0,04), et son DSR sur le total vaut 0,53.",
        "3. **S03 à S08 sont indistinguables de zéro sur toutes les fenêtres.** S06 est la plus proche (t = 1,97 sur le "
        "total, DSR 0,91–0,93), mais un des IC bootstrap contient 0. S05 a le DSR le plus faible (0,07 à 0,33 sur le total).",
        "4. **Le book est distinguable de zéro sur toutes les fenêtres** (t de 2,8 à 4,8, borne basse de l'IC ≥ 0,32). "
        "C'est un effet de diversification : des sleeves individuellement faibles, peu corrélées, s'additionnent. Le DSR "
        "n'est pas calculé pour le book : aucun nombre d'essais au niveau du book n'est documenté. Or le choix des "
        "sleeves et de S04 a été fait en voyant la fenêtre 2024-01 → 2026-03.",
        "5. **La validation n'est pas propre.** Toutes les fenêtres « validation » ont été consultées avant le gel des "
        "clusters. Les t de validation sont donc un plafond, pas une preuve indépendante.",
        "",
        "Conséquence pratique : à l'exception de S01, aucune sleeve n'a à elle seule un historique qui permettrait de "
        "détecter rapidement une dégradation. C'est ce qui rend la règle d'arrêt de la phase 5 nécessairement lente "
        "ou bruitée.",
    ]
    (AUDIT / "02_significativite.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
