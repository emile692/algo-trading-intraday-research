"""Phase 6 — the same rules applied identically to the nine sleeves, with the data available on 2026-11-01.

Data as of 2026-11-01:
  S01      frozen history <= 2026-03-19 + the (O) one-shot window 2026-03-23 -> 2026-10-02 (outputs/04b_*);
  S02-S06  research/frozen_vote_forward_replay_20261101/s0X_daily.csv (frozen history + replay to 2026-10-02);
  S07-S09  frozen history to 2026-10-02.
Rules (phase 5 thresholds), rolled over the whole series: R0 (registered), R2 (trailing 250-session max drawdown beyond
its p99), R3 (120-session PnL < p5).  Status the proposed rule would give on 2026-11-01:
  shadow if R0 or R2 fired within the last 252 sessions (no reinstatement possible yet);
  revue  if R3 fired within the last 120 sessions, or the phase-2 class on the reused validation is "négative";
  actif  otherwise.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .common import AUDIT, CODES, INSTRUMENT, OUT, ROOT, fr, load_daily, write_json

REPLAY = ROOT / "research" / "frozen_vote_forward_replay_20261101"
CURRENT = {c: "actif" for c in CODES} | {"S04": "shadow"}


def series_as_of_20261101() -> dict[str, pd.Series]:
    out = {}
    s01 = load_daily("S01")
    s01 = s01[(s01.index.dayofweek < 5) & (s01.index <= "2026-03-19")]
    mnq_cal = load_daily("S02").index
    s01 = s01.reindex(mnq_cal).fillna(0.0)
    pre = json.loads((OUT / "04b_preread.json").read_text(encoding="utf-8"))
    win = pd.read_csv(OUT / "04b_s01_window_trades.csv", parse_dates=["session_date"])
    replay02 = pd.read_csv(REPLAY / "s02_daily.csv", parse_dates=["session_date"]).set_index("session_date")
    win_cal = replay02.index[(replay02.index >= "2026-03-20") & (replay02.index <= "2026-10-02")]
    if len(win_cal) != pre["H"]:
        raise RuntimeError(f"MNQ window calendar {len(win_cal)} != H {pre['H']}")
    s01_win = win.groupby("session_date")["net_pnl_usd"].sum().reindex(win_cal).fillna(0.0)
    out["S01"] = pd.concat([s01, s01_win])
    for c in ("S02", "S03", "S04", "S05", "S06"):
        out[c] = pd.read_csv(REPLAY / f"s0{c[2]}_daily.csv", parse_dates=["session_date"]).set_index("session_date")["daily_pnl_usd"]
        frozen = load_daily(c)
        if float((out[c].reindex(frozen.index) - frozen).abs().max()) > 1e-6:
            raise RuntimeError(f"{c}: replay differs from frozen history")
    for c in ("S07", "S08", "S09"):
        out[c] = load_daily(c)
    return out


def rolled(x: pd.Series, th: dict) -> pd.DataFrame:
    v = x.to_numpy(float)
    r126 = x.rolling(126).sum()
    r120 = x.rolling(120).sum()
    trail = np.full(len(v), np.nan)
    for t in range(249, len(v)):
        e = np.cumsum(v[t - 249:t + 1])
        trail[t] = (e - np.maximum.accumulate(np.r_[0.0, e])[1:]).min()
    return pd.DataFrame({"r126": r126, "r120": r120, "dd250": trail,
                         "R0": r126 < th["R0"], "R2": pd.Series(trail, index=x.index) < th["R2_dd_p01_250"],
                         "R1": pd.Series(trail, index=x.index) < th["R1_dd_p05_250"],
                         "R3": r120 < th["R3_pnl120_p05"]}, index=x.index)


def main() -> dict:
    th = json.loads((OUT / "05_thresholds.json").read_text(encoding="utf-8"))
    sig = pd.read_csv(OUT / "02_significance.csv")
    o = json.loads((OUT / "04b_result.json").read_text(encoding="utf-8"))
    data = series_as_of_20261101()
    rows = []
    for c in CODES:
        x = data[c]
        f = rolled(x, th[c])
        last252, last120 = f.iloc[-252:], f.iloc[-120:]
        fired = {r: f.index[f[r]].tolist() for r in ("R0", "R1", "R2", "R3")}
        cls_val = sig[(sig["série"] == c) & (sig["fenêtre"] == "val")]["classe"].iloc[0]
        cls_full = sig[(sig["série"] == c) & (sig["fenêtre"] == "full")]["classe"].iloc[0]
        shadow = bool(last252["R0"].any() or last252["R2"].any())
        revue = bool(last120["R3"].any() or cls_val == "négative")
        status = "shadow" if shadow else ("revue" if revue else "actif")
        reasons = []
        for r in ("R0", "R2"):
            if last252[r].any():
                d = last252.index[last252[r]]
                reasons.append(f"{r} du {d.min().date()} au {d.max().date()}")
        if last120["R3"].any():
            d = last120.index[last120["R3"]]
            reasons.append(f"R3 du {d.min().date()} au {d.max().date()}")
        if cls_val == "négative":
            reasons.append("classe « négative » sur la validation réutilisée")
        rows.append({"sleeve": c, "instrument": INSTRUMENT[c], "données jusqu'au": str(x.index.max().date()),
                     "séances": len(x), "R0 en alarme au 2026-10-02": bool(f["R0"].iloc[-1]),
                     "R126 final": float(f["r126"].iloc[-1]), "seuil R0": th[c]["R0"],
                     "R120 final": float(f["r120"].iloc[-1]), "seuil R3": th[c]["R3_pnl120_p05"],
                     "maxDD 250 final": float(f["dd250"].iloc[-1]), "seuil R2": th[c]["R2_dd_p01_250"],
                     "R0 déclenchée (dates)": f"{len(fired['R0'])} séances" + (f", {fired['R0'][0].date()} → {fired['R0'][-1].date()}" if fired["R0"] else ""),
                     "R2 déclenchée": len(fired["R2"]), "R1 déclenchée": len(fired["R1"]),
                     "R3 déclenchée": len(fired["R3"]),
                     "classe total": cls_full, "classe validation": cls_val,
                     "statut règle": status, "statut actuel": CURRENT[c], "motifs": "; ".join(reasons) or "—",
                     "cohérent": status == CURRENT[c] or (status == "revue" and CURRENT[c] == "actif"),
                     # Corrected proposal of 2026-11-15 (phase 5, point 8): R2 alone -> shadow; R0 and R3 -> alert.
                     "statut proposition corrigée": ("shadow" if last252["R2"].any() else
                                                     "revue" if (revue or last120["R0"].any()) else "actif")})
    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "06_status.csv", index=False)

    # S04: the leave-one-out that motivated the decision is on the reused window.
    loo = pd.read_csv(OUT / "01_section5_loo.csv")
    def loo_row(variant, book, removed):
        return loo[(loo["variant"] == variant) & (loo["book"].astype(str) == book) & (loo["removed"] == removed)].iloc[0]
    b9 = json.loads((OUT / "01_calendar.json").read_text(encoding="utf-8"))["books"]
    s04 = {"lock_loo_window": "2024-01-02 → 2026-03-19 (validation réutilisée)",
           "book9_sharpe_published": b9["A_lock_publie__9"]["sharpe"],
           "book_without_s04_sharpe_published": float(loo_row("A_lock_publie", "9", "S04")["sharpe"]),
           "book9_sharpe_corrected": b9["C_S01ref_union__9"]["sharpe"],
           "book_without_s04_sharpe_corrected": float(loo_row("C_S01ref_union", "9", "S04")["sharpe"]),
           "s04_val_pnl": float(sig[(sig["série"] == "S04") & (sig["fenêtre"] == "val")]["pnl"].iloc[0]),
           "s09_val_pnl": float(sig[(sig["série"] == "S09") & (sig["fenêtre"] == "val")]["pnl"].iloc[0]),
           "s09_val_sharpe": float(sig[(sig["série"] == "S09") & (sig["fenêtre"] == "val")]["sharpe"].iloc[0]),
           "s04_replay_pnl": float(data["S04"][(data["S04"].index >= "2026-03-20")].sum()),
           "s01_oneshot_pnl": o["pnl_usd"], "s01_oneshot_verdict": o["verdict"]}
    write_json(OUT / "06_status.json", {"rows": rows, "s04_s09": s04})
    render(tab, s04, o)
    return s04


def render(tab: pd.DataFrame, s04: dict, o: dict) -> None:
    t = ["| Sleeve | Données jusqu'au | R0 : 126 s. final / seuil | R3 : 120 s. final / seuil | maxDD 250 s. final / seuil R2 | "
         "Séances en alarme R0 · R1 · R2 · R3 | Classe total / validation | **Statut règle** | Statut actuel | Motifs |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for r in tab.itertuples(index=False):
        d = r._asdict() if hasattr(r, "_asdict") else {}
        t.append(f"| {r[0]} | {r[2]} | {fr(r[5])} / {fr(r[6])} | {fr(r[7])} / {fr(r[8])} | {fr(r[9])} / {fr(r[10])} | "
                 f"{r[11].split(' ')[0]} · {r[13]} · {r[12]} · {r[14]} | {r[15]} / {r[16]} | **{r[17]}** | {r[18]} | {r[19]} |")
    incoh = tab[~tab["cohérent"]]
    lines = [
        "# Phase 6 — Cohérence des statuts S04 / S09",
        "",
        "*Mêmes règles pour les 9 sleeves, avec les seules données disponibles au 2026-11-01. Script : "
        "[phase6_status.py](phase6_status.py). Table : [06_status.csv](outputs/06_status.csv). Rien n'est changé en production.*",
        "",
        "## Données et règle appliquée",
        "",
        "- **S01** : historique gelé ATR 18 jusqu'au 2026-03-19, plus la fenêtre du one-shot (O), du 2026-03-23 au 2026-10-02.",
        "- **S02 à S06** : replay forward des votes gelés jusqu'au 2026-10-02 (identique à l'historique gelé sur le recouvrement, vérifié).",
        "- **S07 à S09** : historique gelé jusqu'au 2026-10-02.",
        "",
        "Règle proposée en phase 5, appliquée en glissant sur toute la série :",
        "",
        "- **shadow** si R0 ou R2 s'est déclenchée dans les 252 dernières séances, puisqu'aucune réintégration n'est "
        "encore possible ;",
        "- **revue** si R3 s'est déclenchée dans les 120 dernières séances, ou si la classe de la phase 2 sur la validation "
        "réutilisée est « négative » ;",
        "- **actif** sinon.",
        "",
        *t,
        "",
        "## Incohérences avec les statuts actuels",
        "",
    ]
    if len(incoh):
        for r in incoh.itertuples(index=False):
            lines.append(f"- **{r[0]}** : la règle donne « {r[17]} », le statut actuel est « {r[18]} ». Motifs : {r[19]}.")
    else:
        lines.append("- Aucune.")
    rev = tab[(tab["statut règle"] == "revue") & (tab["statut actuel"] == "actif")]
    for r in rev.itertuples(index=False):
        lines.append(f"- **{r[0]}** est active, alors que la règle la placerait « en revue » : {r[19]}.")
    corr = tab[["sleeve", "statut règle", "statut proposition corrigée", "statut actuel"]]
    lines += [
        "",
        "## Avec la proposition corrigée du 2026-11-15",
        "",
        "La phase 5 a été corrigée : faire passer en shadow sur « R0 ou R2 » coupe à tort 21 % des sleeves saines, contre "
        "7 % pour R2 seule. La proposition corrigée réserve le shadow à R2 ; R0 et R3 deviennent des alertes « revue » "
        "(déclenchées dans les 120 dernières séances).",
        "",
        "| Sleeve | Règle initiale (R0 ou R2 → shadow) | Proposition corrigée (R2 → shadow) | Statut actuel |",
        "|---|---|---|---|",
        *[f"| {r[0]} | {r[1]} | **{r[2]}** | {r[3]} |" for r in corr.itertuples(index=False)],
        "",
        "Avec la proposition corrigée, aucune sleeve active ne passe en shadow. S04 y reste, puisque R2 s'est déclenchée "
        "pour elle. S05 passe « en revue » au lieu de « shadow ».",
    ]
    lines += [
        "",
        "## S04 et S09 : deux poids, deux mesures",
        "",
        f"- **La décision du 03/11 sur S04 s'appuie sur un leave-one-out calculé sur la fenêtre réutilisée** "
        f"{s04['lock_loo_window']}. Le book à 9 sleeves y passe d'un Sharpe de {fr(s04['book9_sharpe_published'], 3)} à "
        f"{fr(s04['book_without_s04_sharpe_published'], 3)} sans S04, chiffres du lock publié. Après correction (S01 ATR 18, "
        f"calendrier union), il passe de {fr(s04['book9_sharpe_corrected'], 3)} à {fr(s04['book_without_s04_sharpe_corrected'], 3)}. "
        f"Sur cette même fenêtre, S04 gagnait pourtant {fr(s04['s04_val_pnl'])} USD : le leave-one-out mesure une baisse "
        "de risque, pas une perte.",
        f"- Le déclencheur réel a été le replay du 2026-03-20 au 2026-10-02, où S04 perd {fr(s04['s04_replay_pnl'])} USD. "
        "Cette fenêtre a ensuite été déclarée contaminée : la décision ne peut plus être validée que par le forward.",
        f"- **S09** a une validation réutilisée nulle ({fr(s09 := s04['s09_val_pnl'])} USD, Sharpe {fr(s04['s09_val_sharpe'], 3)}). "
        "R1 l'aurait signalée dès le 2025-08-13 (phase 5), et elle reste active. Avec la règle uniforme, S09 passe au "
        "moins « en revue ».",
        "- **Asymétrie de données.** Les seuils R0 sont le minimum de leur propre historique. Pour S07 à S09, cet historique "
        "va jusqu'au 2026-10-02 : R0 ne peut donc pas s'être déclenchée sur leurs données. Pour S01 à S06, elle peut se "
        "déclencher sur la fenêtre 2026-03-20 → 2026-10-02, absente de leur calibrage. À règle égale, les sleeves MBT ne "
        "peuvent pas être prises en défaut par R0.",
        f"- **S01** : le one-shot (O) est COHÉRENT ({fr(o['pnl_usd'])} USD sur 137 séances, seuil {fr(o['preread']['threshold_pnl_p05'])}). "
        "Mais il ne contient que 11 trades : la sleeve s'assèche (phase 4).",
        "",
        "Rien n'est modifié. Ces statuts sont une lecture de cohérence, à décider par vous.",
    ]
    (AUDIT / "06_coherence_statuts.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    print(main())
