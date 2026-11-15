"""Phase 4b (O) — S01 ATR 18 one-shot on 2026-03-20 -> 2026-10-02, exactly as preregistered.

Order of operations (preregistration.md, section 1 and 5):
  0. refuse to run twice; verify the preregistration and criteria hashes;
  1. H and the threshold are fixed from the timestamps of the long MNQ file only, and written to disk
     (outputs/04b_preread.json) before any price is loaded;
  2. S01 is replayed on the long file with the frozen IS thresholds of thresholds_is.csv (never recalibrated);
  3. technical gate: trades up to 2026-03-19 must equal the 701 frozen trades; on failure nothing about the window
     is computed or printed;
  4. window statistics and the verdict, written once.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .common import AUDIT, OUT, ROOT, fr, max_dd, sharpe, write_json
from .phase0_engine import _norm

LONG = ROOT / "data" / "processed" / "parquet" / "MNQ_c_0_1m_20190505_20261003.parquet"
THRESHOLDS = ROOT / "research" / "signal_sheet_01_atr18" / "thresholds_is.csv"
FROZEN_TRADES = ROOT / "research" / "signal_sheet_01_atr18" / "trades.csv"
BASELINE_EXT = ROOT / "research" / "nine_sleeve_paper_extension_20261013" / "daily_nine_sleeves_through_20261002.csv"
WIN = ("2026-03-20", "2026-10-02")
RESULT = OUT / "04b_result.json"
PREREAD = OUT / "04b_preread.json"


def window_sessions() -> list[pd.Timestamp]:
    """MNQ sessions of the window from the timestamp column only (no price column is read)."""
    a = pd.Timestamp(WIN[0], tz="America/New_York")
    b = pd.Timestamp(WIN[1], tz="America/New_York") + pd.Timedelta(days=1)
    col = "timestamp"
    tz = str(pq.read_schema(LONG).field(col).type.tz)
    ts = pd.read_parquet(LONG, columns=[], filters=[(col, ">=", a.tz_convert(tz)), (col, "<", b.tz_convert(tz))]).index
    ts = pd.DatetimeIndex(ts).tz_convert("America/New_York")
    minutes = ts.hour * 60 + ts.minute
    rth = ts[(minutes >= 9 * 60 + 30) & (minutes < 16 * 60) & (ts.dayofweek < 5)]
    return sorted(pd.to_datetime(pd.Series(rth.date).unique()))


def run_s01_frozen() -> pd.DataFrame:
    spec = importlib.util.spec_from_file_location("audit_rebuild_s01_oneshot", ROOT / "scripts" / "rebuild_signal_sheet_01_atr18.py")
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    th = pd.read_csv(THRESHOLDS)
    if len(th) != 18:
        raise RuntimeError("thresholds_is.csv must hold 18 models")
    entry = m.BaselineEntryConfig()
    minute = m.prepare_minute_dataset(LONG, entry, m.ATR_PERIODS)
    minute = m.attach_daily_reference(minute, m.build_daily_reference(minute))
    candidates = m.build_candidate_universe(minute, baseline_entry=entry)
    pre = m.first_pass_signal_rows(candidates, candidates["candidate_base_pass"].fillna(False).astype(bool))
    votes = {}
    for r in th.itertuples(index=False):
        votes[r.model] = pd.to_numeric(pre[f"atr_{int(r.atr_period)}"], errors="coerce").between(
            float(r.low_threshold_is), float(r.high_threshold_is), inclusive="both").fillna(False)
    v = pd.DataFrame(votes, index=pre.index)
    pre = pre.copy()
    pre["atr_vote_count"] = v.sum(axis=1).astype(int)
    final = pre.loc[pre["atr_vote_count"].ge(m.MINIMUM_VOTES)].copy()
    exp = m._build_incremental_experiments(entry, m.BaselineEnsembleConfig())[0]
    sig = m.build_signal_frame_for_backtest(minute, final)
    return m.run_exit_variant_backtest(signal_df=sig, execution_model=m.ExecutionModel(), baseline=entry, exit_cfg=exp.exit)


def main() -> dict:
    if RESULT.exists():
        raise SystemExit("04b (O) has already been read once; refusing a second run (preregistration, section 7)")
    from .run_all import check_prereg
    check_prereg()

    # 1. H and threshold, before any price is loaded.
    sessions = window_sessions()
    H = len(sessions)
    crit = pd.read_csv(OUT / "04_criteria.csv")
    row = crit[(crit["series"] == "S01") & (crit["block"] == 20) & (crit["H"] == H)].iloc[0]
    pre = {"written_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "window": list(WIN), "H": H,
           "first_session": str(sessions[0].date()), "last_session": str(sessions[-1].date()),
           "threshold_pnl_p05": float(row.pnl_p05), "context_pnl_p50": float(row.pnl_p50),
           "context_pnl_p95": float(row.pnl_p95), "context_mdd_p05": float(row.mdd_p05)}
    write_json(PREREAD, pre)

    # 2-3. Replay and gate.
    trades = run_s01_frozen()
    trades["session_date"] = pd.to_datetime(trades["session_date"])
    before = trades[trades["session_date"] <= pd.Timestamp("2026-03-19")]
    a, b = _norm(pd.read_csv(FROZEN_TRADES)), _norm(before)
    cols = ["session_date", "entry_time", "exit_time", "position", "quantity"]
    ka, kb = set(map(tuple, a[cols].astype(str).to_numpy())), set(map(tuple, b[cols].astype(str).to_numpy()))
    merged = a.merge(b, on=cols, suffixes=("_f", "_r"))
    pnl_gap = float((merged["net_pnl_usd_f"] - merged["net_pnl_usd_r"]).abs().max()) if len(merged) else float("inf")
    gate = {"frozen": len(a), "replayed_to_20260319": len(b), "only_frozen": len(ka - kb), "only_replayed": len(kb - ka),
            "max_abs_pnl_gap": pnl_gap}
    gate["passed"] = gate["only_frozen"] == 0 and gate["only_replayed"] == 0 and pnl_gap <= 0.01
    if not gate["passed"]:
        result = {"preread": pre, "gate": gate, "verdict": "NON ÉVALUÉ — gate technique échoué",
                  "read_at": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
        write_json(RESULT, result)
        render(result, None)
        return result

    # 4. Window statistics and verdict.
    win = trades[(trades["session_date"] >= pd.Timestamp(WIN[0])) & (trades["session_date"] <= pd.Timestamp(WIN[1]))].copy()
    off = set(win["session_date"]) - set(sessions)
    if off:
        raise RuntimeError(f"trades on dates outside the H sessions: {sorted(off)}")
    win.to_csv(OUT / "04b_s01_window_trades.csv", index=False)
    daily = win.groupby("session_date")["net_pnl_usd"].sum().reindex(pd.DatetimeIndex(sessions)).fillna(0.0)
    pnl = float(daily.sum())
    verdict = "ÉCHEC" if pnl < pre["threshold_pnl_p05"] else "COHÉRENT"
    base = pd.read_csv(BASELINE_EXT, parse_dates=["session_date"]).set_index("session_date")["mnq_orb_vwap_atr"]
    base = base[(base.index >= WIN[0]) & (base.index <= WIN[1])]
    result = {"preread": pre, "gate": gate, "H": H, "trades": int(len(win)), "pnl_usd": pnl,
              "sharpe": sharpe(daily) if daily.std() > 0 else None, "max_dd_usd": max_dd(daily),
              "worst_day_usd": float(daily.min()), "win_rate_trades": float((win["net_pnl_usd"] > 0).mean()) if len(win) else None,
              "verdict": verdict, "criterion": f"ÉCHEC si PnL < p5(H={H}) = {pre['threshold_pnl_p05']:.2f} USD",
              "info_baseline_fixed": {"sessions_in_file": int(len(base)), "pnl_usd": float(base.sum()),
                                      "note": "already read on 2026-11-01; reported for information only, no criterion"},
              "label": "partiellement contaminé (baseline_fixed déjà lu sur cette fenêtre)",
              "read_at": dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    write_json(RESULT, result)
    render(result, daily)
    return result


def render(r: dict, daily: pd.Series | None) -> None:
    p = r["preread"]
    g = r["gate"]
    lines = [
        "# Phase 4 — Holdout",
        "",
        "*Exécuté une seule fois par [phase4b_s01_oneshot.py](phase4b_s01_oneshot.py), selon le "
        "[pré-enregistrement validé](preregistration.md). Sorties : [04b_preread.json](outputs/04b_preread.json), "
        "écrit avant tout chargement de prix, et [04b_result.json](outputs/04b_result.json).*",
        "",
        "## (O) — S01 ATR 18, 2026-03-20 → 2026-10-02 (partiellement contaminé)",
        "",
        f"- **H** = {p['H']} séances MNQ ({p['first_session']} → {p['last_session']}), comptées sur les timestamps seuls.",
        f"- **Seuil pré-enregistré** : p5 du PnL à H = **{fr(p['threshold_pnl_p05'])} USD** (médiane {fr(p['context_pnl_p50'])}, "
        f"p95 {fr(p['context_pnl_p95'])}).",
        f"- **Gate technique** : {g['replayed_to_20260319']} trades rejoués jusqu'au 2026-03-19 pour {g['frozen']} gelés ; "
        f"{g['only_frozen']} seulement gelés, {g['only_replayed']} seulement rejoués, écart PnL max {fr(g['max_abs_pnl_gap'])} USD. "
        f"**{'Réussi' if g['passed'] else 'ÉCHOUÉ'}.**",
        "",
    ]
    if not g["passed"]:
        lines += ["**Évaluation arrêtée.** Le gate a échoué : aucune statistique de la fenêtre n'a été calculée. "
                  "L'écart est documenté ci-dessus, et aucune relance n'est faite sur cette fenêtre (pré-enregistrement, section 7)."]
    else:
        lines += [
            "| Critère (tel qu'écrit) | Valeur | Verdict |",
            "|---|---:|---|",
            f"| PnL cumulé à H < p5 ⇒ ÉCHEC | {fr(r['pnl_usd'])} USD vs {fr(p['threshold_pnl_p05'])} | **{r['verdict']}** |",
            "",
            "Statistiques reportées, sans critère :",
            "",
            f"- trades : {r['trades']} ; Sharpe (√252, {p['H']} séances) : {fr(r['sharpe'], 3) if r['sharpe'] is not None else 'n/d'} ; "
            f"maxDD : {fr(r['max_dd_usd'])} USD ; pire jour : {fr(r['worst_day_usd'])} USD ;",
            f"- pour information, `baseline_fixed` sur la même fenêtre (déjà lu le 2026-11-01) : {fr(r['info_baseline_fixed']['pnl_usd'])} USD.",
            "",
            "Ce résultat ne sert à aucun retuning. Il alimente le statut de S01 en phase 6.",
        ]
    lines += [
        "",
        "## (P) — Suivi prospectif après le 2026-11-03",
        "",
        "Non lu : les données ne sont pas sur disque. La lecture se fait avec `forward_check.py`, une fois par horizon "
        "(20, 60, 120, 250 séances). Première lecture possible vers le 2026-12-03, après téléchargement des données.",
    ]
    (AUDIT / "04_holdout.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    out = main()
    print(json.dumps({k: out[k] for k in ("verdict", "gate")}, ensure_ascii=False, default=str))
