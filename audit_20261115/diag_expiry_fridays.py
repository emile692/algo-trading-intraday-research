"""Descriptive diagnostic: the MNQ sleeves (S01-S04) on the quarterly expiry Fridays missing from MNQ.c.0 (gap 3).

The frozen file MNQ_c_0_1m_20260321_094501.parquet (input of the S01-S04 engines) has no RTH bar on 27 expiry Fridays
of 2019-06-21 -> 2025-12-19; only a few overnight bars of the expiring contract.  MNQ.v.0 bars of those sessions were
downloaded on 2026-11-15 (scripts/download_mnq_expiry_fridays.py, 0.28 USD).  Method:
  1. roll spread = median(v.0 close - c.0 close) over the common minutes of the Thursday 14:00-16:00 ET (c.0 is still the
     expiring contract on Thursday, v.0 the next one);
  2. the whole Friday session (Thursday 18:00 -> Friday 17:00 ET) of c.0 is replaced by the v.0 bars minus the spread,
     rounded to the tick, with the instrument_id/symbol of c.0 (back-adjusted continuity);
  3. the four engines are rerun unchanged on that file (S01 with the frozen IS thresholds, S02 rebuild, S03/S04 vote
     freeze), with the same windows as the frozen ledgers;
  4. trades are compared with the frozen ledgers: trades on the Fridays, and knock-on changes on the other days (the
     Friday now enters the daily features of the following sessions).
2026-03-20 and 2026-09-18 are outside the frozen windows and are not used.  No parameter changes; descriptive only.
Command: python -m audit_20261115.diag_expiry_fridays  (about 30 minutes).  Output: outputs/diag_expiry_fridays*.
"""

from __future__ import annotations

import importlib.util
import json
import sys

import numpy as np
import pandas as pd

from .common import CACHE, OUT, ROOT, load_trades, sha256, write_json
from .phase0_engine import KEY_COLS, _norm

C0 = ROOT / "data" / "processed" / "parquet" / "MNQ_c_0_1m_20260321_094501.parquet"
V0 = ROOT / "data" / "downloaded" / "MNQ_v_0_1m_expiry_fridays_20190621_20260918.parquet"
MERGED = CACHE / "MNQ_c_0_with_expiry_fridays_v0_adjusted.parquet"
LAST_FRIDAY = pd.Timestamp("2025-12-19")
TICK = 0.25


def build_merged() -> tuple[pd.DataFrame, list]:
    meta = json.loads(V0.with_suffix(".metadata.json").read_text(encoding="utf-8"))
    fridays = [pd.Timestamp(f) for f in meta["fridays"] if pd.Timestamp(f) <= LAST_FRIDAY]
    c0 = pd.read_parquet(C0)
    v0 = pd.read_parquet(V0)
    rows, pieces, drop = [], [], []
    for f in fridays:
        thu = f - pd.Timedelta(days=1)
        tz = "America/New_York"
        a, b = pd.Timestamp(f"{thu.date()} 14:00", tz=tz), pd.Timestamp(f"{thu.date()} 16:00", tz=tz)
        cc, vv = c0.loc[a:b - pd.Timedelta(minutes=1), "close"], v0.loc[a:b - pd.Timedelta(minutes=1), "close"]
        common = cc.index.intersection(vv.index)
        s0, s1 = pd.Timestamp(f"{thu.date()} 18:00", tz=tz), pd.Timestamp(f"{f.date()} 17:00", tz=tz)
        if len(common) == 0:  # e.g. 2025-06-19 (Juneteenth): last 120 common minutes before Thursday 18:00
            cc, vv = c0.loc[:s0 - pd.Timedelta(minutes=1), "close"], v0.loc[:s0 - pd.Timedelta(minutes=1), "close"]
            common = cc.index.intersection(vv.index)[-120:]
        spread = float(np.median(vv.loc[common] - cc.loc[common]))
        old = c0.loc[s0:s1 - pd.Timedelta(minutes=1)]
        new = v0.loc[s0:s1 - pd.Timedelta(minutes=1)].copy()
        ref = c0.loc[:s0].iloc[-1]
        for col in ("open", "high", "low", "close"):
            new[col] = (np.round((new[col] - spread) / TICK) * TICK).astype(float)
        new["instrument_id"] = np.uint32(ref["instrument_id"])
        new["symbol"] = ref["symbol"]
        new = new[c0.columns].astype(c0.dtypes.to_dict())
        m = new.index.hour * 60 + new.index.minute
        rows.append({"friday": str(f.date()), "thursday_common_minutes": int(len(common)), "roll_spread_points": spread,
                     "spread_pct": spread / float(cc.loc[common].median()) * 100, "c0_bars_replaced": int(len(old)),
                     "v0_bars_inserted": int(len(new)), "v0_rth_bars": int(((m >= 570) & (m < 960)).sum()),
                     "v0_instrument_ids": sorted(map(int, v0.loc[s0:s1, "instrument_id"].unique())),
                     "c0_thursday_instrument_id": int(ref["instrument_id"])})
        drop.append(old.index)
        pieces.append(new)
    to_drop = drop[0].append(drop[1:]) if len(drop) > 1 else drop[0]
    merged = pd.concat([c0.drop(index=to_drop), *pieces]).sort_index()
    if merged.index.duplicated().any():
        raise RuntimeError("duplicated timestamps after merge")
    merged.to_parquet(MERGED)
    return pd.DataFrame(rows), fridays


def _load(name: str, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def run_s01() -> pd.DataFrame:
    m = _load("diag_fri_s01", ROOT / "scripts" / "rebuild_signal_sheet_01_atr18.py")
    th = pd.read_csv(ROOT / "research" / "signal_sheet_01_atr18" / "thresholds_is.csv")
    entry = m.BaselineEntryConfig()
    minute = m.prepare_minute_dataset(MERGED, entry, m.ATR_PERIODS)
    minute = m.attach_daily_reference(minute, m.build_daily_reference(minute))
    cand = m.build_candidate_universe(minute, baseline_entry=entry)
    pre = m.first_pass_signal_rows(cand, cand["candidate_base_pass"].fillna(False).astype(bool))
    votes = {r.model: pd.to_numeric(pre[f"atr_{int(r.atr_period)}"], errors="coerce").between(
        float(r.low_threshold_is), float(r.high_threshold_is), inclusive="both").fillna(False) for r in th.itertuples(index=False)}
    final = pre.loc[pd.DataFrame(votes, index=pre.index).sum(axis=1).ge(m.MINIMUM_VOTES)].copy()
    exp = m._build_incremental_experiments(entry, m.BaselineEnsembleConfig())[0]
    sig = m.build_signal_frame_for_backtest(minute, final)
    t = m.run_exit_variant_backtest(signal_df=sig, execution_model=m.ExecutionModel(), baseline=entry, exit_cfg=exp.exit)
    return t[pd.to_datetime(t["session_date"]) <= pd.Timestamp("2026-03-20")]


def run_s02() -> pd.DataFrame:
    import rebuild_signal_sheet_02_cluster as r
    r.DATA_PATH = MERGED
    manifest = json.loads(r.MANIFEST_PATH.read_text(encoding="utf-8"))
    specs, _, _ = r.load_frozen_specs(r.FROZEN_DIR, r.FROZEN_MANIFEST)
    spec = next(s for s in specs if s["mechanism_id"] == "volatility_breakout")
    prepared = r.prepare_data(MERGED, start_date=r.IS_START, end_date=r.OOS_END)
    positions, _ = r.build_cell_positions(spec, prepared, manifest)
    result, _ = r.aggregate_vote(spec, prepared, positions, manifest)
    return result.trades


def run_cluster(key: str) -> pd.DataFrame:
    import freeze_signal_sheet_cluster_vote as f
    import run_signal_sheet_parameter_refinement as rr
    rr.MNQ_DATA = MERGED
    if hasattr(f, "MNQ_DATA"):
        f.MNQ_DATA = MERGED
    campaign = f.CAMPAIGNS[key]
    prepared, base_spec, source = f.prepare_campaign(key, campaign)
    manifest = f.build_manifest(key, base_spec, source)
    positions, _ = f.build_cell_positions(key, base_spec, prepared)
    result, _ = f.aggregate_vote(base_spec, prepared, positions, manifest)
    return f.relabel(result.trades, key)


RUNNERS = {"S01": run_s01, "S02": run_s02, "S03": lambda: run_cluster("s03"), "S04": lambda: run_cluster("s04")}


def compare(code: str, fridays: list) -> dict:
    path = CACHE / f"diag_fri_engine_{code}.csv"
    if not path.exists():
        RUNNERS[code]().to_csv(path, index=False)
    new, old = _norm(pd.read_csv(path)), _norm(load_trades(code))
    fset = {str(f.date()) for f in fridays}
    kn, ko = new[KEY_COLS].astype(str).apply(tuple, axis=1), old[KEY_COLS].astype(str).apply(tuple, axis=1)
    new_fri = new[new["session_date"].isin(fset)]
    new_other, old_other = new[~new["session_date"].isin(fset)], old[~old["session_date"].isin(fset)]
    only_new = new_other[~kn[new_other.index].isin(set(ko))]
    only_old = old_other[~ko[old_other.index].isin(set(kn))]
    new_fri.assign(sleeve=code).to_csv(OUT / f"diag_expiry_fridays_trades_{code}.csv", index=False)
    return {"frozen_trades_on_fridays": int(old["session_date"].isin(fset).sum()),
            "friday_trades": int(len(new_fri)), "friday_pnl_usd": float(new_fri["net_pnl_usd"].sum()),
            "friday_win_rate": float((new_fri["net_pnl_usd"] > 0).mean()) if len(new_fri) else None,
            "other_days_trades_added": int(len(only_new)), "other_days_trades_removed": int(len(only_old)),
            "other_days_pnl_change_usd": float(new_other["net_pnl_usd"].sum() - old_other["net_pnl_usd"].sum()),
            "other_days_changed_dates": sorted(set(only_new["session_date"]) | set(only_old["session_date"]))[:20],
            "frozen_total_usd": float(old["net_pnl_usd"].sum()), "new_total_usd": float(new["net_pnl_usd"].sum())}


def main() -> dict:
    spreads, fridays = build_merged()
    spreads.to_csv(OUT / "diag_expiry_fridays_spreads.csv", index=False)
    res = {"fridays": [str(f.date()) for f in fridays], "merged_file_sha256": sha256(MERGED), "v0_sha256": sha256(V0),
           "spread_points_median": float(spreads["roll_spread_points"].median()),
           "sleeves": {}}
    for code in RUNNERS:
        res["sleeves"][code] = compare(code, fridays)
        print(code, json.dumps(res["sleeves"][code]), flush=True)
    write_json(OUT / "diag_expiry_fridays.json", res)
    return res


if __name__ == "__main__":
    main()
