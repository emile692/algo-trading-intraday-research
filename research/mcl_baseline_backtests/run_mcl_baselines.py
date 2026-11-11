"""Run the 16 frozen mechanisms on MCL, once each, with no optimization."""
from __future__ import annotations

import copy
import hashlib
import json
from datetime import date, time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from research.baseline_backtests.baseline_engine import (
    NY, PreparedData, _resample_rth, _sha256, _timestamp, _true_range,
    _validate_minute_data, _wilder_rma, build_buy_and_hold, compute_metrics,
    load_frozen_specs, run_baseline_backtest, serialize_checks,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CONFIG = OUT / "mcl_baseline_config.json"
FROZEN = ROOT / "strategy_registry/psf_mnq/frozen_baselines"
MANIFEST = ROOT / "strategy_registry/psf_mnq/frozen_baselines_manifest.csv"


def canonical_hash(payload: dict[str, Any]) -> str:
    clean = {k: v for k, v in payload.items() if not k.startswith("_")}
    return hashlib.sha256(json.dumps(clean, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def adapt(source: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    spec = copy.deepcopy(source); inst, sess = cfg["instrument"], cfg["session"]
    execution = spec["canonical_parameters"]["execution"]
    execution.update({
        "tick_size": inst["tick_size"], "tick_value_usd": inst["tick_value_usd"],
        "point_value_usd": inst["point_value_usd"],
        "commission_per_side_usd": inst["commission_per_side_usd"],
        "slippage_ticks_per_side": inst["slippage_ticks_per_side"],
        "session_timezone": sess["timezone"], "rth_start": sess["rth_start"],
        "cash_close": sess["cash_close"], "last_entry_time": sess["last_entry_time"],
        "forced_flatten_time": sess["forced_flatten_time"],
        "early_close_calendar": sess["policy"],
    })
    risk = spec["canonical_parameters"]["risk_management"]
    risk["fixed_quantity"] = 1; risk["hard_flatten_time"] = sess["forced_flatten_time"]
    rel, mech = cfg["relative_time_adaptations"], spec["mechanism_id"]
    if mech == "gap_continuation": execution["last_entry_time"] = rel["gap_continuation_last_entry"]
    if mech == "gap_fade":
        execution["last_entry_time"] = rel["gap_fade_last_entry"]
        risk["midday_time_exit"] = rel["gap_fade_midday_time_exit"]
    if mech == "session_range_breakout":
        spec["canonical_parameters"]["structural"]["opening_range_start"] = rel["session_range_breakout_opening_range_start"]
    spec["_effective_mcl_spec_sha256"] = canonical_hash(spec)
    return spec


def adapt_benchmark(source: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    spec = copy.deepcopy(source)
    spec["mechanism_id"] = spec["strategy_id"] = "MCL_BUY_AND_HOLD"
    spec["canonical_parameters"]["structural"]["cash_close"] = cfg["session"]["cash_close"]
    spec["_effective_mcl_spec_sha256"] = canonical_hash(spec)
    return spec


def prepare(path: Path, start: date, end: date, cfg: dict[str, Any]) -> PreparedData:
    raw = pd.read_parquet(path, columns=["open", "high", "low", "close", "volume"])
    _validate_minute_data(raw)
    raw = raw.copy(); raw.index = raw.index.tz_convert(NY); raw.index.name = "timestamp"
    raw = raw[(raw.index.date >= start) & (raw.index.date <= end)]
    raw["session_date"] = raw.index.date
    excluded = {date.fromisoformat(x) for x in cfg["session"]["explicit_excluded_sessions"]}
    open_t, close_t = time(9, 0), time(14, 30)
    sessions, warnings = [], []
    for day, group in raw.groupby("session_date", sort=True):
        if day.weekday() >= 5 or day in excluded: continue
        a, z = _timestamp(day, open_t), _timestamp(day, close_t)
        core = group[(group.index >= a) & (group.index < z)]
        complete = (len(core) >= cfg["session"]["minimum_core_bars"] and not core.empty
                    and core.index[0] == a and core.index[-1] == z-pd.Timedelta(minutes=1))
        if not complete:
            warnings.append(f"excluded_incomplete_core:{day}:{len(core)}")
            continue
        sessions.append({"session_date":day, "rth_start":a, "cash_close":z,
                         "calendar_last_entry":_timestamp(day,time(14,25)),
                         "hard_flatten":_timestamp(day,time(16,8)),
                         "early_close":False,"early_close_reason":"full_session"})
    sf = pd.DataFrame(sessions)
    valid = set(sf.session_date)
    minute = raw[raw.session_date.isin(valid)].copy()
    ends = sf.set_index("session_date").hard_flatten.to_dict()
    starts = sf.set_index("session_date").rth_start.to_dict()
    minute = minute.loc[[starts[d] <= ts <= ends[d] for ts,d in zip(minute.index,minute.session_date)],
                        ["open","high","low","close","volume","session_date"]]
    rows=[]
    by_day={d:g for d,g in minute.groupby("session_date",sort=False)}
    for s in sf.itertuples(index=False):
        g=by_day[s.session_date]; r=g[(g.index>=s.rth_start)&(g.index<s.cash_close)]
        rows.append({"session_date":s.session_date,"open":r.open.iloc[0],"high":r.high.max(),
                     "low":r.low.min(),"close":r.close.iloc[-1],"close_timestamp":r.index[-1],"volume":r.volume.sum()})
    daily=pd.DataFrame(rows)
    daily["true_range"]=_true_range(daily); daily["atr14"]=_wilder_rma(daily.true_range,14)
    daily["prior_close"]=daily.close.shift(1); daily["prior_atr14"]=daily.atr14.shift(1)
    bars={n:_resample_rth(minute,sf,n) for n in (5,15,60)}
    benchmark=daily[["session_date","close","close_timestamp"]].copy()
    benchmark["cash_close"]=[_timestamp(d,close_t) for d in benchmark.session_date]
    benchmark["early_close"]=False; benchmark["early_close_reason"]="full_session"
    return PreparedData(minute,sf,bars,daily,benchmark,path,_sha256(path),warnings)


def execute() -> None:
    cfg=json.loads(CONFIG.read_text()); path=ROOT/cfg["data"]["path"]
    if _sha256(path)!=cfg["data"]["sha256"]: raise RuntimeError("MCL data hash mismatch")
    source_specs, source_benchmark, manifest=load_frozen_specs(FROZEN,MANIFEST)
    specs=[adapt(s,cfg) for s in source_specs]; benchspec=adapt_benchmark(source_benchmark,cfg)
    before={str(p):_sha256(p) for p in [*sorted(FROZEN.glob("*.json")),MANIFEST]}
    metrics=[]; checks=[]; audit=[]
    for period,(a,z) in cfg["evaluation_windows"].items():
        print(f"Preparing {period} {a} -> {z}",flush=True)
        data=prepare(path,date.fromisoformat(a),date.fromisoformat(z),cfg)
        bench=build_buy_and_hold(benchspec,data)
        pdir=OUT/period.lower(); tdir=pdir/"trades"; tdir.mkdir(parents=True,exist_ok=True)
        for i,spec in enumerate(specs,1):
            print(f"[{period} {i:02d}/16] {spec['mechanism_id']}",flush=True)
            run=run_baseline_backtest(spec,data); row=compute_metrics(spec,run,bench.daily)
            row.update(period=period,effective_mcl_spec_sha256=spec["_effective_mcl_spec_sha256"])
            metrics.append(row); checks.append({"period":period,**serialize_checks(spec["mechanism_id"],run.checks)})
            run.trades.to_csv(tdir/f"{spec['mechanism_id']}.csv",index=False)
        audit.append({"period":period,"sessions":len(data.sessions),"minute_rows":len(data.minute),
                      "excluded_or_warning_count":len(data.data_warnings),"warnings":";".join(data.data_warnings)})
    if before!={str(p):_sha256(p) for p in [*sorted(FROZEN.glob("*.json")),MANIFEST]}:
        raise RuntimeError("Frozen specifications changed")
    result=pd.DataFrame(metrics); result.to_csv(OUT/"mcl_baseline_results.csv",index=False)
    pd.DataFrame(checks).to_csv(OUT/"implementation_checks.csv",index=False)
    pd.DataFrame(audit).to_csv(OUT/"data_audit.csv",index=False)
    meta={"status":"COMPLETE","optimization":False,"grid_search":False,"mechanisms":16,
          "data_sha256":_sha256(path),"databento_cost_usd":cfg["data"]["databento_cost_usd"],
          "source_frozen_files_unchanged":True,"config_sha256":_sha256(CONFIG)}
    (OUT/"run_metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(result[["period","mechanism_id","trade_count","net_pnl_usd","profit_factor","sharpe_annualized","max_drawdown_usd","backtest_valid"]].to_string(index=False))


if __name__ == "__main__": execute()
