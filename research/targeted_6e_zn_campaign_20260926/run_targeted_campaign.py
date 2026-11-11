"""Run the locked six-framework 6E/ZN hypothesis campaign.

The configuration and event calendar must have been hashed before this module
is run.  No parameter selection is performed: every canonical rule and its two
local perturbations are reported.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CFG_PATH = HERE / "campaign_config.json"
CAL_PATH = HERE / "event_calendar.csv"
MANIFEST_PATH = HERE / "locked_input_manifest.json"
NY = ZoneInfo("America/New_York")
BERLIN = ZoneInfo("Europe/Berlin")
LONDON = ZoneInfo("Europe/London")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def local_stamp(day: date, clock: str, zone: ZoneInfo = NY) -> pd.Timestamp:
    return pd.Timestamp(datetime.combine(day, time.fromisoformat(clock.split()[0]), zone)).tz_convert(NY)


def load_data(spec: dict, start: str, end: str) -> pd.DataFrame:
    frame = pd.read_parquet(ROOT / spec["data_path"], columns=["open", "high", "low", "close", "volume"])
    idx = pd.DatetimeIndex(frame.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    frame.index = idx.tz_convert(NY)
    frame = frame.sort_index().loc[start: end + " 23:59"]
    frame["local_date"] = frame.index.date
    frame["trade_date"] = frame["local_date"]
    after = frame.index.time >= time(18, 0)
    frame.loc[after, "trade_date"] = [d + timedelta(days=1) for d in frame.loc[after, "local_date"]]
    return frame


def execution_price(raw: float, side: int, is_entry: bool, tick: float, slip: float) -> float:
    # Long entry/short exit pay up; short entry/long exit sell down.
    sign = side if is_entry else -side
    return raw + sign * tick * slip


def make_trade(asset: str, framework: str, variant: str, leg: str, side: int,
               entry_ts: pd.Timestamp, exit_ts: pd.Timestamp, data: pd.DataFrame,
               spec: dict, slip: float = 1.0, exit_raw: float | None = None) -> dict | None:
    if entry_ts not in data.index or exit_ts not in data.index or exit_ts <= entry_ts:
        return None
    eraw = float(data.at[entry_ts, "open"])
    xraw = float(data.at[exit_ts, "open"]) if exit_raw is None else float(exit_raw)
    ep = execution_price(eraw, side, True, spec["tick_size"], slip)
    xp = execution_price(xraw, side, False, spec["tick_size"], slip)
    gross = side * (xraw - eraw) * spec["point_value_usd"]
    net = side * (xp - ep) * spec["point_value_usd"] - 2 * spec["commission_per_side_usd"]
    return {"asset": asset, "framework": framework, "variant": variant, "leg": leg,
            "trade_date": str(data.at[entry_ts, "trade_date"]), "entry_ts": entry_ts,
            "exit_ts": exit_ts, "side": side, "entry_raw": eraw, "exit_raw": xraw,
            "gross_pnl": gross, "net_pnl": net, "slippage_ticks": slip}


def exact_or_prior_close(data: pd.DataFrame, ts: pd.Timestamp) -> tuple[pd.Timestamp, float] | None:
    if ts in data.index:
        return ts, float(data.at[ts, "open"])
    loc = data.index.searchsorted(ts) - 1
    if loc < 0 or data.index[loc].date() != ts.date():
        return None
    return data.index[loc], float(data.iloc[loc]["close"])


def time_frameworks(asset: str, data: pd.DataFrame, cfg: dict) -> list[dict]:
    spec = cfg["instruments"][asset]
    out: list[dict] = []
    days = sorted({d for d in data.local_date if d.weekday() < 5})
    if asset == "6E":
        f = cfg["frameworks"]["eur_benchmark_fix_reversal"]
        for v in f["variants"]:
            b = timedelta(minutes=v["inward_buffer_minutes"])
            for d in days:
                eu_fix = local_stamp(d, "14:15:00" if d < date(2016, 7, 1) else "14:00:00", BERLIN)
                windows = [("europe_pre_fix_short", -1, local_stamp(d, "08:00:00", BERLIN)+b, eu_fix-b),
                           ("london_fix_to_us_close_long", 1, local_stamp(d, "16:00:00", LONDON)+b, local_stamp(d, "17:00:00")-b)]
                for leg, side, a, z in windows:
                    ex = exact_or_prior_close(data, z)
                    if ex:
                        tr = make_trade(asset, "eur_benchmark_fix_reversal", v["variant"], leg, side, a, ex[0], data, spec, exit_raw=ex[1])
                        if tr: out.append(tr)
        f = cfg["frameworks"]["regional_working_hours_fx_flow"]
        for v in f["variants"]:
            b = timedelta(minutes=v["inward_buffer_minutes"])
            for d in days:
                windows = [("europe_only_short", -1, local_stamp(d,"08:00:00",BERLIN)+b, local_stamp(d,"08:00:00")-b),
                           ("us_only_long", 1, local_stamp(d,"17:00:00",BERLIN)+b, local_stamp(d,"17:00:00")-b)]
                for leg, side, a, z in windows:
                    ex = exact_or_prior_close(data, z)
                    if ex:
                        tr = make_trade(asset, "regional_working_hours_fx_flow", v["variant"], leg, side, a, ex[0], data, spec, exit_raw=ex[1])
                        if tr: out.append(tr)
    else:
        f = cfg["frameworks"]["treasury_day_night_segmentation"]
        for v in f["variants"]:
            for d in days:
                a, z = local_stamp(d, v["day_entry"]), local_stamp(d, f["day_exit"])
                tr = make_trade(asset, "treasury_day_night_segmentation", v["variant"], "us_day_long", 1, a, z, data, spec)
                if tr: out.append(tr)
    return out


def auction_framework(data: pd.DataFrame, calendar: pd.DataFrame, cfg: dict) -> list[dict]:
    spec, f, out = cfg["instruments"]["ZN"], cfg["frameworks"]["ten_year_auction_concession_reversal"], []
    days = [date.fromisoformat(x) for x in calendar.loc[calendar.event_type.eq("10Y_TREASURY_AUCTION"), "event_date"]]
    for v in f["variants"]:
        delta = timedelta(minutes=v["half_window_minutes"])
        for d in days:
            mid = local_stamp(d, "13:00:00")
            for leg, side, a, z in [("pre_auction_concession_short",-1,mid-delta,mid), ("post_auction_reversal_long",1,mid,mid+delta)]:
                tr = make_trade("ZN", "ten_year_auction_concession_reversal", v["variant"], leg, side, a, z, data, spec)
                if tr: out.append(tr)
    return out


def window_volume_by_day(data: pd.DataFrame, clock: str, minutes: int, offset_minutes: int = 0) -> pd.Series:
    vals = {}
    for d, g in data.groupby("local_date", sort=True):
        a = local_stamp(d, clock) + timedelta(minutes=offset_minutes)
        z = a + timedelta(minutes=minutes)
        vals[d] = float(g.loc[(g.index >= a) & (g.index < z), "volume"].sum())
    return pd.Series(vals).sort_index()


def invalidation_exit(data: pd.DataFrame, entry: pd.Timestamp, deadline: pd.Timestamp,
                      low: float, high: float) -> pd.Timestamp:
    scan = data.loc[(data.index >= entry) & (data.index < deadline)]
    inside = scan.index[(scan.close >= low) & (scan.close <= high)]
    if len(inside):
        nxt = inside[0] + pd.Timedelta(minutes=1)
        if nxt in data.index and nxt < deadline:
            return nxt
    return deadline


def macro_framework(asset: str, framework: str, data: pd.DataFrame,
                    calendar: pd.DataFrame, cfg: dict) -> list[dict]:
    f, spec, out = cfg["frameworks"][framework], cfg["instruments"][asset], []
    variants = f["variants"]
    if framework == "scheduled_macro_acceptance":
        strata = {"NFP_CPI_0830": {"types": f["event_types"], "time": "08:30:00", "time_exit": "09:30:00", "canonical_impulse_minutes": 5}}
    else:
        strata = f["event_strata"]
    for stratum, ss in strata.items():
        events = calendar[calendar.event_type.isin(ss["types"])]
        for v in variants:
            if framework == "scheduled_macro_acceptance":
                n = int(v["impulse_minutes"]); second = False
            else:
                n = max(1, int(round(ss["canonical_impulse_minutes"] * v["window_multiplier"]))); second = True
            vol_by_day = window_volume_by_day(data, ss["time"], n, n if second else 0)
            for row in events.itertuples():
                d = date.fromisoformat(row.event_date); event = local_stamp(d, ss["time"])
                pre = data.loc[(data.index >= event-pd.Timedelta(minutes=f["pre_release_range_minutes"])) & (data.index < event)]
                first = data.loc[(data.index >= event) & (data.index < event+pd.Timedelta(minutes=n))]
                check_start = event + pd.Timedelta(minutes=n) if second else event
                check = data.loc[(data.index >= check_start) & (data.index < check_start+pd.Timedelta(minutes=n))]
                if len(pre) != f["pre_release_range_minutes"] or len(first) != n or len(check) != n: continue
                lo, hi = float(pre.low.min()), float(pre.high.max())
                initial = 1 if first.close.iloc[-1] > hi else (-1 if first.close.iloc[-1] < lo else 0)
                confirm = check.close.iloc[-1] > hi if initial == 1 else check.close.iloc[-1] < lo
                hist = vol_by_day.loc[vol_by_day.index < d].tail(f["participation_reference_sessions"])
                if initial == 0 or not confirm or len(hist) < f["minimum_participation_history_sessions"] or float(check.volume.sum()) <= float(hist.median()): continue
                entry = check.index[-1] + pd.Timedelta(minutes=1)
                deadline = local_stamp(d, ss["time_exit"])
                exit_ts = invalidation_exit(data, entry, deadline, lo, hi)
                tr = make_trade(asset, framework, v["variant"], stratum, initial, entry, exit_ts, data, spec)
                if tr: out.append(tr)
    return out


def daily_curve(trades: pd.DataFrame, start: str, end: str) -> pd.Series:
    idx = pd.bdate_range(start, end)
    if trades.empty: return pd.Series(0.0, index=idx)
    s = trades.groupby(pd.to_datetime(trades.trade_date))["net_pnl"].sum()
    return s.reindex(idx, fill_value=0.0)


def metrics(trades: pd.DataFrame, start: str, end: str, capital: float) -> dict:
    t = trades[(trades.trade_date >= start) & (trades.trade_date <= end)]
    d = daily_curve(t, start, end); ret = d / capital
    sd = ret.std(ddof=1)
    sharpe = ret.mean()/sd*math.sqrt(252) if sd and np.isfinite(sd) else np.nan
    eq = d.cumsum(); dd = eq-eq.cummax()
    wins, losses = t.loc[t.net_pnl>0,"net_pnl"].sum(), -t.loc[t.net_pnl<0,"net_pnl"].sum()
    return {"trades":len(t), "pnl":t.net_pnl.sum(), "sharpe":sharpe,
            "profit_factor": wins/losses if losses else np.nan, "max_drawdown":dd.min() if len(dd) else 0.0}


def execute() -> None:
    cfg = json.loads(CFG_PATH.read_text()); manifest = json.loads(MANIFEST_PATH.read_text())
    if manifest["performance_computed"] or digest(CFG_PATH) != manifest["campaign_config_sha256"] or digest(CAL_PATH) != manifest["event_calendar_sha256"]:
        raise RuntimeError("Locked inputs changed or campaign was already run")
    cal = pd.read_csv(CAL_PATH, dtype=str)
    full_start, full_end = cfg["periods"]["full"]
    d6 = load_data(cfg["instruments"]["6E"], full_start, full_end)
    dz = load_data(cfg["instruments"]["ZN"], full_start, full_end)
    rows = time_frameworks("6E", d6, cfg) + time_frameworks("ZN", dz, cfg)
    rows += auction_framework(dz, cal, cfg)
    rows += macro_framework("6E", "scheduled_macro_acceptance", d6, cal, cfg)
    rows += macro_framework("ZN", "scheduled_treasury_information_assimilation", dz, cal, cfg)
    trades = pd.DataFrame(rows).sort_values(["framework","variant","entry_ts"])
    trades.to_csv(HERE/"trades.csv", index=False)
    period_rows=[]; annual_rows=[]; slip_rows=[]; roll_rows=[]; leg_rows=[]
    capital=cfg["capital"]["initial_capital_usd"]
    for (fw,var), t in trades.groupby(["framework","variant"]):
        for name,(a,z) in cfg["periods"].items():
            period_rows.append({"framework":fw,"variant":var,"period":name,**metrics(t,a,z,capital)})
        for year in range(2015,2027):
            annual_rows.append({"framework":fw,"variant":var,"year":year,**metrics(t,f"{year}-01-01",f"{year}-12-31",capital)})
        for leg, lt in t.groupby("leg"):
            leg_rows.append({"framework":fw,"variant":var,"leg":leg,**metrics(lt,full_start,full_end,capital)})
        d=daily_curve(t,full_start,full_end)/capital
        rs=d.rolling(252,min_periods=126).mean()/d.rolling(252,min_periods=126).std()*math.sqrt(252)
        roll_rows += [{"date":str(k.date()),"framework":fw,"variant":var,"rolling_sharpe":x} for k,x in rs.dropna().items()]
        for slip in cfg["diagnostics"]["slippage_ticks_per_side"]:
            ts=t.copy(); inst=cfg["instruments"][t.asset.iloc[0]]
            ts["net_pnl"] = ts.gross_pnl - 2*inst["commission_per_side_usd"] - 2*slip*inst["tick_value_usd"]
            slip_rows.append({"framework":fw,"variant":var,"slippage_ticks_per_side":slip,**metrics(ts,full_start,full_end,capital),
                              "oos_pnl":metrics(ts,*cfg["periods"]["oos"],capital)["pnl"]})
    periods=pd.DataFrame(period_rows); years=pd.DataFrame(annual_rows)
    periods.to_csv(HERE/"period_metrics.csv",index=False); years.to_csv(HERE/"yearly_metrics.csv",index=False)
    pd.DataFrame(slip_rows).to_csv(HERE/"slippage_sensitivity.csv",index=False)
    pd.DataFrame(roll_rows).to_csv(HERE/"rolling_sharpe.csv",index=False)
    pd.DataFrame(leg_rows).to_csv(HERE/"economic_leg_metrics.csv",index=False)
    # Mechanical evidence gates, evaluated only on canonical specifications.
    gates=[]
    for fw in cfg["frameworks"]:
        p=periods[(periods.framework==fw)&(periods.variant=="canonical")].set_index("period")
        if p.empty: continue
        canon=trades[(trades.framework==fw)&(trades.variant=="canonical")]
        event=fw in {"scheduled_macro_acceptance","ten_year_auction_concession_reversal","scheduled_treasury_information_assimilation"}
        mintr=cfg["evidence_gates"]["minimum_oos_trades_event_framework" if event else "minimum_oos_trades_daily_framework"]
        positive_year_fraction=float((years[(years.framework==fw)&(years.variant=="canonical")].pnl>0).mean())
        variants=periods[(periods.framework==fw)&(periods.period=="oos")]
        pert=float((variants.pnl>0).mean())
        s2=pd.DataFrame(slip_rows); s2=s2[(s2.framework==fw)&(s2.variant=="canonical")&(s2.slippage_ticks_per_side==2)]
        passed=bool(p.loc["oos","trades"]>=mintr and p.loc["is","pnl"]>0 and p.loc["oos","pnl"]>0 and
                    p.loc["is","profit_factor"]>1 and p.loc["oos","profit_factor"]>1 and positive_year_fraction>=.5 and pert>=2/3 and len(s2) and s2.iloc[0].oos_pnl>0)
        gates.append({"framework":fw,"canonical_trades":len(canon),"positive_year_fraction":positive_year_fraction,
                      "positive_oos_variant_fraction":pert,"standalone_survivor":passed})
    gates=pd.DataFrame(gates); gates.to_csv(HERE/"evidence_gates.csv",index=False)
    # Portfolio work is intentionally conditional on survival.
    (HERE/"portfolio_test_status.json").write_text(json.dumps({"status":"NOT_RUN_NO_STANDALONE_SURVIVOR" if not gates.standalone_survivor.any() else "PENDING_SURVIVOR_INTEGRATION",
        "survivors":gates.loc[gates.standalone_survivor,"framework"].tolist()},indent=2)+"\n")
    manifest.update({"performance_computed":True,"status":"CAMPAIGN_COMPLETE","trade_count":len(trades),"standalone_survivors":gates.loc[gates.standalone_survivor,"framework"].tolist()})
    MANIFEST_PATH.write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    print(periods[(periods.variant=="canonical")&periods.period.isin(["is","oos","full"])].to_string(index=False))
    print("\nEvidence gates\n",gates.to_string(index=False))

if __name__ == "__main__": execute()
