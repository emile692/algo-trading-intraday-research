# 6E London-fix reversal — Phase 1/2

The one-minute 6E volume-ranked continuous series covers 1 January 2015 through 22 September 2026. Each event is created at 16:00 in `Europe/London` and only then converted to UTC, so UK/US DST mismatch weeks are handled without hard-coded offsets.

## Sign convention

The paper defines an increase as foreign-currency appreciation versus USD. 6E is quoted USD per EUR and therefore has the same direction as EURUSD. The documented mechanism implies 6E should fall before the fix (long USD = short 6E) and rise after it (short USD = long 6E).

## Evidence

- 2,959 events with at least 90% path coverage; 2,262 IS and 697 OOS.
- Pre-fix T−180 to T−1: **+0.44 bps**, t = 0.81, p = 0.417; expected sign was negative.
- Immediate T−1 to T+5: +0.24 bps, t = 2.47, p = 0.014.
- Post-fix T+5 to T+180: +0.22 bps, t = 0.65, p = 0.518.
- From 2021 onward: pre +0.78 bps and post −0.56 bps, both opposite to the canonical V-shaped mechanism as a pair.
- Yearly signs are unstable; the paper-like pattern is clearer only in some early years and does not persist recently.

The small immediate move is not enough to establish the multi-hour inventory reversal and is not evidence of tradability by itself.

## Phase gate

`DO_NOT_ADVANCE`. The London-fix event pattern fails the canonical sign and recent-period tests before costs. No execution or timing landscape is run.

Files: `event_level.csv`, `event_path_summary.csv`, `yearly_metrics.csv`, `aggregate_metrics.json`, and `six_e_london_fix_event_path.png`.
