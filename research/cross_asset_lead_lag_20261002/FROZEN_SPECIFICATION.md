# Frozen specification — cross-asset lead/lag

Frozen on 2026-10-02 before viewing any cross-asset event result.

## Pairs and data

- Potential leaders of MNQ: MES and M2K.
- Potential leaders of MGC: 6E and ZN.
- Local one-minute continuous data only; no download. Common evaluation interval: 2019-05-06 through 2026-03-19.
- IS: 2019-05-06–2022-12-31; validation: calendar 2023; untouched campaign OOS: 2024-01-01–2026-03-19.
- America/New_York. Target RTH: MNQ 09:30–16:00; MGC 08:20–13:30. Five-minute bars are left-closed/right-open and synchronized by exact timestamp with an inner join.
- A synchronized session is rejected when either available `instrument_id` changes inside it. The processed 6E file lacks `instrument_id`; roll-safety is therefore unavailable for that leg and is reported as a limitation.

## Causal event definition

- At completed five-minute leader bar `t`, calculate leader log impulse over 5, 15, or 30 minutes.
- Leader impulse z-score uses rolling standard deviation of five-minute leader returns from completed bars through `t-1` only, scaled by the square root of the impulse length.
- Rolling beta of target returns on leader returns uses 60 or 120 synchronized five-minute observations ending at `t-1`. The current leader bar and current target bar are excluded from beta estimation.
- To ensure no contemporaneous target return becomes tradable information, target response is measured only through `t-1`. Expected target move is `beta(t) × leader_impulse(t)`. Residual under-reaction is the target move through `t-1` minus that expected move.
- An event requires leader |z| ≥ 1.5, 2.0, or 2.5; a non-zero expected target direction; lagged target response in that direction below 50% of the expected move; and residual opposite to the expected direction. Only the first event per pair/side/config/session is retained.
- Signal becomes known at the end of bar `t`; earliest hypothetical entry is the exact target open of the next bar. Forward target returns are measured from that open to +5, +15, +30, and +60 minutes. Missing exact entry/exit bars reject the observation.
- Long and short, every pair, beta window, impulse window, and threshold are reported separately. Overlap across configurations is measured.

## Inference and gates

- Small contiguous grid: beta windows 60/120 five-minute bars; impulse windows 5/15/30 minutes; leader thresholds 1.5/2.0/2.5.
- Canonical cell fixed ex ante: beta 120 bars, impulse 15 minutes, threshold 2.0; primary forward horizon 30 minutes. Neighboring cells diagnose plateaus and are not candidates selected by maximum Sharpe.
- Session-cluster bootstrap with 1,000 samples and seed 20261002. Bonferroni correction is applied within pair/side families.
- Monotonicity requires the signed 30-minute mean to be non-decreasing across the three thresholds for the canonical beta/impulse window, allowing a numerical tolerance of 0.05 bps.
- Stability requires positive canonical signed expectancy in IS, validation, and OOS; at least 60% of eligible years positive; at least 80 full-sample and 25 OOS events; and positive contiguous-neighbor majority.
- Economic gate requires the canonical lower 95% cluster-bootstrap bound above zero and mean gross target PnL above the repository round-trip cost. The effect must remain positive net after +1/+2/+3 ticks per side, doubled commission, and an additional five-minute entry delay.
- If any directional pair fails these event gates, it receives `REJECTED_NO_LEAD_LAG_EDGE`. No strategy or Topstep simulation is permitted for a rejected pair.

## Conditional strategy, portfolio and Topstep

- Only an event-gate survivor may become a one-contract strategy: enter at the canonical next-bar open in the expected direction, exit after 30 minutes, maximum one position per target and session.
- Target costs: MNQ tick 0.25, point value USD 2; MGC tick 0.10, point value USD 10; commission USD 1.25 per side and one adverse tick per side.
- Standalone validation uses the frozen IS/validation/OOS splits and the same cost/delay stresses. Marginal comparison uses the six frozen sleeve trade files without modifying or recalibrating them.
- A full Topstep PASS/MLL/EXPIRY simulation is allowed only after both standalone and diversification gates pass.

