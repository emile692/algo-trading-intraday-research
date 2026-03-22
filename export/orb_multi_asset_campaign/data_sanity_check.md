# Data Sanity Check

The campaign uses already processed datasets from `data/processed/parquet`.

## MNQ

- File: `MNQ_c_0_1m_20260321_094501.parquet`
- Rows: 2,401,697
- Date range: 2019-05-05 18:03:00-04:00 -> 2026-03-20 09:29:00-04:00
- Timezone: `America/New_York`
- Duplicate timestamps after cleaning: 0
- OHLC incoherent rows: 0
- Sessions with OR available: 1747
- Sessions with incomplete 15-minute opening window: 399
- Sessions missing at least one RTH minute: 477
- Median RTH missing bars per session: 0.0

## MES

- File: `MES_c_0_1m_20260322_135702.parquet`
- Rows: 2,399,361
- Date range: 2019-05-05 18:00:00-04:00 -> 2026-03-20 09:29:00-04:00
- Timezone: `America/New_York`
- Duplicate timestamps after cleaning: 0
- OHLC incoherent rows: 0
- Sessions with OR available: 1747
- Sessions with incomplete 15-minute opening window: 399
- Sessions missing at least one RTH minute: 477
- Median RTH missing bars per session: 0.0

## M2K

- File: `M2K_c_0_1m_20260322_134808.parquet`
- Rows: 2,232,652
- Date range: 2019-05-05 18:01:00-04:00 -> 2026-03-20 09:29:00-04:00
- Timezone: `America/New_York`
- Duplicate timestamps after cleaning: 0
- OHLC incoherent rows: 0
- Sessions with OR available: 1747
- Sessions with incomplete 15-minute opening window: 410
- Sessions missing at least one RTH minute: 593
- Median RTH missing bars per session: 0.0
