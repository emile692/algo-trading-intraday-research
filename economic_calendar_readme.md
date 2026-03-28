# Economic Calendar Research Pipeline

This repo now includes a reproducible economic calendar preparation pipeline for intraday futures research.

## Expected raw input

Default fallback file:

- `data/raw/economic_calendar_us.csv`

Expected columns:

- `event_name`
- `country`
- `date`
- `time`
- `impact`
- `actual`
- `forecast`
- `previous`

Optional columns are also supported when available, including:

- `source_timezone`
- `currency`
- `timestamp`
- `event_timestamp`

The included repo CSV is a small illustrative sample so the pipeline runs locally without any external dependency. For real research coverage, replace or extend it with the full historical event file for your study window.

## How to run

Build the cleaned event-level dataset and the daily feature dataset from the local CSV fallback:

```bash
PYTHONPATH=. python -m src.data.economic_calendar.build_event_features --raw-path data/raw/economic_calendar_us.csv --output-dir data/processed/economic_calendar
```

Optional API-assisted raw fetch or validation step:

```bash
PYTHONPATH=. python -m src.data.economic_calendar.fetch_calendar --raw-path data/raw/economic_calendar_us.csv --output-path data/raw/economic_calendar_us.csv
```

If an API is available, set `ECON_CAL_API_URL` and `ECONOMIC_CALENDAR_API_KEY`, then pass `--api-url` or rely on the environment variable.

## Output files

Written by default to:

- `data/processed/economic_calendar/economic_calendar_events.csv`
- `data/processed/economic_calendar/economic_calendar_daily_features.csv`

The event-level dataset includes:

- raw and canonical event names
- local and UTC timestamps
- local date, time, weekday, and precise-time flag
- event grouping and high-impact flags

The daily feature dataset includes:

- one row per local date in the covered range
- event-type day flags
- high-impact macro day summary flags
- first event time
- pre-09:30 / RTH / post-16:00 timing buckets

## Supported event universe

Mandatory event types:

- FOMC rate decision
- FOMC minutes
- Powell / Fed Chair speech
- CPI
- Core CPI
- NFP
- Core PCE
- PPI
- ISM Manufacturing PMI
- ISM Services PMI
- Retail Sales
- GDP QoQ

Optional event types already mapped if present:

- ADP Nonfarm Employment
- Initial Jobless Claims
- University of Michigan Sentiment

Only US events are retained.

## Limitations

- The bundled raw CSV is only a sample, not a full research history.
- Missing or tentative times are preserved as imprecise events; the pipeline does not invent timestamps.
- Intraday timing helpers currently focus on precise high-impact events and use the next/last known high-impact timestamp.
- Market-holiday-specific trading calendars are not yet applied to the daily date range.

## Recommended next research steps

1. Backfill the raw CSV for the full MNQ research period.
2. Merge `economic_calendar_daily_features.csv` into daily strategy results.
3. Compare conditional performance on:
   - normal days
   - FOMC days
   - CPI / NFP days
   - all high-impact macro days
4. Add pre-event risk throttling or event-day exclusion rules inside the Topstep survivability simulations.
