# Auction timing audit

## A — Competitive bid deadline

The official FiscalData `auctions_query` dataset supplies `auction_date` and `closing_time_comp`. These are localized in `America/New_York` and converted to UTC with DST handling. This field is complete for the retained 5Y, 7Y and 10Y auctions and is the sole event reference used in the final study.

## B — Result/publication timestamp

The FiscalData dataset and historical result PDFs provide the result date, but not a consistent exact publication timestamp. Treasury's timeline documents changing release targets: about five minutes in 2002, official XML delivery from 2003, around two minutes from August 2004, and participant award notices at 2 minutes 45 seconds from March 2014. These are targets or system milestones, not actual timestamps for every auction.

Fleming, Liu & Nguyen instead identify the first Bloomberg story reporting each result and impute some missing observations. That proprietary series is not present in this repository or the official API.

Decision: `result_ts_utc` remains null. Deadline-based and result-based evidence are kept separate by omitting the unavailable result-based study rather than silently substituting a fixed delay.
