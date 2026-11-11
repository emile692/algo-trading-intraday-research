# Frozen timestamp sources

- **NQ/MNQ, MES and M2K — 16:00 America/New_York.** The experiment follows the U.S. cash-market close convention used by the published mechanism. CME states that the affected E-mini and Micro E-mini equity-index daily settlement determination moved to 16:00 ET, and that Micro E-mini settlement uses the 15:59:30–16:00:00 ET VWAP: https://www.cmegroup.com/education/articles-and-reports/faq-daily-settlement-price-determination-time-change.html and https://www.cmegroup.com/articles/faqs/micro-e-mini-equity-index-futures-frequently-asked-questions.html
- **MGC — 13:30 America/New_York.** COMEX Gold's active-month settlement interval is 13:29–13:30 ET: https://www.cmegroup.com/tools-information/lookups/advisories/market-regulation/SER-5147.html
- **MCL — 14:30 America/New_York.** NYMEX WTI's settlement interval is 14:28–14:30 ET. MCL is tested at the corresponding crude-oil settlement boundary: https://www.cmegroup.com/trading/energy/files/NYMEX_Energy_Futures_Daily_Settlement_Procedure.pdf
- **6E — 14:00 America/Chicago.** CME's daily-settlement documentation identifies the lead-month FX settlement at 14:00 CT: https://www.cmegroup.com/market-regulation/files/13-433R.pdf
- **ZN — 14:00 America/Chicago.** CME materials identify 14:00 CT as the Treasury futures daily settlement: https://www.cmegroup.com/content/dam/cmegroup/education/files/treasury-futures-basis-spreads.pdf

The IANA zones, rather than fixed UTC offsets, are frozen so U.S. daylight-saving transitions are handled explicitly. These are economic marker times, not the end of the nearly 24-hour electronic trading session.
