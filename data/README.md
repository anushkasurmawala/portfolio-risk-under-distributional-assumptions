# Data provenance and units

Source: Kenneth R. French Data Library, 6 Portfolios Formed on Size and
Book-to-Market (2 × 3), daily, average value-weighted returns.

Dataset construction:
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/six_portfolios.html

Download:
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/6_Portfolios_2x3_daily_CSV.zip

The downloaded file is retained without modification in raw/. Its first line
identifies the 202608 CRSP database snapshot; the SHA-256 fingerprint is saved
in reports/run_metadata.json. The runner can download it when absent, although
future provider revisions can change historical values and hence results.

The parser selects ONLY the first Average Value Weighted Returns -- Daily
section. The separate equal-weighted section is not used. Returns in the
source are percentage points, converted to decimal fractions by dividing by
100. -99.99 and -999 are missing-value markers, removed before conversion.
Only complete daily observations from 2010-01-01 through 2024-12-31 are used.
No dates are forward-filled and no prices are reconstructed.

The six sleeves represent intersections of two size groups and three
book-to-market groups. The underlying portfolios are formed at the end of
June according to the provider's methodology. Within each sleeve, source
returns are value-weighted. Our separate allocation is one sixth to each sleeve
at the start of every trading day, ignoring turnover costs.

These are academic research portfolio returns, not ETF prices, investable
products, bond returns, or a tradable fund track record. They are total returns,
not excess returns; no risk-free rate is subtracted. The current data library
uses CRSP CIZ inputs, with a provider-announced transition starting in January
2025. The study uses a downloaded historical snapshot, not a point-in-time
vintage backtest of what data were available at each historical date.

The data are publicly downloadable but are not covered by the project's MIT
license. Credit the provider and consult its terms before further redistribution.
