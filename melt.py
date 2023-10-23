"""Code related to melt detection.

Note: quotes below (>) are taken from the CDR v4 ATBD:
https://nsidc.org/sites/default/files/documents/technical-reference/cdrp-atbd-final.pdf

Includes code repsonsible for creating the `melt_onset_day_cdr_seaice_conc`
variable in final CDR files:

> contains the day of year on which melting sea ice was first detected in each
  grid cell. Once detected, the value is retained for the rest of the year. For
  example, if a grid cell started melting on day 73, the variable for the grid
  cell on that day will be 73, as will all subsequent days until the end of the
  year. The melt onset day is only calculated for the melt season: days 60
  through 244, inclusive. Before melting is detected or if no melt is ever
  detected for that grid cell, the value will be -1 (missing / fill value).
"""

# Start and end DOYs for the melt season (inclusive)
MELT_SEASON_START_DOY = 60
MELT_SEASON_END_DOY = 244

# Flag value for grid cells before melting is detected or if no melt is ever
# detected.
MELT_ONSET_DAY_FILL_VALUE = -1
