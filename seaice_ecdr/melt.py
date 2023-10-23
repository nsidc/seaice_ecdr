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

> The conditions for melt onset at a particular grid cell are the following:
  * Melt detected:
    - Concentration >= 50% at the beginning of the season
    - Grid cell is not land, coast, shore (1 grid cell from coast), near-shore
      (2 grid cells from coast), or lake
  * Current sea ice concentration >= 50%
  * Brightness temperature delta (19H - 37H) < 2K
  * Presence of brightness temperatures for both channels (19H, 37H)


Other notes:

* In the CDR code, it looks like we use spatially interpolated TBs as input.
"""
import numpy as np

# Start and end DOYs for the melt season (inclusive)
MELT_SEASON_START_DOY = 60
MELT_SEASON_END_DOY = 244

# Flag value for grid cells before melting is detected or if no melt is ever
# detected.
MELT_ONSET_DAY_FILL_VALUE = -1

# Melt detection requires a SIC of >= this value
CONCENTRATION_THRESHOLD_PERCENT = 50

# TODO: this assumes that the TBs are provided in tenths of K as well, but e.g.,
# AU_SI12 has TBS in degrees K.
TB_DELTA_THRESHOLD_TENTHS_K = 20

# TODO: In the CDR, tbs that are missing are assumed to be 0. That's not the case
# anymore. We expect NaN.
TB_MISSING = 0


def melting(concentrations, tb19, tb37):
    melting = np.logical_and(
        concentrations >= CONCENTRATION_THRESHOLD_PERCENT,
        tb19 - tb37 < TB_DELTA_THRESHOLD_TENTHS_K,
    )
    melting[tb19 == TB_MISSING] = False
    melting[tb37 == TB_MISSING] = False

    return melting
