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
import numpy.typing as npt


# Start and end DOYs for the melt season (inclusive)
MELT_SEASON_START_DOY = 60
MELT_SEASON_END_DOY = 244

# Flag value for grid cells before melting is detected or if no melt is ever
# detected.
MELT_ONSET_DAY_FILL_VALUE = -1

# Melt detection requires a SIC of >= this value
CONCENTRATION_THRESHOLD_PERCENT = 50

# One of the conditions for melt detection is that the delta between 19H and 37H
# (19H-37H) is less than this value.
TB_DELTA_THRESHOLD_K = 2

# TODO: In the CDR, tbs that are missing are assumed to be 0. That's not the case
# anymore. We expect NaN.
TB_MISSING = 0


def melting(
    concentrations: npt.NDArray,
    tb19: npt.NDArray,
    tb37: npt.NDArray,
) -> npt.NDArray[np.bool_]:
    """Determine melting locations.

    Expects brightness temperatures in degrees Kelvin.
    """
    melting = np.logical_and(
        concentrations >= CONCENTRATION_THRESHOLD_PERCENT,
        tb19 - tb37 < TB_DELTA_THRESHOLD_K,
    )
    melting[tb19 == TB_MISSING] = False
    melting[tb37 == TB_MISSING] = False

    return melting
