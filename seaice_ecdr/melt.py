"""Code related to melt detection.

Note: quotes below (>) are taken from the CDR v4 ATBD:
https://nsidc.org/sites/default/files/documents/technical-reference/cdrp-atbd-final.pdf

Includes code repsonsible for creating the `cdr_melt_onset_day`
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

import datetime as dt

import numpy as np
import numpy.typing as npt

# Start and end DOYs for the melt season (inclusive)
MELT_SEASON_FIRST_DOY = 60
MELT_SEASON_LAST_DOY = 244

# Flag value for grid cells before melting is detected or if no melt is ever
# detected.
MELT_ONSET_FILL_VALUE = 255

# Melt detection requires a SIC of >= this value
CONCENTRATION_THRESHOLD_FRACTION = 0.5

# One of the conditions for melt detection is that the delta between 19H and 37H
# (19H-37H) is less than this value.
TB_DELTA_THRESHOLD_K = 2

# Concentrations are provided as sea_ice_area_fraction, a number between 0 and 1
MAX_VALID_CONCENTRATION = 1.0


def date_in_nh_melt_season(*, date: dt.date) -> bool:
    """Return `True` if the date is during the NH melt season."""
    day_of_year = int(date.strftime("%j"))
    outside_of_melt_season = (day_of_year < MELT_SEASON_FIRST_DOY) or (
        day_of_year > MELT_SEASON_LAST_DOY
    )

    inside_melt_season = not outside_of_melt_season

    return inside_melt_season


def melting(
    concentrations: npt.NDArray,
    tb_h19: npt.NDArray,
    tb_h37: npt.NDArray,
) -> npt.NDArray[np.bool_]:
    """Determine melting locations.

    Expects brightness temperatures in degrees Kelvin.
    """
    is_valid_concentration = np.logical_and(
        concentrations >= CONCENTRATION_THRESHOLD_FRACTION,
        concentrations <= MAX_VALID_CONCENTRATION,
    )
    melting = np.logical_and(
        is_valid_concentration,
        tb_h19 - tb_h37 < TB_DELTA_THRESHOLD_K,
    )

    melting[np.isnan(tb_h19)] = False
    melting[np.isnan(tb_h37)] = False

    return melting
