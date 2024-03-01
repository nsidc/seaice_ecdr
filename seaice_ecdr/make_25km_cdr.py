"""Produce versions of the 25km CDR.

F17 starts on 2008-01-01.
AMSR2 starts on 2012-07-02.

So we should target 2012-07-02 as the start date for comparisons between the two
platforms.

We want to be able to:
* Use both nasateam land spillover techniques.
* 
"""

import datetime as dt
from pathlib import Path
from typing import Literal

import xarray as xr
from pm_tb_data._types import Hemisphere

from seaice_ecdr.grid_id import get_grid_id
from seaice_ecdr.util import get_complete_output_dir


def get_25km_daily_cdr(
    *,
    alg: Literal["BT_NT", "NT2"],
    hemisphere: Hemisphere,
    platform: Literal["am2", "F17"],
) -> xr.Dataset:
    """Return the 25km CDR for the given algorithm."""
    if alg == "BT_NT":
        base_dir = Path("/share/apps/G02202_V5/25km/BT_NT")
    elif alg == "NT2":
        base_dir = Path("/share/apps/G02202_V5/25km/NT2")
    else:
        raise RuntimeError(f"Unrecognized alg {alg}")

    complete_dir = get_complete_output_dir(
        base_output_dir=base_dir, hemisphere=hemisphere, is_nrt=False
    )

    grid_id = get_grid_id(hemisphere=hemisphere, resolution="25")

    glob_pattern = f"sic_{grid_id}_*_{platform}_*.nc"
    matching_files = list(complete_dir.glob(f"**/{glob_pattern}"))

    if not matching_files:
        raise RuntimeError(f"No files found matching {glob_pattern} in {base_dir}")

    xr_ds = xr.open_mfdataset(matching_files, engine="rasterio")
    # Convert `DatetimeGregorian` to native date. Not sure why this happens when engine is rasterio.
    xr_ds["time"] = [
        dt.date(thing.year, thing.month, thing.day) for thing in xr_ds.time.values
    ]

    return xr_ds


if __name__ == "__main__":
    f17_nt = get_25km_daily_cdr(alg="BT_NT", hemisphere="north", platform="F17")
    breakpoint()
