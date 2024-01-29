"""Adapt the CDR v4 ancillary daily climatology files for CDR v5.
"""
from pathlib import Path
from typing import get_args

import numpy as np
import xarray as xr
from loguru import logger
from pm_tb_data._types import Hemisphere

from seaice_ecdr.ancillary import get_ancillary_ds
from seaice_ecdr.constants import CDR_ANCILLARY_DIR
from seaice_ecdr.grid_id import get_grid_id

# TODO: update this path to a local copy of the `seaice_cdr`.
CDR_V4_ANCILLARY_DIR = Path("/home/trst2284/code/seaice_cdr/source/ancillary/")


def get_v4_climatology(*, hemisphere: Hemisphere) -> xr.Dataset:
    ds = xr.open_dataset(CDR_V4_ANCILLARY_DIR / f"doy-validice-{hemisphere}-smmr.nc")

    return ds


def make_v5_climatology(*, hemisphere: Hemisphere):
    v4_ds = get_v4_climatology(hemisphere=hemisphere)

    # Create invalid ice mask from v4 climatology ancillary files. This sets
    # ocean as nan, valid ice as 1, 2, or 3, and 254 for land.
    daily_icemask = v4_ds.daily_icemask.data
    valid_icemask = (daily_icemask > 0) & (daily_icemask <= 3)
    invalid_icemask = ~valid_icemask

    ancillary_ds = get_ancillary_ds(
        hemisphere=hemisphere,
        resolution="25",
    )

    invalid_ice_mask_arr = xr.DataArray(
        invalid_icemask.astype(np.uint8),
        dims=("doy", "y", "x"),
        attrs=ancillary_ds.invalid_ice_mask.attrs,
    )
    invalid_ice_mask_arr.encoding["_FillValue"] = None

    v5_ds = xr.Dataset(
        data_vars=dict(
            invalid_ice_mask=invalid_ice_mask_arr,
            crs=ancillary_ds.crs,
        ),
        coords=dict(
            # 366 days to account for leap years.
            doy=np.arange(1, 366 + 1, dtype=np.uint8),
            y=ancillary_ds.y,
            x=ancillary_ds.x,
        ),
    )

    # TODO: any other attrs for the day of year coordinate?
    v5_ds.doy.attrs = dict(
        long_name="Day of year",
        comment="366 days are provided to account for leap years.",
    )

    # Preserve the geospatial global attrs
    v5_ds.attrs = {k: v for k, v in ancillary_ds.attrs.items() if "geospatial_" in k}
    v5_ds.attrs["comment"] = v4_ds.comment

    # Ensure coordinate vars don't get a fill value
    v5_ds.doy.encoding["_FillValue"] = None
    v5_ds.y.encoding["_FillValue"] = None
    v5_ds.x.encoding["_FillValue"] = None

    grid_id = get_grid_id(hemisphere=hemisphere, resolution="25")
    output_filepath = (
        CDR_ANCILLARY_DIR / f"ecdr-ancillary-{grid_id}-smmr-invalid-ice.nc"
    )
    v5_ds.to_netcdf(output_filepath)
    logger.info(f"Wrote {output_filepath}")


if __name__ == "__main__":
    for hemisphere in get_args(Hemisphere):
        make_v5_climatology(hemisphere=hemisphere)
