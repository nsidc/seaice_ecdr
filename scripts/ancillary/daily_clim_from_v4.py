"""Adapt the CDR v4 ancillary daily climatology files for CDR v5.
"""

from pathlib import Path
from typing import get_args

import numpy as np
import numpy.typing as npt
import xarray as xr
from loguru import logger
from pm_tb_data._types import Hemisphere
from scipy.signal import convolve2d

from seaice_ecdr.ancillary import get_ancillary_ds
from seaice_ecdr.constants import CDR_ANCILLARY_DIR
from seaice_ecdr.grid_id import get_grid_id

CDR_V4_CODE_DIR = Path("~/seaice_cdr").resolve()
if not CDR_V4_CODE_DIR.is_dir():
    # Alert the user to this dependency.
    raise RuntimeError(
        "The `seaice_cdr` (https://bitbucket.org/nsidc/seaice_cdr/) repository is"
        " expected to be manually cloned to `CDR_V4_CODE_DIR` for this script to work."
    )
CDR_V4_ANCILLARY_DIR = CDR_V4_CODE_DIR / "/source/ancillary/"


def get_v4_climatology(*, hemisphere: Hemisphere) -> xr.Dataset:
    ds = xr.load_dataset(
        CDR_V4_ANCILLARY_DIR / f"doy-validice-{hemisphere}-smmr.nc",
        mask_and_scale=False,
    )

    return ds


def get_v4_icemask(
    *, surftype: npt.NDArray, old_vim: npt.NDArray
) -> npt.NDArray[np.uint8]:
    # Investigate arrays
    is_new_ocean = surftype == 50
    is_new_land = ~is_new_ocean

    n_days, ydim, xdim = old_vim.shape
    new_vim = np.zeros((n_days, ydim, xdim), dtype=np.uint8)
    new_vim[:] = 255
    for d in range(n_days):
        if (d % 20) == 0:
            print(f"{d} of {n_days}", flush=True)

        old_slice = old_vim[d, :, :]
        new_slice = new_vim[d, :, :]
        new_slice[is_new_land] = 3

        # Valid ice in CDRv4 file has bit 1 set, so values of 1 and 3 match
        # Invalid (daily!) ice in CDRv4 does not have bit 0 set, so ... is 0 or 2
        #   because there are a few places where the daily SMMR mask is valid but where
        #   the CDRv4 monthly mask was not.  So, invalid is 0 or 2
        # Values above 250 encoded lake, coast, land and are ignored here
        is_valid = is_new_ocean & ((old_slice == 1) | (old_slice == 3))
        is_invalid = is_new_ocean & ((old_slice == 0) | (old_slice == 2))

        new_slice[is_valid] = 1
        new_slice[is_invalid] = 0

        n_unassigned = np.sum(np.where(is_new_ocean & (new_slice == 255), 1, 0))
        n_unassigned_prior = n_unassigned
        kernel = np.ones((3, 3), dtype=np.uint8)
        while n_unassigned > 0:
            # Expand valid
            convolved = convolve2d(
                (new_slice == 1).astype(np.uint8),
                kernel,
                mode="same",
                boundary="fill",
                fillvalue=0,
            )
            is_newly_valid = is_new_ocean & (convolved > 0) & (new_slice == 255)
            new_slice[is_newly_valid] = 1

            # Expand invalid
            convolved = convolve2d(
                (new_slice == 0).astype(np.uint8),
                kernel,
                mode="same",
                boundary="fill",
                fillvalue=0,
            )
            is_newly_invalid = is_new_ocean & (convolved > 0) & (new_slice == 255)
            new_slice[is_newly_invalid] = 0

            # Check to see if we've assigned all ocean grid cells to valid or invalid
            n_unassigned = np.sum(np.where(is_new_ocean & (new_slice == 255), 1, 0))
            if n_unassigned == n_unassigned_prior:
                print(f"breaking with {n_unassigned} still unassigned")
                break

            n_unassigned_prior = n_unassigned

        if not np.all(new_slice <= 3):
            print(f"uh oh...not all points were assigned! d={d}")
            breakpoint()

    # Here, new_vim is the (366, ydim, xdim) VALID ice_mask on the new "land" mask
    assert np.all(new_vim <= 3)

    # Convert to INVALID ice mask by swapping 0s and 1s and by replacing the land values
    # with 0s
    vim_is_valid = new_vim == 1
    vim_is_invalid = new_vim == 0
    vim_is_land = new_vim == 3  # this was set above

    # Reassign values, and make sure we got them all
    new_vim[:] = 255
    new_vim[vim_is_invalid] = 1
    new_vim[vim_is_valid] = 0
    new_vim[vim_is_land] = 1

    assert np.all(new_vim <= 1)

    return new_vim


def make_v5_climatology(*, hemisphere: Hemisphere):
    v4_ds = get_v4_climatology(hemisphere=hemisphere)
    ancillary_ds = get_ancillary_ds(
        hemisphere=hemisphere,
        resolution="25",
    )
    icemask_arr = get_v4_icemask(
        surftype=ancillary_ds.surface_type.data, old_vim=v4_ds.daily_icemask.data
    )

    invalid_ice_mask_arr = xr.DataArray(
        icemask_arr,
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
            doy=np.arange(1, 366 + 1, dtype=np.int16),
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
