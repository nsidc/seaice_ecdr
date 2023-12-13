"""Gather/compile various masks/ancillary fields for ECDR into ancillary NetCDF files.

CDR v4 has ancillary files with the following variables:
* `crs`
* `landmask`
* `latitude`
* `longitude`
* `min_concentration`
* `month`
* `polehole` (NH only)
* `valid_ice_mask`
* `x`
* `y`


NOTE: this code assumes `pm_icecon` v0.2.0 and `seaice_ecdr` git hash 27e5c09.


TODO: Add shoremap? The shoremap has values 1-5 that indicate land, coast, and
cells away from coast (3-5). Used by nasateam.
"""

import datetime as dt

import numpy as np
import xarray as xr
from pm_icecon.bt.masks import get_ps_invalid_ice_mask
from pm_icecon.nt.params.amsr2 import get_amsr2_params
from pm_tb_data._types import NORTH, SOUTH, Hemisphere

from seaice_ecdr.constants import CDR_ANCILLARY_DIR
from seaice_ecdr.masks import get_surfgeo_ds


def ecdr_invalid_ice_masks_12km(
    *, hemisphere: Hemisphere, surfgeo_ds: xr.Dataset
) -> xr.DataArray:
    """Gather invalid ice mask fields returned by pm_icecon.bt.fields.get_bootstrap_fields for the 12.5km resolution.

    This function re-implements the logic from pm_icecon's
    `get_ps_invalid_ice_mask` for the 12.5km resolution so that it is more clear
    from this project's perspective the provenance of the ancillary files.

    Currently used by the initial_daily_ecdr module.

    Masks are available for both 12.5 and 25km resolutions. Currently the ECDR
    only supports 12.5km resolution.

    In the Northern Hemisphere, invalid ice masks are based on SST mask files
    originally found in
    `/share/apps/amsr2-cdr/cdr_testdata/bt_goddard_ANCILLARY/`.
    """
    invalid_icemasks = []
    months = range(1, 12 + 1)
    for month in months:
        invalid_icemask = get_ps_invalid_ice_mask(
            hemisphere=hemisphere,
            # Only the month is used.
            date=dt.date(2023, month, 1),
            resolution="12",
        )
        invalid_icemasks.append(invalid_icemask)

    da = xr.DataArray(
        data=invalid_icemasks,
        coords=dict(
            month=months,
            y=surfgeo_ds.y,
            x=surfgeo_ds.x,
        ),
        attrs=dict(
            grid_mapping="crs",
            flag_meanings="valid_seaice_location invalid_seaice_location",
            flag_values=[np.byte(0), np.byte(1)],
        ),
    )

    da.encoding = dict(zlib=True)

    return da


def ecdr_min_ice_concentration_12km(
    *, hemisphere: Hemisphere, surfgeo_ds: xr.Dataset
) -> xr.DataArray:
    """Used by the nasateam algorithm."""
    params = get_amsr2_params(
        hemisphere=hemisphere,
        resolution="12",
    )

    da = xr.DataArray(
        data=params.minic,
        coords=dict(
            y=surfgeo_ds.y,
            x=surfgeo_ds.x,
        ),
        attrs=dict(
            grid_mapping="crs",
        ),
    )

    da.encoding = dict(zlib=True)

    return da


if __name__ == "__main__":
    for hemisphere in (NORTH, SOUTH):
        surfgeo_ds = get_surfgeo_ds(
            hemisphere=hemisphere,
            resolution="12.5",
        )
        invalid_ice_masks = ecdr_invalid_ice_masks_12km(
            hemisphere=hemisphere, surfgeo_ds=surfgeo_ds
        )
        minimum_concentration = ecdr_min_ice_concentration_12km(
            hemisphere=hemisphere, surfgeo_ds=surfgeo_ds
        )

        data_vars = dict(
            crs=surfgeo_ds.crs,
            surface_type=surfgeo_ds.surface_type,
            latitude=surfgeo_ds.latitude,
            longitude=surfgeo_ds.longitude,
            min_concentration=minimum_concentration,
            month=invalid_ice_masks.month,
            invalid_ice_mask=invalid_ice_masks,
            x=surfgeo_ds.x,
            y=surfgeo_ds.y,
        )

        ancillary_ds = xr.Dataset(data_vars=data_vars)
        if hemisphere == NORTH:
            # TODO: should this have `flag_masks` instead of `flag_meanings`?
            ancillary_ds["polehole_bitmask"] = surfgeo_ds.polehole_bitmask

        ancillary_ds.to_netcdf(
            CDR_ANCILLARY_DIR / f"ecdr-ancillary-{hemisphere[0]}h.nc"
        )
