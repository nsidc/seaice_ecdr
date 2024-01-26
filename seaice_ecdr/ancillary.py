"""Code for interacting with ancillary data required by the ECDR.

Ancillary fields (e.g., land mask, invalid ice masks, l90c field, etc) required
for ECDR processing are stored in an ancillary NetCDF file that is published
alongside the ECDR.
"""

import datetime as dt
from functools import cache
from pathlib import Path
from typing import get_args

import numpy as np
import pandas as pd
import xarray as xr
from pm_tb_data._types import NORTH, Hemisphere

from seaice_ecdr._types import ECDR_SUPPORTED_RESOLUTIONS
from seaice_ecdr.constants import CDR_ANCILLARY_DIR
from seaice_ecdr.grid_id import get_grid_id
from seaice_ecdr.platforms import get_platform_by_date


def get_ancillary_filepath(
    *, hemisphere: Hemisphere, resolution: ECDR_SUPPORTED_RESOLUTIONS
) -> Path:
    grid_id = get_grid_id(
        hemisphere=hemisphere,
        resolution=resolution,
    )

    filepath = CDR_ANCILLARY_DIR / f"ecdr-ancillary-{grid_id}.nc"

    return filepath


@cache
def get_ancillary_ds(
    *, hemisphere: Hemisphere, resolution: ECDR_SUPPORTED_RESOLUTIONS
) -> xr.Dataset:
    """Return xr Dataset of ancillary data for this hemisphere/resolution."""
    # TODO: This list could be determined from an examination of
    #       the ancillary directory
    if resolution not in get_args(ECDR_SUPPORTED_RESOLUTIONS):
        raise NotImplementedError(
            "ECDR currently only supports {get_args(ECDR_SUPPORTED_RESOLUTIONS)} resolutions."
        )

    filepath = get_ancillary_filepath(hemisphere=hemisphere, resolution=resolution)
    ds = xr.load_dataset(filepath)

    return ds


def bitmask_value_for_meaning(*, var: xr.DataArray, meaning: str):
    # TODO: where do we encounter the ValueError that requires setting
    # `meaning.lower()`? Can we just use a `.lower()` on the `flag_meanings` and
    # `meaning` to be consistent and never run into this problem? Are there any
    # cases where case matters?
    try:
        index = var.flag_meanings.split(" ").index(meaning)
    except ValueError:
        index = var.flag_meanings.split(" ").index(meaning.lower())

    value = var.flag_masks[index]

    return value


def flag_value_for_meaning(*, var: xr.DataArray, meaning: str):
    index = var.flag_meanings.split(" ").index(meaning)
    value = var.flag_values[index]

    return value


def get_surfacetype_da(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    sat,
) -> xr.DataArray:
    """Return a dataarray with surface type information for this date."""
    ancillary_ds = get_ancillary_ds(
        hemisphere=hemisphere,
        resolution=resolution,
    )

    xvar = ancillary_ds.variables["x"]
    yvar = ancillary_ds.variables["y"]
    surftypevar = ancillary_ds.variables["surface_type"].copy()
    polehole_surface_type = 100
    if "polehole_bitmask" in ancillary_ds.data_vars.keys():
        polehole_bitmask = ancillary_ds.polehole_bitmask
        if sat is None:
            sat = get_platform_by_date(date)
        elif sat == "ame":
            # TODO: Verify that AMSR-E pole hole is same as AMSR2
            # Use the AMSR2 pole hole for AMSR-E
            sat = "am2"
        polehole_bitlabel = f"{sat}_polemask"
        polehole_bitvalue = bitmask_value_for_meaning(
            var=polehole_bitmask,
            meaning=polehole_bitlabel,
        )
        polehole_mask = (
            np.bitwise_and(
                polehole_bitmask.data,
                polehole_bitvalue,
            )
            > 0
        )

        surftype_flag_values_arr = np.array(
            (50, 75, polehole_surface_type, 200, 250), dtype=np.uint8
        )
        surftype_flag_meanings_str = "ocean lake polehole_mask coast land"
    else:
        polehole_mask = np.empty(surftypevar.shape, dtype=np.bool_)
        polehole_mask[:] = False
        surftype_flag_values_arr = surftypevar.attrs["flag_values"]
        surftype_flag_meanings_str = surftypevar.attrs["flag_meanings"]

    surftypevar = surftypevar.where(
        ~polehole_mask,
        other=polehole_surface_type,
    )

    surface_mask_da = xr.DataArray(
        name="surface_type_mask",
        data=surftypevar.data,
        dims=["y", "x"],
        coords=dict(
            y=yvar,
            x=xvar,
        ),
        attrs={
            "grid_mapping": "crs",
            "flag_values": surftype_flag_values_arr,
            "flag_meanings": surftype_flag_meanings_str,
        },
    )

    # Add the date
    # NOTE: This date variable will NOT have any associated attributes
    #       and will not conform to CF-conventions.
    surface_mask_da = surface_mask_da.expand_dims(time=[pd.to_datetime(date)])

    return surface_mask_da


def nh_polehole_mask(
    *,
    date: dt.date,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    sat=None,
) -> xr.DataArray:
    """Return the northern hemisphere pole hole mask for the given date and resolution."""
    ancillary_ds = get_ancillary_ds(
        hemisphere=NORTH,
        resolution=resolution,
    )

    polehole_bitmask = ancillary_ds.polehole_bitmask

    if sat is None:
        sat = get_platform_by_date(
            date=date,
        )
    elif sat == "ame":
        sat = "am2"

    polehole_bitlabel = f"{sat}_polemask"
    polehole_bitvalue = bitmask_value_for_meaning(
        var=polehole_bitmask,
        meaning=polehole_bitlabel,
    )

    polehole_mask = (polehole_bitmask & polehole_bitvalue) > 0

    polehole_mask.attrs = dict(
        grid_mapping="crs",
        standard_name="pole_binary_mask",
        long_name="pole mask",
        comment="Mask indicating where pole hole might be",
        units="1",
    )

    return polehole_mask


def get_invalid_ice_mask(
    *, hemisphere: Hemisphere, month: int, resolution: ECDR_SUPPORTED_RESOLUTIONS
) -> xr.DataArray:
    ancillary_ds = get_ancillary_ds(
        hemisphere=hemisphere,
        resolution=resolution,
    )

    invalid_ice_mask = ancillary_ds.invalid_ice_mask.sel(month=month)

    # The invalid ice mask is indexed by month in the ancillary dataset. Drop
    # that coordinate.
    invalid_ice_mask = invalid_ice_mask.drop_vars("month")

    return invalid_ice_mask


def get_ocean_mask(
    *, hemisphere: Hemisphere, resolution: ECDR_SUPPORTED_RESOLUTIONS
) -> xr.DataArray:
    """Return a binary mask where True values represent `ocean`."""
    ancillary_ds = get_ancillary_ds(
        hemisphere=hemisphere,
        resolution=resolution,
    )

    surface_type = ancillary_ds.surface_type

    ocean_val = flag_value_for_meaning(
        var=surface_type,
        meaning="ocean",
    )

    ocean_mask = surface_type == ocean_val

    ocean_mask.attrs = dict(
        grid_mapping="crs",
        standard_name="ocean_binary_mask",
        long_name="ocean mask",
        comment="Mask indicating where ocean is",
        units="1",
    )

    ocean_mask.encoding = dict(zlib=True)

    return ocean_mask


def get_land_mask(
    *, hemisphere: Hemisphere, resolution: ECDR_SUPPORTED_RESOLUTIONS
) -> xr.DataArray:
    """Return a binary mask where True values represent `land`.

    This mask includes both land & coast.
    """
    ancillary_ds = get_ancillary_ds(
        hemisphere=hemisphere,
        resolution=resolution,
    )

    surface_type = ancillary_ds.surface_type

    land_val = flag_value_for_meaning(
        var=surface_type,
        meaning="land",
    )
    coast_val = flag_value_for_meaning(
        var=surface_type,
        meaning="coast",
    )

    land_mask = (surface_type == land_val) | (surface_type == coast_val)

    land_mask.attrs = dict(
        grid_mapping="crs",
        standard_name="land_binary_mask",
        long_name="land mask",
        comment="Mask indicating where land is",
        units="1",
    )

    land_mask.encoding = dict(zlib=True)

    return land_mask


def get_land90_conc_field(
    *, hemisphere: Hemisphere, resolution: ECDR_SUPPORTED_RESOLUTIONS
) -> xr.DataArray:
    ancillary_ds = get_ancillary_ds(
        hemisphere=hemisphere,
        resolution=resolution,
    )

    land90_da = ancillary_ds.l90c

    return land90_da


def get_adj123_field(
    *, hemisphere: Hemisphere, resolution: ECDR_SUPPORTED_RESOLUTIONS
) -> xr.DataArray:
    ancillary_ds = get_ancillary_ds(
        hemisphere=hemisphere,
        resolution=resolution,
    )

    adj123_da = ancillary_ds.adj123

    return adj123_da
