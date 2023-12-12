"""Create netCDF file with surface type and geolocation arrays.

The routines here use the output of
scripts/surface_geo_mask/create_surface_geo_mask.py
"""

import datetime as dt
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pm_tb_data._types import NORTH, Hemisphere

from seaice_ecdr._types import ECDR_SUPPORTED_RESOLUTIONS, SUPPORTED_SAT
from seaice_ecdr.constants import CDR_ANCILLARY_DIR
from seaice_ecdr.grid_id import GRID_ID, get_grid_id


def get_surfacegeomask_filepath(grid_id: str) -> Path:
    filepath = CDR_ANCILLARY_DIR / f"cdrv5_surfgeo_{grid_id}.nc"

    return filepath


@cache
def get_surfgeo_ds(grid_id: GRID_ID) -> xr.Dataset:
    """Return xr Dataset of ancillary surface/geolocation for this grid."""
    return xr.load_dataset(get_surfacegeomask_filepath(grid_id))


# TODO: move to util  module?
def _get_sat_by_date(
    date: dt.date,
) -> SUPPORTED_SAT:
    """Return the satellite used for this date."""
    # TODO: these date ranges belong in a config location
    if date >= dt.date(2012, 7, 2) and date <= dt.date(2030, 12, 31):
        return "am2"
    else:
        raise RuntimeError(f"Could not determine sat for date: {date}")


def _bitmask_value_for_meaning(*, var: xr.DataArray, meaning: str):
    index = var.flag_meanings.split(" ").index(meaning)
    value = var.flag_values[index]

    return value


def get_surfacetype_da(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
) -> xr.DataArray:
    """Return a dataarray with surface type information for this date."""
    grid_id = get_grid_id(
        hemisphere=hemisphere,
        resolution=resolution,
    )
    surfgeo_ds = get_surfgeo_ds(grid_id)

    xvar = surfgeo_ds.variables["x"]
    yvar = surfgeo_ds.variables["y"]
    surftypevar = surfgeo_ds.variables["surface_type"].copy()
    polehole_surface_type = 100
    if "polehole_bitmask" in surfgeo_ds.data_vars.keys():
        polehole_bitmask = surfgeo_ds.polehole_bitmask
        sat = _get_sat_by_date(date)
        polehole_bitlabel = f"{sat}_polemask"
        polehole_bitvalue = _bitmask_value_for_meaning(
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
        name="surface_mask",
        data=surftypevar.data,
        dims=["y", "x"],
        coords=dict(
            y=(
                [
                    "y",
                ],
                yvar.data,
            ),
            x=(
                [
                    "x",
                ],
                xvar.data,
            ),
        ),
        attrs={
            "grid_mapping": "crs",
            "flag_values": surftype_flag_values_arr,
            "flag_meanings": surftype_flag_meanings_str,
        },
    )

    # Add the date
    surface_mask_da = surface_mask_da.expand_dims(time=[pd.to_datetime(date)])

    return surface_mask_da


def nh_polehole_mask(
    *, date: dt.date, resolution: ECDR_SUPPORTED_RESOLUTIONS
) -> xr.DataArray:
    """Return the northern hemisphere pole hole mask for the given date and resolution."""
    grid_id = get_grid_id(
        hemisphere=NORTH,
        resolution=resolution,
    )
    surfgeo_ds = get_surfgeo_ds(
        grid_id=grid_id,
    )

    polehole_bitmask = surfgeo_ds.polehole_bitmask

    sat = _get_sat_by_date(
        date=date,
    )

    polehole_bitlabel = f"{sat}_polemask"
    polehole_bitvalue = _bitmask_value_for_meaning(
        var=polehole_bitmask,
        meaning=polehole_bitlabel,
    )

    polehole_mask = (polehole_bitmask & polehole_bitvalue) > 0

    return polehole_mask
