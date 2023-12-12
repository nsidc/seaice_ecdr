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
from pm_tb_data._types import Hemisphere

from seaice_ecdr._types import ECDR_SUPPORTED_RESOLUTIONS
from seaice_ecdr.constants import CDR_ANCILLARY_DIR


def get_surfacegeomask_filepath(grid_id: str) -> Path:
    filepath = CDR_ANCILLARY_DIR / f"cdrv5_surfgeo_{grid_id}.nc"

    return filepath


@cache
def get_surfgeo_ds(gridid):
    """Return xr Dataset of ancillary surface/geolocation for this grid."""
    return xr.load_dataset(get_surfacegeomask_filepath(gridid))


def get_polehole_mask(ds, sensor):
    """Return the 2d boolean mask where this sensor's polehole is."""
    ydim, xdim = ds.variables["latitude"].shape


def get_sensor_by_date(
    date: dt.date,
) -> str:
    """Return the sensor used for this date."""
    # TODO: these date ranges belong in a config location
    if date >= dt.date(2012, 7, 2) and date <= dt.date(2030, 12, 31):
        return "amsr2"
    else:
        raise RuntimeError(f"Could not determine sensor for date: {date}")


def get_surfacetype_da(
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
) -> xr.DataArray:
    """Return a dataarray with surface type information for this date."""
    sensor = get_sensor_by_date(date)

    if hemisphere == "north" and resolution == "12.5":
        surfgeo_ds = get_surfgeo_ds("psn12.5")
    elif hemisphere == "south" and resolution == "12.5":
        surfgeo_ds = get_surfgeo_ds("pss12.5")
    else:
        raise RuntimeError(
            f"""
        Could not determine grid for:
            hemisphere: {hemisphere}
            resolution: {resolution} ({type(resolution)})"""
        )

    xvar = surfgeo_ds.variables["x"]
    yvar = surfgeo_ds.variables["y"]
    surftypevar = surfgeo_ds.variables["surface_type"].copy()
    polehole_surface_type = 100
    if "polehole_bitmask" in surfgeo_ds.data_vars.keys():
        polehole_bitmask = surfgeo_ds.variables["polehole_bitmask"]
        polehole_bitlabel = f"{sensor}_polemask"
        polehole_index = (
            polehole_bitmask.attrs["flag_meanings"].split(" ").index(polehole_bitlabel)
        )
        polehole_bitvalue = polehole_bitmask.attrs["flag_values"][polehole_index]
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
