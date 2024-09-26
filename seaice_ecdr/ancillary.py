"""Code for interacting with ancillary data required by the ECDR.

Ancillary fields (e.g., land mask, invalid ice masks, l90c field, etc) required
for ECDR processing are stored in an ancillary NetCDF file that is published
alongside the ECDR.
"""

import datetime as dt
from functools import cache
from pathlib import Path
from typing import Literal, get_args

import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from pm_tb_data._types import NORTH, Hemisphere

from seaice_ecdr._types import ECDR_SUPPORTED_RESOLUTIONS
from seaice_ecdr.constants import CDR_ANCILLARY_DIR
from seaice_ecdr.grid_id import get_grid_id
from seaice_ecdr.platforms import PLATFORM_CONFIG, Platform
from seaice_ecdr.platforms.config import N07_PLATFORM

ANCILLARY_SOURCES = Literal["CDRv4", "CDRv5"]


def get_ancillary_filepath(
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ancillary_source: ANCILLARY_SOURCES,
) -> Path:
    grid_id = get_grid_id(
        hemisphere=hemisphere,
        resolution=resolution,
    )

    if ancillary_source == "CDRv5":
        filepath = CDR_ANCILLARY_DIR / f"ecdr-ancillary-{grid_id}.nc"
    elif ancillary_source == "CDRv4":
        filepath = CDR_ANCILLARY_DIR / f"ecdr-ancillary-{grid_id}-v04r00.nc"
    else:
        raise ValueError(f"Unknown ancillary source: {ancillary_source}")

    return filepath


@cache
def get_ancillary_ds(
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ancillary_source: ANCILLARY_SOURCES,
) -> xr.Dataset:
    """Return xr Dataset of ancillary data for this hemisphere/resolution."""
    # TODO: This list could be determined from an examination of
    #       the ancillary directory
    if resolution not in get_args(ECDR_SUPPORTED_RESOLUTIONS):
        raise NotImplementedError(
            "ECDR currently only supports {get_args(ECDR_SUPPORTED_RESOLUTIONS)} resolutions."
        )

    filepath = get_ancillary_filepath(
        hemisphere=hemisphere,
        resolution=resolution,
        ancillary_source=ancillary_source,
    )
    ds = xr.load_dataset(filepath)

    return ds


def bitmask_value_for_meaning(*, var: xr.DataArray, meaning: str):
    if meaning not in var.flag_meanings:
        raise ValueError(f"Could not determine bitmask value for {meaning=}")

    index = var.flag_meanings.split(" ").index(meaning)
    value = var.flag_masks[index]

    return value


def flag_value_for_meaning(*, var: xr.DataArray, meaning: str):
    if meaning not in var.flag_meanings:
        raise ValueError(f"Could not determine flag value for {meaning=}")

    index = var.flag_meanings.split(" ").index(meaning)
    value = var.flag_values[index]

    return value


def get_surfacetype_da(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ancillary_source: ANCILLARY_SOURCES,
) -> xr.DataArray:
    """Return a dataarray with surface type information for this date."""
    ancillary_ds = get_ancillary_ds(
        hemisphere=hemisphere,
        resolution=resolution,
        ancillary_source=ancillary_source,
    )

    xvar = ancillary_ds.variables["x"]
    yvar = ancillary_ds.variables["y"]
    surftypevar = ancillary_ds.variables["surface_type"].copy()
    polehole_surface_type = 100
    if "polehole_bitmask" in ancillary_ds.data_vars.keys():
        polehole_bitmask = ancillary_ds.polehole_bitmask
        platform = PLATFORM_CONFIG.get_platform_by_date(date)
        platform_id = platform.id
        # TODO: Use F17 polemask if F18 is being used. There is currently no F18
        # polemask defined in the ancilary file
        if platform.id == "F18":
            platform_id = "F17"
        polehole_bitlabel = f"{platform_id}_polemask"
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
        # TODO: get these attrs from ancillary file.
        attrs={
            "grid_mapping": "crs",
            "flag_values": surftype_flag_values_arr,
            "flag_meanings": surftype_flag_meanings_str,
            "standard_name": "area_type",
            "long_name": "Mask for ocean, lake, polehole, coast, and land areas",
            "comment": (
                "Note: Not all of the flag meanings derive from the current list of"
                " acceptable labels for area_type because there are area types in"
                " this field that are not present in that list."
            ),
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
    ancillary_source: ANCILLARY_SOURCES,
    platform: Platform | None = None,
) -> xr.DataArray:
    """Return the northern hemisphere pole hole mask for the given date and resolution."""
    ancillary_ds = get_ancillary_ds(
        hemisphere=NORTH,
        resolution=resolution,
        ancillary_source=ancillary_source,
    )

    polehole_bitmask = ancillary_ds.polehole_bitmask

    if platform is None:
        platform = PLATFORM_CONFIG.get_platform_by_date(
            date=date,
        )

    platform_id = platform.id
    # TODO: Use F17 polemask if F18 is being used. There is currently no F18
    # polemask defined in the ancilary file
    if platform.id == "F18":
        platform_id = "F17"
    polehole_bitlabel = f"{platform_id}_polemask"
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


# def get_smmr_ancillary_filepath(*, hemisphere, ancillary_source: ANCILLARY_SOURCES) -> Path:
def get_daily_ancillary_filepath(
    *, hemisphere, ancillary_source: ANCILLARY_SOURCES
) -> Path:
    """Return filepath to SMMR ancillary NetCDF.

    Contains a day-of-year climatology used for SMMR.
    """
    grid_id = get_grid_id(
        hemisphere=hemisphere,
        # Hard-coded to 25km resolution, which is what we expect for SMMR.
        resolution="25",
    )

    if ancillary_source == "CDRv5":
        filepath = CDR_ANCILLARY_DIR / f"ecdr-ancillary-{grid_id}-smmr-invalid-ice.nc"
    elif ancillary_source == "CDRv4":
        filepath = (
            CDR_ANCILLARY_DIR / f"ecdr-ancillary-{grid_id}-smmr-invalid-ice-v04r00.nc"
        )
    else:
        raise ValueError(f"Unknown smmr ancillary source: {ancillary_source}")

    return filepath


def get_smmr_invalid_ice_mask(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ancillary_source: ANCILLARY_SOURCES,
) -> xr.DataArray:
    # TODO: Consider using daily instead of monthly icemask for SMMR?  Others?
    # ancillary_file = get_daily_ancillary_filepath(hemisphere=hemisphere, ancillary_source=ancillary_source)
    ancillary_file = get_ancillary_filepath(
        hemisphere=hemisphere,
        resolution=resolution,
        ancillary_source=ancillary_source,
    )

    with xr.open_dataset(ancillary_file) as ds:
        invalid_ice_mask = ds.invalid_ice_mask.copy().astype(bool)

    if "month" in invalid_ice_mask.variable.dims:
        # Monthly ice mask
        month = date.month

        icemask_for_date = invalid_ice_mask.sel(month=month)
        icemask_for_date = icemask_for_date.drop_vars("month")
    elif "doy" in invalid_ice_mask.variable.dims:
        # day-of-year (doy) ice mask
        doy = date.timetuple().tm_yday
        icemask_for_date = invalid_ice_mask.sel(doy=doy)

        # Drop the DOY dim. This is consistent with the other case returned by
        # `get_invalid_ice_mask`, which has `month` as a dimension instead.
        icemask_for_date = icemask_for_date.drop_vars("doy")

    # Ice mask needs to be boolean
    icemask_for_date = icemask_for_date.astype("bool")

    return icemask_for_date


def get_invalid_ice_mask(
    *,
    hemisphere: Hemisphere,
    date: dt.date,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ancillary_source: ANCILLARY_SOURCES,
    platform: Platform,
) -> xr.DataArray:
    """Return an invalid ice mask for the given date.

    SMMR (n07) uses a day-of-year based climatology. All other platforms use a
    month-based mask.
    """
    # SMMR / n07 case:
    if platform == N07_PLATFORM:
        # TODO: Daily (SMMR) mask is used at end for cleanup,
        #       but not for initial TB field generation
        # Skip the smmr invalid ice mask for now...
        print("WARNING: Using non-SMMR invalid ice masks")
        # return get_smmr_invalid_ice_mask(hemisphere=hemisphere, date=date)
    # All other platforms:
    ancillary_ds = get_ancillary_ds(
        hemisphere=hemisphere,
        resolution=resolution,
        ancillary_source=ancillary_source,
    )

    invalid_ice_mask = ancillary_ds.invalid_ice_mask.sel(month=date.month)

    # The invalid ice mask is indexed by month in the ancillary dataset. Drop
    # that coordinate.
    invalid_ice_mask = invalid_ice_mask.drop_vars("month")

    # xarray does not currently permit netCDF files with boolean data type
    # so this array must be explicitly cast as boolean upon being read in.
    invalid_ice_mask = invalid_ice_mask.astype("bool")

    return invalid_ice_mask


def get_ocean_mask(
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ancillary_source: ANCILLARY_SOURCES,
) -> xr.DataArray:
    """Return a binary mask where True values represent `ocean`.

    This mask includes the polehole for NH.
    """
    ancillary_ds = get_ancillary_ds(
        hemisphere=hemisphere,
        resolution=resolution,
        ancillary_source=ancillary_source,
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


def get_non_ocean_mask(
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ancillary_source: ANCILLARY_SOURCES,
) -> xr.DataArray:
    """Return a binary mask where True values represent non-ocean pixels.

    This mask includes land, coast, and lake.
    """
    ancillary_ds = get_ancillary_ds(
        hemisphere=hemisphere,
        resolution=resolution,
        ancillary_source=ancillary_source,
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
    lake_val = flag_value_for_meaning(
        var=surface_type,
        meaning="lake",
    )

    non_ocean_mask = (
        (surface_type == land_val)
        | (surface_type == coast_val)
        | (surface_type == lake_val)
    )

    non_ocean_mask.attrs = dict(
        grid_mapping="crs",
        long_name="non-ocean mask",
        comment="Mask indicating where non-ocean is",
        units="1",
    )

    non_ocean_mask.encoding = dict(zlib=True)

    return non_ocean_mask


def get_land90_conc_field(
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ancillary_source: ANCILLARY_SOURCES,
) -> xr.DataArray:
    ancillary_ds = get_ancillary_ds(
        hemisphere=hemisphere,
        resolution=resolution,
        ancillary_source=ancillary_source,
    )

    land90_da = ancillary_ds.l90c

    return land90_da


def get_adj123_field(
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ancillary_source: ANCILLARY_SOURCES,
) -> xr.DataArray:
    ancillary_ds = get_ancillary_ds(
        hemisphere=hemisphere,
        resolution=resolution,
        ancillary_source=ancillary_source,
    )

    adj123_da = ancillary_ds.adj123

    return adj123_da


def get_nt_landmask(
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ancillary_source: ANCILLARY_SOURCES,
) -> npt.NDArray:
    """Returns a numpy array equivalent to that used in the original
    NT code, particularly for the NT land spillover algorithm."""

    ancillary_ds = get_ancillary_ds(
        hemisphere=hemisphere,
        resolution=resolution,
        ancillary_source=ancillary_source,
    )
    if "cdrv4_landmask" in ancillary_ds.variables.keys():
        return np.array(ancillary_ds.variables["cdrv4_nt_landmask"])

    # If a landmask field like that used in CDRv4 is not available in
    # the ancillary field, create it
    raise RuntimeError("gen_cdrv4_nt_landmask() not yet implemented.")
    # cdrv4_nt_landmask = gen_cdrv4_nt_landmask(
    #     ancillary_ds=ancillary_ds,
    # )

    # return cdrv4_nt_landmask


def get_nt_shoremap(
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ancillary_source: ANCILLARY_SOURCES,
) -> npt.NDArray:
    """Returns a numpy array equivalent to that used in the original
    NT code, particularly for the NT land spillover algorithm."""

    ancillary_ds = get_ancillary_ds(
        hemisphere=hemisphere,
        resolution=resolution,
        ancillary_source=ancillary_source,
    )
    if "cdrv4_nt_shoremap" in ancillary_ds.variables.keys():
        return np.array(ancillary_ds.variables["cdrv4_nt_shoremap"])

    # If a shoremap field like that used in CDRv4 is not available in
    # the ancillary field, create it
    raise RuntimeError("gen_cdrv4_nt_shoremap() not yet implemented.")
    # cdrv4_nt_shoremap = gen_cdrv4_nt_shoremap(
    #     ancillary_ds=ancillary_ds,
    # )

    # return cdrv4_nt_shoremap


def get_nt_minic(
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ancillary_source: ANCILLARY_SOURCES,
) -> npt.NDArray:
    """Returns a numpy array equivalent to that used in the original
    NT code, particularly for the NT land spillover algorithm."""

    ancillary_ds = get_ancillary_ds(
        hemisphere=hemisphere,
        resolution=resolution,
        ancillary_source=ancillary_source,
    )
    if "cdrv4_nt_minic" in ancillary_ds.variables.keys():
        return np.array(ancillary_ds.variables["cdrv4_nt_minic"])

    # If a minic field like that used in CDRv4 is not available in
    # the ancillary field, create it
    raise RuntimeError("gen_cdrv4_nt_minic() not yet implemented.")
    # cdrv4_nt_minic = gen_cdrv4_nt_minic(
    #     ancillary_ds=ancillary_ds,
    # )

    # return cdrv4_nt_minic


def get_empty_ds_with_time(
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    date: dt.date,
    ancillary_source: ANCILLARY_SOURCES,
) -> xr.Dataset:
    """Return an "empty" xarray dataset with x, y, crs, and time set."""
    ancillary_ds = get_ancillary_ds(
        hemisphere=hemisphere,
        resolution=resolution,
        ancillary_source=ancillary_source,
    )

    time_as_int = (date - dt.date(1970, 1, 1)).days
    time_da = xr.DataArray(
        name="time",
        data=[int(time_as_int)],
        dims=["time"],
        attrs={
            "standard_name": "time",
            "long_name": "ANSI date",
            "calendar": "standard",
            "axis": "T",
            "units": "days since 1970-01-01",
            "coverage_content_type": "coordinate",
            "valid_range": [int(0), int(30000)],
        },
    )

    return_ds = xr.Dataset(
        data_vars=dict(
            x=ancillary_ds.x,
            y=ancillary_ds.y,
            crs=ancillary_ds.crs,
            time=time_da,
        ),
    )

    return return_ds
