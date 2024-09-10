"""Routines for generating temporally composited file.

"""

import copy
import datetime as dt
from functools import cache
from pathlib import Path
from typing import Iterable, cast, get_args

import click
import numpy as np
import numpy.typing as npt
import xarray as xr
from loguru import logger
from pm_icecon.fill_polehole import fill_pole_hole
from pm_tb_data._types import NORTH, Hemisphere

from seaice_ecdr._types import ECDR_SUPPORTED_RESOLUTIONS
from seaice_ecdr.ancillary import (
    ANCILLARY_SOURCES,
    get_non_ocean_mask,
    nh_polehole_mask,
)
from seaice_ecdr.cli.util import datetime_to_date
from seaice_ecdr.constants import DEFAULT_BASE_OUTPUT_DIR
from seaice_ecdr.initial_daily_ecdr import (
    create_idecdr_for_date,
    get_idecdr_filepath,
)
from seaice_ecdr.platforms import PLATFORM_CONFIG
from seaice_ecdr.spillover import LAND_SPILL_ALGS
from seaice_ecdr.util import (
    date_range,
    get_intermediate_output_dir,
    standard_daily_filename,
)


def yield_dates_from_temporal_interpolation_flags(
    ref_date: dt.date,
    ti_flags: np.ndarray,
):
    """A generator for dates used in temporal interpolation per the flags."""
    date_offset_list = []
    ti_flag_set = np.unique(ti_flags)
    for flag_val in np.nditer(ti_flag_set):
        tens_val = np.floor_divide(flag_val, 10)
        if tens_val < 10:
            date_offset_list.append(ref_date - dt.timedelta(days=int(tens_val)))
        ones_val = np.mod(flag_val, 10)
        if ones_val < 10:
            date_offset_list.append(ref_date + dt.timedelta(days=int(ones_val)))

    date_offset_set = set(date_offset_list)
    for date in sorted(date_offset_set):
        yield date


def temporally_interpolate_dataarray_using_flags(
    ref_date: dt.date, data_array: xr.DataArray, ti_flags: np.ndarray
) -> np.ndarray:
    """Yield the temporal interpolation of a data cube using flag values.

    The tens-place-value of the temporal_interpolation_flag value is the number
    of days in the past to draw the value, the ones-place-value is the number
    of days in the future to use for interpolation."""

    initial_value = 255
    tdim, ydim, xdim = data_array.shape
    prior_val = np.zeros((ydim, xdim), dtype=float)
    prior_val[:] = initial_value
    prior_dist = np.zeros((ydim, xdim), dtype=int)
    next_val = np.zeros((ydim, xdim), dtype=float)
    next_val[:] = initial_value
    next_dist = np.zeros((ydim, xdim), dtype=int)

    for date in data_array.time:
        date_offset = (date.data - ref_date).days

        da_slice = np.squeeze(data_array.isel(time=data_array.time == date).copy().data)
        if date_offset < 0:
            is_this_prior = (ti_flags // 10) == -date_offset
            prior_val[is_this_prior] = da_slice[is_this_prior]
            prior_dist[is_this_prior] = -date_offset
            next_val[is_this_prior] = da_slice[is_this_prior]
            next_dist[is_this_prior] = -date_offset
        elif date_offset > 0:
            is_this_next = (ti_flags % 10) == date_offset
            next_val[is_this_next] = da_slice[is_this_next]
            next_dist[is_this_next] = date_offset
        else:
            # date_offset is zero
            is_this = ti_flags == 0
            prior_val[is_this] = da_slice[is_this]
            prior_dist[is_this] = 0
            next_val[is_this] == da_slice[is_this]
            next_dist[is_this] = 0

    # Fill in where there was only one side to the interpolation
    next_val[next_val == initial_value] = prior_val[next_val == initial_value]
    prior_val[prior_val == initial_value] = next_val[prior_val == initial_value]

    # Linearly interpolate between prior and next values
    linint_rise = next_val.astype(np.float32) - prior_val.astype(np.float32)
    linint_run = prior_dist + next_dist
    linint_run[linint_run == 0] = 1  # avoid div by zero

    filled_array = prior_val + prior_dist * linint_rise / linint_run
    if data_array.dtype == np.uint8:
        filled_array = np.round(filled_array).astype(np.uint8)
    else:
        filled_array = filled_array.astype(data_array.dtype)

    return filled_array


@cache
def get_tie_dir(*, intermediate_output_dir: Path, hemisphere: Hemisphere) -> Path:
    """Daily complete output dir for TIE processing"""
    tie_dir = intermediate_output_dir / "temporal_interp"
    tie_dir.mkdir(parents=True, exist_ok=True)

    return tie_dir


def get_tie_filepath(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    intermediate_output_dir: Path,
) -> Path:
    """Return the complete daily tie file path."""

    platform = PLATFORM_CONFIG.get_platform_by_date(date)
    platform_id = platform.id

    standard_fn = standard_daily_filename(
        hemisphere=hemisphere,
        date=date,
        platform_id=platform_id,
        resolution=resolution,
    )
    # Add `tiecdr` to the beginning of the standard name to distinguish it as a
    # WIP.
    tie_filename = "tiecdr_" + standard_fn
    tie_dir = get_tie_dir(
        intermediate_output_dir=intermediate_output_dir, hemisphere=hemisphere
    )

    tie_filepath = tie_dir / tie_filename

    return tie_filepath


def iter_dates_near_date(
    target_date: dt.date,
    day_range: int = 0,
    skip_future: bool = False,
    date_step: int = 1,
):
    """Return iterator of dates near a given date.

    This routine aids in the construction of a DataArray for use in
    temporally_composite_dataarray().  It provides a series of dates
    around a "seed" date.  The temporal compositing process fills in
    data missing at a target date with data from a few days before and
    (optionally) after that date.  The number of days away from the
    target date is the day_range.  if skip_future is True, then only
    dates prior to the target date are provided.  This is suitable for
    near-real-time use, because data from "the future" are not available.
    """
    earliest_date = target_date - dt.timedelta(days=day_range)
    beginning_of_platform_coverage = PLATFORM_CONFIG.get_first_platform_start_date()
    if earliest_date < beginning_of_platform_coverage:
        logger.warning(
            f"Resetting temporal interpolation earliest date from {earliest_date} to {beginning_of_platform_coverage}"
        )
        earliest_date = beginning_of_platform_coverage

    if skip_future:
        latest_date = target_date
    else:
        latest_date = target_date + dt.timedelta(days=day_range)

    date = earliest_date
    while date <= latest_date:
        yield date
        date += dt.timedelta(days=date_step)


def is_seaice_conc(
    field: npt.NDArray,
) -> npt.NDArray:
    """Returns boolean index array where field has values 0-100."""
    where_seaice = (field >= 0) & (field <= 100)

    return where_seaice


def temporally_composite_dataarray(
    *,
    target_date: dt.date,
    da: xr.DataArray,
    interp_range: int = 5,
    one_sided_limit: int = 3,
    still_missing_flag: int = 255,
    non_ocean_mask: xr.DataArray,
    daily_climatology_mask: None | npt.NDArray = None,
) -> tuple[xr.DataArray, npt.NDArray]:
    """Temporally composite a DataArray referenced to given reference date
    up to interp_range days.

    target_date is the date that we are temporally filling with this routine
    da is an xr.DataArray that has all the fields we might use in the interp
    interp_range is the number of days forward and back that we are willing
    to look to do two-sided -- meaning we need data from both prior and
    subsequent ("next") days in order to do the interpolation.
    one_sided_limit is the max number of days we are willing to look in only
    one direction.  It will generally (always?) be less than the interp_range.
    """
    logger.debug(f"Temporally compositing {da.name} dataarray around {target_date}")
    # Our flag system requires that the value be expressible by no more than
    # nine days in either direction
    if interp_range > 9:
        interp_range_error_message = (
            f"interp_range in {__name__} is > 9: {interp_range}."
            "  This value must be <= 9 in order to be expressable in the "
            " temporal_flags field."
        )
        logger.error(interp_range_error_message)
        raise RuntimeError(interp_range_error_message)

    _, ydim, xdim = da.shape

    temp_comp_da = da.isel(time=da.time.dt.date == target_date).copy()

    if temp_comp_da.size == 0:
        # This time was not in the time slice; need to init to all-missing
        # TODO: Do we need dims, coords, attrs here?
        # temp_comp_da = zeros_xrdataarray(xdim, ydim, np.nan)
        raise RuntimeError(f" the target_date was not in the dataarray: {target_date}")

    temp_comp_2d = np.squeeze(temp_comp_da.data)
    assert temp_comp_2d.shape == (ydim, xdim)

    # TODO:  These lines are commented out in order to reproduce the
    #        CDRv4 ERROR where the "home" smmr day does NOT have daily_clim
    #        applied to it.
    # if daily_climatology_mask is not None:
    #    temp_comp_2d[daily_climatology_mask] = 0

    # Initialize arrays
    initial_missing_locs = np.isnan(temp_comp_2d.data)

    pconc = np.zeros_like(temp_comp_2d)
    pdist = np.zeros((ydim, xdim), dtype=np.uint8)

    nconc = np.zeros_like(temp_comp_2d)
    ndist = np.zeros((ydim, xdim), dtype=np.uint8)

    need_values = initial_missing_locs.copy()
    n_missing = np.sum(np.where(need_values, 1, 0))

    pconc[need_values] = 0
    pdist[need_values] = 0
    nconc[need_values] = 0
    # TODO: Fix this for v5 release (implemented to match v04f00)
    # ndist[need_values] = 0  # Correct
    pdist[need_values] = 0  # Error as CDRv04r00 error

    for time_offset in range(1, interp_range + 1):
        if n_missing == 0:
            continue

        prior_date = target_date - dt.timedelta(days=time_offset)
        next_date = target_date + dt.timedelta(days=time_offset)

        prior_field = np.squeeze(da.isel(time=da.time.dt.date == prior_date).to_numpy())
        next_field = np.squeeze(da.isel(time=da.time.dt.date == next_date).to_numpy())
        if daily_climatology_mask is not None:
            prior_field[daily_climatology_mask] = 0
            next_field[daily_climatology_mask] = 0

        # update prior arrays
        n_prior = prior_field.size
        if n_prior != 0:
            need_prior = need_values & (pdist == 0)
            have_prior = is_seaice_conc(prior_field) & need_prior
            pconc[have_prior] = prior_field[have_prior]
            pdist[have_prior] = time_offset

        # update next arrays
        n_next = next_field.size
        if n_next != 0:
            need_next = need_values & (ndist == 0)
            have_next = is_seaice_conc(next_field) & need_next
            nconc[have_next] = next_field[have_next]
            ndist[have_next] = time_offset

        # Update still-missing arrays
        need_values = initial_missing_locs & ((pdist == 0) | (ndist == 0))
        n_missing = np.sum(np.where(need_values, 1, 0))

    # NOTE: Need to update the QA flag where temporal interpolation occurred

    # Temporal flag field will get filled with:
    #   0 where values already exist and don't need to be filled (inc land)
    #   pconc/nconc/linearly-interpolsted if have prior/next values
    #   otherwise, defaults to "still_missing_flag" value
    temporal_flags = np.zeros(temp_comp_2d.shape, dtype=np.uint8)
    temporal_flags[initial_missing_locs] = still_missing_flag

    # *** If we have values from BOTH prior and subsequent days, interpolate
    have_both_prior_and_next = (pdist > 0) & (ndist > 0)
    # *** If we only have values from prior days,
    #     and within <one_sided_limit> days, use it
    have_only_prior = (pdist > 0) & (pdist <= one_sided_limit) & (ndist == 0)
    # *** If we only have values from subsequent days,
    #     and within <one_sided_limit> days, use it
    have_only_next = (ndist > 0) & (ndist <= one_sided_limit) & (pdist == 0)

    linint_rise = nconc.astype(np.float32) - pconc.astype(np.float32)
    linint_run = pdist + ndist
    linint_run[linint_run == 0] = 1  # avoid div by zero
    linint = pconc + pdist * linint_rise / linint_run
    temp_comp_2d[have_both_prior_and_next] = linint[have_both_prior_and_next]

    # Update the temporal interp flag value
    temporal_flags[have_both_prior_and_next] = (
        10 * pdist[have_both_prior_and_next] + ndist[have_both_prior_and_next]
    )

    temp_comp_2d[have_only_prior] = pconc[have_only_prior]

    # Update the temporal interp flag value
    temporal_flags[have_only_prior] = 10 * pdist[have_only_prior]

    # temp_comp_2d[have_only_next] = nconc[have_only_next]  # Correct
    temp_comp_2d[have_only_next] = pconc[have_only_next]  # Error as CDRv04r00

    # Update the temporal interp flag value
    # temporal_flags[have_only_next] = ndist[have_only_next]  # Correct
    temporal_flags[have_only_next] = pdist[have_only_next]  # Error as CDRv04r00

    # Ensure flag values do not occur over land
    temporal_flags[non_ocean_mask.data] = 0

    temp_comp_da.data[0, :, :] = temp_comp_2d[:, :]

    return temp_comp_da, temporal_flags


# TODO: this function also belongs in the intial daily ecdr module.
def read_or_create_and_read_idecdr_ds(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    intermediate_output_dir: Path,
    land_spillover_alg: LAND_SPILL_ALGS,
    ancillary_source: ANCILLARY_SOURCES,
    overwrite_ide: bool = False,
) -> xr.Dataset:
    """Read an idecdr netCDF file, creating it if it doesn't exist."""
    platform = PLATFORM_CONFIG.get_platform_by_date(
        date,
    )

    ide_filepath = get_idecdr_filepath(
        date=date,
        platform_id=platform.id,
        hemisphere=hemisphere,
        resolution=resolution,
        intermediate_output_dir=intermediate_output_dir,
    )
    if overwrite_ide or not ide_filepath.is_file():
        create_idecdr_for_date(
            date=date,
            hemisphere=hemisphere,
            resolution=resolution,
            intermediate_output_dir=intermediate_output_dir,
            land_spillover_alg=land_spillover_alg,
            ancillary_source=ancillary_source,
        )
    logger.debug(f"Reading ideCDR file from: {ide_filepath}")
    ide_ds = xr.load_dataset(ide_filepath)

    return ide_ds


def calc_stddev_field(
    bt_conc,
    nt_conc,
    min_valid_value,
    max_valid_value,
    fill_value,
) -> xr.DataArray:
    """Compute std dev field for cdr_conc value using BT and NT fields.

    This value is the standard deviation of a given grid cell along with
    its eight surrounding grid cells (for nine values total) from both
    the NASA Team and Bootstrap data fields.

    This means that the standard deviation is computed using a total
    of 18 values: nine from the BT and nine from tne NT field.

    TODO: This could be generalized to n-fields, instead of 2.
    """

    bt_conc_masked = np.ma.masked_outside(
        bt_conc,
        min_valid_value,
        max_valid_value,
    )
    nt_conc_masked = np.ma.masked_outside(
        nt_conc,
        min_valid_value,
        max_valid_value,
    )

    # Initialize the aggregation sum and count arrays
    ydim, xdim = bt_conc.shape
    agg_array = np.ma.empty((18, ydim, xdim), dtype=np.float64)
    agg_count = np.ma.zeros((ydim, xdim), dtype=np.int64)

    # Use rolled arrays to add first bt, then nt to aggregation arrays
    agg_idx = 0
    for yoff in range(-1, 2):
        for xoff in range(-1, 2):
            rolled_array = np.roll(bt_conc_masked, (yoff, xoff), (0, 1))
            agg_array[agg_idx, :, :] = rolled_array[:, :]
            agg_count[~np.isnan(rolled_array)] += 1
            agg_idx += 1
    for yoff in range(-1, 2):
        for xoff in range(-1, 2):
            rolled_array = np.roll(nt_conc_masked, (yoff, xoff), (0, 1))
            agg_array[agg_idx, :, :] = rolled_array[:, :]
            agg_count[~np.isnan(rolled_array)] += 1
            agg_idx += 1

    stddev_raw = np.ma.filled(
        agg_array.std(axis=0, ddof=1).astype(np.float32),
        fill_value=-1,
    )

    stddev = np.ma.empty_like(bt_conc_masked, dtype=np.float32)
    stddev[:] = stddev_raw[:]

    # Mask any locations with insufficient count
    stddev[(agg_count >= 0) & (agg_count < 6)] = fill_value

    # Mask out any calculated missing values
    stddev[stddev == -1] = fill_value

    stddev[0, :] = fill_value
    stddev[-1, :] = fill_value
    stddev[:, 0] = fill_value
    stddev[:, -1] = fill_value

    return stddev


def filter_field_via_bitmask(
    field_da: xr.DataArray,
    flag_da: xr.DataArray,
    filter_ids: list,
) -> xr.DataArray:
    """Apply filters identified by flag_meaning to a field.

    Currently, application of the filter means set-the-value-to-zero.
    """
    output_da = field_da.copy()
    assert filter_ids is not None  # this simply preserves this arg for use

    bitmasks_to_filter = [1, 2, 16]
    for bitmask in bitmasks_to_filter:
        is_to_filter = np.bitwise_and(flag_da.data, bitmask).astype(bool)
        output_da = output_da.where(
            ~is_to_filter,
            other=0,
        )

    return output_da


def get_daily_climatology_mask(
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ancillary_source: ANCILLARY_SOURCES,
) -> None | npt.NDArray:
    """
    Given the date and ancillary source, return a mask where True values
    indicate that the sea ice conc values should be set to zero

    NOTE: The date range for this is hard-coded to correspond to SMMR.
          This should probably be an argument somehow?

    Because the day-of-year climatology includes the land mask as part of
    the invalid ice mask, we get the non_ocean_mask to dis-convolve that.
    (We don't necessarily want to set the land to sea-ice-conc=0.)
    """
    from netCDF4 import Dataset

    # This date comes from PLATFORM_AVAILABILITY in platforms.py
    # TODO: This should be refactored to have less hard-coding!
    if date > dt.date(1987, 7, 9):
        return None

    non_ocean_mask = get_non_ocean_mask(
        hemisphere=hemisphere,
        resolution=resolution,
        ancillary_source=ancillary_source,
    )

    daily_ds = None
    if ancillary_source == "CDRv4":
        if hemisphere == "north":
            daily_ds = Dataset(
                "/share/apps/G02202_V5/v05r01_ancillary/ecdr-ancillary-psn25-smmr-invalid-ice-v04r00.nc"
            )
        elif hemisphere == "south":
            daily_ds = Dataset(
                "/share/apps/G02202_V5/v05r01_ancillary/ecdr-ancillary-pss25-smmr-invalid-ice-v04r00.nc"
            )
    else:
        if hemisphere == "north":
            daily_ds = Dataset(
                "/share/apps/G02202_V5/v05r01_ancillary/ecdr-ancillary-psn25-smmr-invalid-ice-v05r01.nc"
            )
        if hemisphere == "south":
            daily_ds = Dataset(
                "/share/apps/G02202_V5/v05r01_ancillary/ecdr-ancillary-pss25-smmr-invalid-ice-v05r01.nc"
            )

    if daily_ds is None:
        raise RuntimeError(
            f"Could not load daily_climatology mask for {date=} {hemisphere=} {resolution=} {ancillary_source=}"
        )

    # day-of-year index is doy - 1
    doy_index = int(date.strftime("%j")) - 1
    doy_mask_invalid = np.array(daily_ds.variables["invalid_ice_mask"])[doy_index, :, :]

    # Return mask of invalid seaice, excluding land
    mask = (doy_mask_invalid != 0) & (~non_ocean_mask.data)

    return mask


# TODO: better function name and docstring. This is first pass at refactor.
def temporal_interpolation(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    data_stack: xr.Dataset,
    ancillary_source: ANCILLARY_SOURCES,
    fill_the_pole_hole: bool = True,
    interp_range: int = 5,
    one_sided_limit: int = 3,
) -> xr.Dataset:
    # Initialize a new xr "temporally interpolated CDR" (tiecdr) dataset. The
    # target date is used to initialize this dataset. We retain the `time`
    # dimension by specifying `drop=False` and using list notation when using
    # `sel`.
    tie_ds = data_stack.sel(
        time=[dt.datetime(date.year, date.month, date.day)], drop=False
    ).copy(deep=True)
    # The `crs` variable doesn't need a time dim, so drop here.
    tie_ds["crs"] = tie_ds.crs.isel(time=0).drop_vars("time")

    # Update the cdr_conc var with temporally interpolated cdr_conc field
    #   by creating a DataArray with conc fields +/- interp_range around date
    non_ocean_mask = get_non_ocean_mask(
        hemisphere=hemisphere,
        resolution=resolution,
        ancillary_source=ancillary_source,
    )
    # daily_climatology_mask is True where historically no sea ice.
    #   It can be None if no daily_climatology mask is to be used
    daily_climatology_mask = get_daily_climatology_mask(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        ancillary_source=ancillary_source,
    )

    # Actually compute the cdr_conc temporal composite
    ti_var, ti_flags = temporally_composite_dataarray(
        target_date=date,
        da=data_stack.conc,
        interp_range=interp_range,
        non_ocean_mask=non_ocean_mask,
        one_sided_limit=one_sided_limit,
        daily_climatology_mask=daily_climatology_mask,
    )

    tie_ds["cdr_conc_ti"] = ti_var

    # Update QA flag field
    is_temporally_interpolated = (ti_flags > 0) & (ti_flags <= 55)
    # TODO: this bit mask of 64 added to (equals bitwise "or")
    #       should be looked up from a map of flag mask values
    tie_ds["qa_of_cdr_seaice_conc"] = tie_ds["qa_of_cdr_seaice_conc"].where(
        ~is_temporally_interpolated,
        other=np.bitwise_or(tie_ds["qa_of_cdr_seaice_conc"].data, 64),
    )

    # Add the temporal interp flags to the dataset
    tie_ds["temporal_interpolation_flag"] = (
        ("time", "y", "x"),
        np.expand_dims(ti_flags, axis=0),
        {
            "grid_mapping": "crs",
            "standard_name": "status_flag",
            "valid_range": [np.uint8(0), np.uint8(255)],
            "comment": (
                "Value of 0 indicates no temporal interpolation occurred."
                "  Values greater than 0 and less than 100 are of the form"
                ' "AB" where "A" indicates the number of days prior to the'
                ' current day and "B" indicates the number of days after'
                " the current day used to linearly interpolate the data."
                "  If either A or B are zero, the value was extrapolated"
                " from that date rather than interpolated.  A value of 255"
                " indicates that temporal interpolation could not be"
                " accomplished."
            ),
        },
        {
            "zlib": True,
        },
    )

    cdr_conc = np.squeeze(tie_ds["cdr_conc_ti"].data)
    # TODO: May want to rename this field.  Specifically, after this
    #       operation, this will be both temporally interpolated and
    #       polehole-filled (if appropriate).  For now, "cdr_conc" is okay
    tie_ds["cdr_conc"] = tie_ds["cdr_conc_ti"].copy()

    # TODO: This is a really coarse way of determining which
    #       grid is having its pole hole filled!
    if fill_the_pole_hole and hemisphere == NORTH:
        cdr_conc_pre_polefill = cdr_conc.copy()
        near_pole_hole_mask = nh_polehole_mask(
            date=date,
            resolution=resolution,
            ancillary_source=ancillary_source,
        )
        cdr_conc_pole_filled = fill_pole_hole(
            conc=cdr_conc,
            near_pole_hole_mask=near_pole_hole_mask.data,
        )
        logger.debug("Filled pole hole")
        # Need to use not-isnan() here because NaN == NaN evaluates to False
        is_pole_filled = (cdr_conc_pole_filled != cdr_conc_pre_polefill) & (
            ~np.isnan(cdr_conc_pole_filled)
        )

        if "spatial_interpolation_flag" not in tie_ds.variables.keys():
            raise RuntimeError("Spatial interpolation flag not found in tie_ds.")

        # TODO: These are constants for the eCDR runs.  They should
        #       NOT be defined here (and in the idecdr code...(!))
        # TODO Actually, if this is defined here, the 'pole_filled'
        #      bitmask value should be determined by examining the
        #      bitmask_flags and bitmask_flag_meanings fields of the
        #      DataArray variable.
        TB_SPATINT_BITMASK_MAP = {
            "v19": 1,
            "h19": 2,
            "v22": 4,
            "v37": 8,
            "h37": 16,
            "pole_filled": 32,
        }
        tie_ds["spatial_interpolation_flag"] = tie_ds[
            "spatial_interpolation_flag"
        ].where(
            ~is_pole_filled,
            other=TB_SPATINT_BITMASK_MAP["pole_filled"],
        )

        logger.debug("Updated spatial_interpolation with pole hole value")

        tie_ds["cdr_conc"].data[0, :, :] = cdr_conc_pole_filled[:, :]
    else:
        # TODO: May want to modify attributes of the cdr_conc field to
        #       distinguish it from the cdr_conc_ti field
        pass

    # NOTE: the bt_conc array does not have daily_climatology applied
    bt_conc, _ = temporally_composite_dataarray(
        target_date=date,
        da=data_stack.raw_bt_seaice_conc,
        interp_range=interp_range,
        non_ocean_mask=non_ocean_mask,
        one_sided_limit=one_sided_limit,
    )

    # NOTE: the nt_conc array does not have daily_climatology applied
    nt_conc, _ = temporally_composite_dataarray(
        target_date=date,
        da=data_stack.raw_nt_seaice_conc,
        interp_range=interp_range,
        non_ocean_mask=non_ocean_mask,
        one_sided_limit=one_sided_limit,
    )

    # Note: this pole-filling code is copy-pasted from the cdr_conc
    #       methodology above
    if fill_the_pole_hole and hemisphere == NORTH:
        bt_conc_2d = np.squeeze(bt_conc.data)
        nt_conc_2d = np.squeeze(nt_conc.data)

        # Fill pole hole of BT
        bt_conc_pre_polefill = bt_conc_2d.copy()
        near_pole_hole_mask = nh_polehole_mask(
            date=date,
            resolution=resolution,
            ancillary_source=ancillary_source,
        )
        bt_conc_pole_filled = fill_pole_hole(
            conc=bt_conc_2d,
            near_pole_hole_mask=near_pole_hole_mask.data,
        )
        logger.debug("Filled pole hole (bt)")
        # Need to use not-isnan() here because NaN == NaN evaluates to False
        is_pole_filled = (bt_conc_pole_filled != bt_conc_pre_polefill) & (
            ~np.isnan(bt_conc_pole_filled)
        )
        bt_conc.data[0, :, :] = bt_conc_pole_filled[:, :]

        # Fill pole hole of NT
        nt_conc_pre_polefill = nt_conc_2d.copy()
        nt_conc_pole_filled = fill_pole_hole(
            conc=nt_conc_2d,
            near_pole_hole_mask=near_pole_hole_mask.data,
        )
        logger.debug("Filled pole hole (nt)")
        # Need to use not-isnan() here because NaN == NaN evaluates to False
        is_pole_filled = (nt_conc_pole_filled != nt_conc_pre_polefill) & (
            ~np.isnan(nt_conc_pole_filled)
        )
        nt_conc.data[0, :, :] = nt_conc_pole_filled[:, :]

        # TODO: I noticed that NT raw conc here can be > 100 (!)
        #       So for stdev calc, clamp to 100%
        nt_conc = nt_conc.where(
            nt_conc < 100,
            other=100,
        )

    stddev_field = calc_stddev_field(
        bt_conc=bt_conc.data[0, :, :],
        nt_conc=nt_conc.data[0, :, :],
        min_valid_value=0,
        max_valid_value=100,
        fill_value=-1,
    )

    # Set this to a data array
    tie_ds["stdev_of_cdr_seaice_conc_raw"] = (
        ("time", "y", "x"),
        np.expand_dims(stddev_field.data, axis=0),
        {
            "_FillValue": -1,
            "long_name": (
                "Passive Microwave Daily Sea Ice Concentration",
                " Source Estimated Standard Deviation",
            ),
            "grid_mapping": "crs",
            "valid_range": np.array((0, 300), dtype=np.float32),
            "units": 1,
        },
        {
            "zlib": True,
        },
    )

    # TODO: This should be moved to a CONSTANTS or configuration location
    filter_flags_to_apply = [
        "BT_weather_filter_applied",
        "NT_weather_filter_applied",
        "invalid_ice_mask_applied",
    ]

    stdev_field_filtered = filter_field_via_bitmask(
        field_da=tie_ds["stdev_of_cdr_seaice_conc_raw"],
        flag_da=tie_ds["qa_of_cdr_seaice_conc"],
        filter_ids=filter_flags_to_apply,
    )

    # set non-conc values to -1
    is_non_siconc = tie_ds["cdr_conc"].data > 1
    stdev_field_filtered = stdev_field_filtered.where(
        ~is_non_siconc,
        other=-1,
    )

    # Re-set the stdev data array...
    # Note: probably need to set land values to -1 here?
    tie_ds["stdev_of_cdr_seaice_conc"] = (
        ("time", "y", "x"),
        stdev_field_filtered.data,
        {
            "_FillValue": -1,
            "long_name": (
                "Passive Microwave Daily Sea Ice Concentration",
                " Source Estimated Standard Deviation",
            ),
            "grid_mapping": "crs",
            "valid_range": np.array((0, 300), dtype=np.float32),
            "units": 1,
        },
        {
            "zlib": True,
        },
    )

    tie_ds = tie_ds.drop_vars("stdev_of_cdr_seaice_conc_raw")

    return tie_ds


def temporally_interpolated_ecdr_dataset(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    intermediate_output_dir: Path,
    land_spillover_alg: LAND_SPILL_ALGS,
    ancillary_source: ANCILLARY_SOURCES,
    interp_range: int = 5,
    fill_the_pole_hole: bool = True,
) -> xr.Dataset:
    """Create xr dataset containing the second pass of daily enhanced CDR.

    This function returns
    - a Dataset containing
      - The temporally interpolated field. This is 3d: (time, y, x)
      - a numpy array with the temporal interpolation flags that are
        determined during the temporal interpolation process
    """
    init_datasets = []
    for iter_date in iter_dates_near_date(target_date=date, day_range=interp_range):
        init_dataset = read_or_create_and_read_idecdr_ds(
            date=iter_date,
            hemisphere=hemisphere,
            resolution=resolution,
            intermediate_output_dir=intermediate_output_dir,
            land_spillover_alg=land_spillover_alg,
            ancillary_source=ancillary_source,
        )
        init_datasets.append(init_dataset)

    data_stack = xr.concat(init_datasets, dim="time").sortby("time")

    tie_ds = temporal_interpolation(
        hemisphere=hemisphere,
        resolution=resolution,
        date=date,
        data_stack=data_stack,
        fill_the_pole_hole=fill_the_pole_hole,
        ancillary_source=ancillary_source,
    )

    return tie_ds


def write_tie_netcdf(
    *,
    tie_ds: xr.Dataset,
    output_filepath: Path,
    uncompressed_fields: Iterable[str] = ["crs", "time", "y", "x"],
    excluded_fields: Iterable[str] = [],
    tb_fields: Iterable[str] = ("h19_day_si", "h37_day_si"),
    conc_fields: Iterable[str] = (
        "conc",
        "cdr_conc_ti",
        "cdr_conc",
        "raw_bt_seaice_conc",
        "raw_nt_seaice_conc",
    ),
) -> Path:
    """Write the temporally interpolated ECDR to a netCDF file."""
    logger.info(
        f"Writing netCDF of temporally_interpolated eCDR file to: {output_filepath}"
    )

    for excluded_field in excluded_fields:
        if excluded_field in tie_ds.variables.keys():
            tie_ds = tie_ds.drop_vars(excluded_field)

    # Set netCDF encoding to compress all except excluded fields
    nc_encoding = {}
    for varname in tie_ds.variables.keys():
        varname = cast(str, varname)
        if varname in tb_fields:
            nc_encoding[varname] = {
                "zlib": True,
                "dtype": "int16",
                "scale_factor": 0.1,
                "_FillValue": 0,
            }
        elif varname in conc_fields:
            nc_encoding[varname] = {
                "zlib": True,
                "dtype": "uint8",
                "scale_factor": 0.01,
                "_FillValue": 255,
            }
        elif varname not in uncompressed_fields:
            nc_encoding[varname] = {"zlib": True}

    tie_ds.to_netcdf(
        output_filepath,
        encoding=nc_encoding,
        unlimited_dims=[
            "time",
        ],
    )

    return output_filepath


def make_tiecdr_netcdf(
    date: dt.date,
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    intermediate_output_dir: Path,
    land_spillover_alg: LAND_SPILL_ALGS,
    ancillary_source: ANCILLARY_SOURCES,
    interp_range: int = 5,
    fill_the_pole_hole: bool = True,
    overwrite_tie: bool = False,
):
    output_path = get_tie_filepath(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        intermediate_output_dir=intermediate_output_dir,
    )

    if overwrite_tie or not output_path.is_file():
        try:
            logger.info(f"Creating tiecdr for {date=}, {hemisphere=}, {resolution=}")
            tie_ds = temporally_interpolated_ecdr_dataset(
                date=date,
                hemisphere=hemisphere,
                resolution=resolution,
                interp_range=interp_range,
                intermediate_output_dir=intermediate_output_dir,
                fill_the_pole_hole=fill_the_pole_hole,
                land_spillover_alg=land_spillover_alg,
                ancillary_source=ancillary_source,
            )

            written_tie_ncfile = write_tie_netcdf(
                tie_ds=tie_ds,
                output_filepath=output_path,
            )
            logger.info(
                f"Wrote temporally interpolated daily ncfile: {written_tie_ncfile}"
            )
        except Exception as e:
            logger.exception(
                "Failed to create NetCDF for " f"{hemisphere=}, {date=}, {resolution=}."
            )
            raise e

    return output_path


def create_tiecdr_for_date_range(
    *,
    hemisphere: Hemisphere,
    start_date: dt.date,
    end_date: dt.date,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    intermediate_output_dir: Path,
    land_spillover_alg: LAND_SPILL_ALGS,
    ancillary_source: ANCILLARY_SOURCES,
    overwrite_tie: bool,
) -> None:
    """Generate the temporally composited daily ecdr files for a range of dates."""
    for date in date_range(start_date=start_date, end_date=end_date):
        make_tiecdr_netcdf(
            date=date,
            hemisphere=hemisphere,
            resolution=resolution,
            intermediate_output_dir=intermediate_output_dir,
            overwrite_tie=overwrite_tie,
            land_spillover_alg=land_spillover_alg,
            ancillary_source=ancillary_source,
        )


@click.command(name="tiecdr")
@click.option(
    "-d",
    "--date",
    required=True,
    type=click.DateTime(
        formats=(
            "%Y-%m-%d",
            "%Y%m%d",
            "%Y.%m.%d",
        )
    ),
    callback=datetime_to_date,
)
@click.option(
    "--end-date",
    required=False,
    type=click.DateTime(
        formats=(
            "%Y-%m-%d",
            "%Y%m%d",
            "%Y.%m.%d",
        )
    ),
    # Like `datetime_to_date` but allows `None`.
    callback=lambda _ctx, _param, value: value if value is None else value.date(),
    default=None,
    help="If given, run temporal composite for `--date` through this end date.",
)
@click.option(
    "-h",
    "--hemisphere",
    required=True,
    type=click.Choice(get_args(Hemisphere)),
)
@click.option(
    "--base-output-dir",
    required=True,
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
    default=DEFAULT_BASE_OUTPUT_DIR,
    help=(
        "Base output directory for standard ECDR outputs."
        " Subdirectories are created for outputs of"
        " different stages of processing."
    ),
    show_default=True,
)
@click.option(
    "-r",
    "--resolution",
    required=True,
    type=click.Choice(get_args(ECDR_SUPPORTED_RESOLUTIONS)),
)
@click.option(
    "--overwrite",
    is_flag=True,
)
def cli(
    *,
    date: dt.date,
    end_date: dt.date | None,
    hemisphere: Hemisphere,
    base_output_dir: Path,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    overwrite: bool,
    land_spillover_alg: LAND_SPILL_ALGS,
    ancillary_source: ANCILLARY_SOURCES,
) -> None:
    """Run the temporal composite daily ECDR algorithm with AMSR2 data.

    This requires the creation/existence of initial daily eCDR (idecdr) files.

    TODO: eventually we want to be able to specify: date, grid (grid includes
    projection, resolution, and bounds), and TBtype (TB type includes source and
    methodology for getting those TBs onto the grid)
    """

    if end_date is None:
        end_date = copy.copy(date)

    intermediate_output_dir = get_intermediate_output_dir(
        base_output_dir=base_output_dir,
        hemisphere=hemisphere,
        is_nrt=False,
    )

    create_tiecdr_for_date_range(
        hemisphere=hemisphere,
        start_date=date,
        end_date=end_date,
        resolution=resolution,
        intermediate_output_dir=intermediate_output_dir,
        overwrite_tie=overwrite,
        land_spillover_alg=land_spillover_alg,
        ancillary_source=ancillary_source,
    )
