"""Routines for generating temporally composited file.

"""
import copy
import datetime as dt
import sys
import traceback
from functools import cache
from pathlib import Path
from typing import Iterable, cast, get_args

import click
import numpy as np
import numpy.typing as npt
import pandas as pd
import xarray as xr
from loguru import logger
from pm_icecon.fill_polehole import fill_pole_hole
from pm_icecon.util import date_range
from pm_tb_data._types import NORTH, Hemisphere

from seaice_ecdr._types import ECDR_SUPPORTED_RESOLUTIONS
from seaice_ecdr.cli.util import datetime_to_date
from seaice_ecdr.constants import STANDARD_BASE_OUTPUT_DIR
from seaice_ecdr.initial_daily_ecdr import (
    get_idecdr_filepath,
    initial_daily_ecdr_dataset_for_au_si_tbs,
    make_idecdr_netcdf,
    write_ide_netcdf,
)
from seaice_ecdr.masks import psn_125_near_pole_hole_mask
from seaice_ecdr.util import standard_daily_filename

# Set the default minimum log notification to "info"
try:
    logger.remove(0)  # Removes previous logger info
    logger.add(sys.stderr, level="INFO")
except ValueError:
    logger.debug(f"Started logging in {__name__}")
    logger.add(sys.stderr, level="INFO")


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
def get_tie_dir(*, ecdr_data_dir: Path) -> Path:
    """Daily complete output dir for TIE processing"""
    tie_dir = ecdr_data_dir / "temporal_interp"
    tie_dir.mkdir(exist_ok=True)

    return tie_dir


def get_tie_filepath(
    date,
    hemisphere,
    resolution,
    ecdr_data_dir: Path,
) -> Path:
    """Return the complete daily tie file path."""
    standard_fn = standard_daily_filename(
        hemisphere=hemisphere,
        date=date,
        sat="am2",
        resolution=resolution,
    )
    # Add `tiecdr` to the beginning of the standard name to distinguish it as a
    # WIP.
    tie_filename = "tiecdr_" + standard_fn
    tie_dir = get_tie_dir(ecdr_data_dir=ecdr_data_dir)

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
    if skip_future:
        latest_date = target_date
    else:
        latest_date = target_date + dt.timedelta(days=day_range)

    date = earliest_date
    while date <= latest_date:
        yield date
        date += dt.timedelta(days=date_step)


def read_with_create_initial_daily_ecdr(
    *,
    date,
    hemisphere,
    resolution,
    ecdr_data_dir: Path,
    force_ide_file_creation=False,
):
    """Return init daily ecdr field, creating it if necessary."""
    ide_filepath = get_idecdr_filepath(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        ecdr_data_dir=ecdr_data_dir,
    )

    if not ide_filepath.is_file() or force_ide_file_creation:
        created_ide_ds = initial_daily_ecdr_dataset_for_au_si_tbs(
            date=date, hemisphere=hemisphere, resolution=resolution
        )
        write_ide_netcdf(ide_ds=created_ide_ds, output_filepath=ide_filepath)

    # ide_ds = xr.open_dataset(ide_filepath)
    ide_ds = xr.load_dataset(ide_filepath)

    return ide_ds


def is_seaice_conc(
    field: npt.NDArray,
) -> npt.NDArray:
    """Returns boolean index array where field has values 0-100."""
    where_seaice = (field >= 0) & (field <= 100)

    return where_seaice


def temporally_composite_dataarray(
    target_date: dt.date,
    da: xr.DataArray,
    interp_range: int = 5,
    one_sided_limit: int = 3,
    still_missing_flag: int = 255,
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
    logger.info(f"Temporally compositing {da.name} dataarray around {target_date}")
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

    try:
        temp_comp_da = da.isel(time=da.time.dt.date == target_date).copy()
    except TypeError as e:
        print(f"Got TypeError:\n{e}")
        breakpoint()

    if temp_comp_da.size == 0:
        # This time was not in the time slice; need to init to all-missing
        # TODO: Do we need dims, coords, attrs here?
        # temp_comp_da = zeros_xrdataarray(xdim, ydim, np.nan)
        raise RuntimeError(f" the target_date was not in the dataarray: {target_date}")

    temp_comp_2d = np.squeeze(temp_comp_da.data)
    assert temp_comp_2d.shape == (ydim, xdim)

    # Initialize arrays
    initial_missing_locs = np.isnan(temp_comp_2d.data)

    pconc = np.zeros((ydim, xdim), dtype=np.uint8)
    pdist = np.zeros((ydim, xdim), dtype=np.uint8)

    nconc = np.zeros((ydim, xdim), dtype=np.uint8)
    ndist = np.zeros((ydim, xdim), dtype=np.uint8)

    need_values = initial_missing_locs.copy()
    n_missing = np.sum(np.where(need_values, 1, 0))

    pconc[need_values] = 0
    pdist[need_values] = 0
    nconc[need_values] = 0
    ndist[need_values] = 0

    for time_offset in range(1, interp_range + 1):
        if n_missing == 0:
            continue

        prior_date = target_date - dt.timedelta(days=time_offset)
        next_date = target_date + dt.timedelta(days=time_offset)

        prior_field = np.squeeze(da.isel(time=da.time.dt.date == prior_date).to_numpy())
        next_field = np.squeeze(da.isel(time=da.time.dt.date == next_date).to_numpy())

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
    linint = np.round(linint).astype(np.uint8)
    temp_comp_2d[have_both_prior_and_next] = linint[have_both_prior_and_next]

    # Update the temporal interp flag value
    temporal_flags[have_both_prior_and_next] = (
        10 * pdist[have_both_prior_and_next] + ndist[have_both_prior_and_next]
    )

    temp_comp_2d[have_only_prior] = pconc[have_only_prior]

    # Update the temporal interp flag value
    temporal_flags[have_only_prior] = 10 * pdist[have_only_prior]

    temp_comp_2d[have_only_next] = nconc[have_only_next]

    # Update the temporal interp flag value
    temporal_flags[have_only_next] = ndist[have_only_next]

    temp_comp_da.data[0, :, :] = temp_comp_2d[:, :]

    return temp_comp_da, temporal_flags


# TODO: this function also belongs in the intial daily ecdr module.
def read_or_create_and_read_idecdr_ds(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ecdr_data_dir: Path,
) -> xr.Dataset:
    """Read an idecdr netCDF file, creating it if it doesn't exist."""
    ide_filepath = get_idecdr_filepath(
        date, hemisphere, resolution, ecdr_data_dir=ecdr_data_dir
    )
    # TODO: Perhaps add an overwrite condition here?
    if not ide_filepath.is_file():
        excluded_idecdr_fields = [
            "h18_day",
            "v18_day",
            "v23_day",
            "h36_day",
            "v36_day",
            # "h18_day_si",  # include this field for melt onset calculation
            "v18_day_si",
            "v23_day_si",
            # "h36_day_si",  # include this field for melt onset calculation
            "v36_day_si",
            "shoremap",
            "NT_icecon_min",
        ]
        make_idecdr_netcdf(
            date=date,
            hemisphere=hemisphere,
            resolution=resolution,
            ecdr_data_dir=ecdr_data_dir,
            excluded_fields=excluded_idecdr_fields,
        )
    logger.info(f"Reading ideCDR file from: {ide_filepath}")
    ide_ds = xr.load_dataset(ide_filepath)

    return ide_ds


def grid_is_psn125(hemisphere, gridshape):
    """Return True if this is the 12.5km NSIDC NH polar stereo grid."""
    is_nh = hemisphere == NORTH
    is_125 = gridshape == (896, 608)
    return is_nh and is_125


def get_ti_flag_dates(
    date: dt.date,
    flag_field: xr.DataArray,
) -> list:
    """Return a list of the dates needed for temporal interp per flags."""
    unique_flags = np.unique(flag_field)
    all_dates = []
    for _, flag_val in np.ndenumerate(unique_flags):
        if flag_val >= 0 and flag_val <= 99:
            tens_val = int(np.floor_divide(flag_val, 10))
            ones_val = int(np.mod(flag_val, 10))
            all_dates.append(date - dt.timedelta(days=tens_val))
            all_dates.append(date + dt.timedelta(days=ones_val))

    unique_dates_list = sorted(set(all_dates))

    return unique_dates_list


def fill_conc_field(
    varname: str,
    flag_field: xr.DataArray,
    target_date: dt.date,
) -> np.ndarray:
    """Fill the named variable array using flag information."""

    print(f"flag_field:\n{flag_field}")
    # date_list = get_ti_flag_dates(target_date, flag_field)

    # ti_filled_bt = fill_conc_field('bt_conc_raw', tie_ds['temporal_flag')
    return np.array((0))


def create_sorted_var_timestack(
    varname: str,
    date_list: list,
    ds_function,  # How do I specify a function to mypy?
    ds_function_kwargs: dict,
) -> xr.DataArray:
    # Use the first date to initialize the dataset
    init_date = date_list[0]
    init_ds = ds_function(
        date=init_date,
        **ds_function_kwargs,
    )
    var_stack = init_ds.data_vars[varname].copy()
    if len(var_stack.shape) == 2:
        var_stack = var_stack.expand_dims(time=[pd.to_datetime(init_date)])

    for interp_date in date_list[1:]:
        interp_ds = ds_function(
            date=interp_date,
            **ds_function_kwargs,
        )
        this_var = interp_ds.data_vars[varname].copy()
        if len(this_var.shape) == 2:
            this_var = this_var.expand_dims(time=[pd.to_datetime(interp_date)])
        var_stack = xr.concat([var_stack, this_var], "time")

    var_stack = var_stack.sortby("time")

    return var_stack


def calc_stddev_field(
    bt_conc,
    nt_conc,
    min_valid_value,
    max_valid_value,
    fill_value,
) -> xr.DataArray | None:
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


def temporally_interpolated_ecdr_dataset_for_au_si_tbs(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    interp_range: int = 5,
    ecdr_data_dir: Path,
    fill_the_pole_hole: bool = True,
) -> xr.Dataset:
    """Create xr dataset containing the second pass of daily enhanced CDR.

    This function returns
    - a Dataset containing
      - The temporally interpolated field. This is 3d: (time, y, x)
      - a numpy array with the temporal interpolation flags that are
        determined during the temporal interpolation process
    """
    # Read in the idecdr file for this date
    ide_ds = read_or_create_and_read_idecdr_ds(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        ecdr_data_dir=ecdr_data_dir,
    )

    # Copy ide_ds to a new xr tiecdr dataset
    tie_ds = ide_ds.copy(deep=True)

    # Update the cdr_conc var with temporally interpolated cdr_conc field
    #   by creating a DataArray with conc fields +/- interp_range around date
    var_stack = create_sorted_var_timestack(
        varname="conc",
        date_list=[
            iter_date
            for iter_date in iter_dates_near_date(
                target_date=date, day_range=interp_range
            )
        ],
        ds_function=read_or_create_and_read_idecdr_ds,
        ds_function_kwargs={
            "hemisphere": hemisphere,
            "resolution": resolution,
            "ecdr_data_dir": ecdr_data_dir,
        },
    )

    ti_var, ti_flags = temporally_composite_dataarray(
        target_date=date,
        da=var_stack,
        interp_range=interp_range,
    )

    tie_ds["cdr_conc_ti"] = ti_var

    # Update QA flag field
    is_temporally_interpolated = (ti_flags > 0) & (ti_flags <= 55)
    # TODO: this bit mask of 64 added to (equals bitwise "or")
    #       should be looked up from a map of flag mask values
    tie_ds["qa_of_cdr_seaice_conc"] = tie_ds["qa_of_cdr_seaice_conc"].where(
        ~is_temporally_interpolated,
        other=tie_ds["qa_of_cdr_seaice_conc"] + 64,
    )

    # Add the temporal interp flags to the dataset
    tie_ds["temporal_flag"] = (
        ("y", "x"),
        ti_flags,
        {
            "_FillValue": 255,
            "grid_mapping": "crs",
            "standard_name": "status_flag",
            "valid_range": [np.uint8(0), np.uint8(254)],
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
    #       operation, this will be both temporally interpoalted and
    #       polehole-filled (if appropriate).  For now, "cdr_conc" is okay
    tie_ds["cdr_conc"] = tie_ds["cdr_conc_ti"].copy()

    # TODO: This is a really coarse way of determining which
    #       grid is having its pole hole filled!
    if fill_the_pole_hole and hemisphere == NORTH:
        # TODO: Write code that better captures the logic of whether
        #       or not the grid has a pole hole to fill.  In general,
        #       this is an attribute of the grid.
        # Currently, this code expects psn12.5 grids only
        if grid_is_psn125(hemisphere=hemisphere, gridshape=cdr_conc.shape):
            cdr_conc_pre_polefill = cdr_conc.copy()
            near_pole_hole_mask = psn_125_near_pole_hole_mask()
            cdr_conc_pole_filled = fill_pole_hole(
                conc=cdr_conc,
                near_pole_hole_mask=near_pole_hole_mask,
            )
            logger.info("Filled pole hole")
            # Need to use not-isnan() here because NaN == NaN evaluates to False
            is_pole_filled = (cdr_conc_pole_filled != cdr_conc_pre_polefill) & (
                ~np.isnan(cdr_conc_pole_filled)
            )
            if "spatint_bitmask" in tie_ds.variables.keys():
                # TODO: These are constants for the eCDR runs.  They should
                #       NOT be defined here (and in the idecdr code...(!))
                # TODO Actually, if this is defined here, the 'pole_filled'
                #      bitmask value should be determined by examining the
                #      bitmask_flags and bitmask_flag_meanings fields of the
                #      DataArray variable.
                TB_SPATINT_BITMASK_MAP = {
                    "v18": 1,
                    "h18": 2,
                    "v23": 4,
                    "v36": 8,
                    "h36": 16,
                    "pole_filled": 32,
                }
                tie_ds["spatint_bitmask"] = tie_ds["spatint_bitmask"].where(
                    ~is_pole_filled,
                    other=TB_SPATINT_BITMASK_MAP["pole_filled"],
                )

                logger.info("Updated spatial_interpolation with pole hole value")
            else:
                raise RuntimeError(
                    "temporally interpolated dataset should have ",
                    '"spatint_bitmask_map" field',
                )

            tie_ds["cdr_conc"].data[0, :, :] = cdr_conc_pole_filled[:, :]
        else:
            raise RuntimeError("Only the psn12.5 pole filling is implemented")
    else:
        # TODO: May want to modify attributes of the cdr_conc field to
        #       distinguish it from the cdr_conc_ti field
        pass

    # Create the cdr_conc standard deviation field
    # Create filled bootstrap field
    bt_var_stack = create_sorted_var_timestack(
        varname="bt_conc_raw",
        date_list=[
            iter_date
            for iter_date in iter_dates_near_date(
                target_date=date, day_range=interp_range
            )
        ],
        ds_function=read_or_create_and_read_idecdr_ds,
        ds_function_kwargs={
            "hemisphere": hemisphere,
            "resolution": resolution,
            "ecdr_data_dir": ecdr_data_dir,
        },
    )

    bt_conc, bt_ti_flags = temporally_composite_dataarray(
        target_date=date,
        da=bt_var_stack,
        interp_range=interp_range,
    )

    # Create filled bootstrap field
    nt_var_stack = create_sorted_var_timestack(
        varname="nt_conc_raw",
        date_list=[
            iter_date
            for iter_date in iter_dates_near_date(
                target_date=date, day_range=interp_range
            )
        ],
        ds_function=read_or_create_and_read_idecdr_ds,
        ds_function_kwargs={
            "hemisphere": hemisphere,
            "resolution": resolution,
            "ecdr_data_dir": ecdr_data_dir,
        },
    )

    nt_conc, nt_ti_flags = temporally_composite_dataarray(
        target_date=date,
        da=nt_var_stack,
        interp_range=interp_range,
    )

    # Note: this pole-filling code is copy-pasted from the cdr_conc
    #       methodology above
    if fill_the_pole_hole and hemisphere == NORTH:
        bt_conc_2d = np.squeeze(bt_conc.data)
        nt_conc_2d = np.squeeze(nt_conc.data)

        if grid_is_psn125(hemisphere=hemisphere, gridshape=bt_conc_2d.shape):
            # Fill pole hole of BT
            bt_conc_pre_polefill = bt_conc_2d.copy()
            near_pole_hole_mask = psn_125_near_pole_hole_mask()
            bt_conc_pole_filled = fill_pole_hole(
                conc=bt_conc_2d,
                near_pole_hole_mask=near_pole_hole_mask,
            )
            logger.info("Filled pole hole (bt)")
            # Need to use not-isnan() here because NaN == NaN evaluates to False
            is_pole_filled = (bt_conc_pole_filled != bt_conc_pre_polefill) & (
                ~np.isnan(bt_conc_pole_filled)
            )
            bt_conc.data[0, :, :] = bt_conc_pole_filled[:, :]

            # Fill pole hole of NT
            nt_conc_pre_polefill = nt_conc_2d.copy()
            near_pole_hole_mask = psn_125_near_pole_hole_mask()
            nt_conc_pole_filled = fill_pole_hole(
                conc=nt_conc_2d,
                near_pole_hole_mask=near_pole_hole_mask,
            )
            logger.info("Filled pole hole (nt)")
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

    tie_ds["stdev_of_cdr_seaice_conc"] = (
        ("y", "x"),
        stddev_field,
        {
            "_FillValue": -1,
            "long_name": (
                "Passive Microwave Daily Sea Ice Concentration",
                " Source Estimated Standard Deviation",
            ),
            "grid_mapping": "crs",
            "valid_range": np.array((0, 300), dtype=np.float32),
            "units": "K",
        },
        {
            "zlib": True,
        },
    )

    # Return the tiecdr dataset
    return tie_ds


def write_tie_netcdf(
    *,
    tie_ds: xr.Dataset,
    output_filepath: Path,
    uncompressed_fields: Iterable[str] = ("crs", "time", "y", "x"),
    excluded_fields: Iterable[str] = [],
) -> Path:
    """Write the temporally interpolated ECDR to a netCDF file."""
    logger.info(f"Writing netCDF of initial_daily eCDR file to: {output_filepath}")

    for excluded_field in excluded_fields:
        if excluded_field in tie_ds.variables.keys():
            tie_ds = tie_ds.drop_vars(excluded_field)

    # Set netCDF encoding to compress all except excluded fields
    nc_encoding = {}
    for varname in tie_ds.variables.keys():
        varname = cast(str, varname)
        if varname not in uncompressed_fields:
            nc_encoding[varname] = {"zlib": True}

    tie_ds.to_netcdf(
        output_filepath,
        encoding=nc_encoding,
    )

    return output_filepath


def make_tiecdr_netcdf(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ecdr_data_dir: Path,
    interp_range: int = 5,
    fill_the_pole_hole: bool = True,
) -> None:
    logger.info(f"Creating tiecdr for {date=}, {hemisphere=}, {resolution=}")
    tie_ds = temporally_interpolated_ecdr_dataset_for_au_si_tbs(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        interp_range=interp_range,
        ecdr_data_dir=ecdr_data_dir,
        fill_the_pole_hole=fill_the_pole_hole,
    )
    output_path = get_tie_filepath(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        ecdr_data_dir=ecdr_data_dir,
    )

    written_tie_ncfile = write_tie_netcdf(
        tie_ds=tie_ds,
        output_filepath=output_path,
    )
    logger.info(f"Wrote temporally interpolated daily ncfile: {written_tie_ncfile}")


def create_tiecdr_for_date_range(
    *,
    hemisphere: Hemisphere,
    start_date: dt.date,
    end_date: dt.date,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ecdr_data_dir: Path,
) -> None:
    """Generate the temporally composited daily ecdr files for a range of dates."""
    for date in date_range(start_date=start_date, end_date=end_date):
        try:
            make_tiecdr_netcdf(
                date=date,
                hemisphere=hemisphere,
                resolution=resolution,
                ecdr_data_dir=ecdr_data_dir,
            )

        # TODO: either catch and re-throw this exception or throw an error after
        # attempting to make the netcdf for each date. The exit code should be
        # non-zero in such a case.
        except Exception:
            logger.error(
                "Failed to create NetCDF for " f"{hemisphere=}, {date=}, {resolution=}."
            )
            # TODO: These error logs should be written to e.g.,
            # `/share/apps/logs/seaice_ecdr`. The `logger` module should be able
            # to handle automatically logging error details to such a file.
            # TODO: Perhaps this function should come from seaice_ecdr
            err_filepath = get_tie_filepath(
                date=date,
                hemisphere=hemisphere,
                resolution=resolution,
                ecdr_data_dir=ecdr_data_dir,
            )
            err_filename = err_filepath.name + ".error"
            logger.info(f"Writing error info to {err_filename}")
            with open(err_filepath.parent / err_filename, "w") as f:
                traceback.print_exc(file=f)
                traceback.print_exc(file=sys.stdout)


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
    "--ecdr-data-dir",
    required=True,
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
    default=STANDARD_BASE_OUTPUT_DIR,
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
def cli(
    *,
    date: dt.date,
    end_date: dt.date | None,
    hemisphere: Hemisphere,
    ecdr_data_dir: Path,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
) -> None:
    """Run the temporal composite daily ECDR algorithm with AMSR2 data.

    This requires the creation/existence of initial daily eCDR (idecdr) files.

    TODO: eventually we want to be able to specify: date, grid (grid includes
    projection, resolution, and bounds), and TBtype (TB type includes source and
    methodology for getting those TBs onto the grid)
    """

    if end_date is None:
        end_date = copy.copy(date)

    create_tiecdr_for_date_range(
        hemisphere=hemisphere,
        start_date=date,
        end_date=end_date,
        resolution=resolution,
        ecdr_data_dir=ecdr_data_dir,
    )
