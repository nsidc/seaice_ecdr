"""Routines for generating temporally composited file.

"""

import datetime as dt
import sys
import numpy as np
import numpy.typing as npt
import xarray as xr
from loguru import logger
from pathlib import Path
from pm_icecon.util import standard_output_filename
from pm_tb_data._types import Hemisphere, NORTH
from pm_tb_data.fetch.au_si import AU_SI_RESOLUTIONS

from seaice_ecdr.initial_daily_ecdr import (
    initial_daily_ecdr_dataset_for_au_si_tbs,
    make_idecdr_netcdf,
    write_ide_netcdf,
)


# Set the default minimum log notification to "info"
try:
    logger.remove(0)  # Removes previous logger info
    logger.add(sys.stderr, level="INFO")
except ValueError:
    logger.debug(f"Started logging in {__name__}")
    logger.add(sys.stderr, level="INFO")


def get_sample_idecdr_filename(
    date,
    hemisphere,
    resolution,
):
    """Return name of sample inidial daily ecdr file."""
    sample_idecdr_filename = (
        f"sample_idecdr_{hemisphere}_{resolution}_" + f'{date.strftime("%Y%m%d")}.nc'
    )

    return sample_idecdr_filename


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


def get_standard_initial_daily_ecdr_filename(
    date,
    hemisphere,
    resolution,
    output_directory="",
):
    """Return standard ide file name."""
    standard_initial_daily_ecdr_filename = standard_output_filename(
        algorithm="idecdr",
        hemisphere=hemisphere,
        date=date,
        sat="ausi",
        resolution=f"{resolution}km",
    )
    initial_daily_ecdr_filename = Path(
        output_directory,
        standard_initial_daily_ecdr_filename,
    )

    return initial_daily_ecdr_filename


def read_with_create_initial_daily_ecdr(
    *,
    date,
    hemisphere,
    resolution,
    ide_filepath=None,
    force_ide_file_creation=False,
):
    """Return init daily ecdr field, creating it if necessary."""
    if ide_filepath is None:
        ide_filepath = get_standard_initial_daily_ecdr_filename(
            date=date,
            hemisphere=hemisphere,
            resolution=resolution,
        )

    if not ide_filepath.exists() or force_ide_file_creation:
        created_ide_ds = initial_daily_ecdr_dataset_for_au_si_tbs(
            date=date, hemisphere=hemisphere, resolution=resolution
        )
        write_ide_netcdf(ide_ds=created_ide_ds, output_filepath=ide_filepath)

    ide_ds = xr.open_dataset(ide_filepath)

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
):
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
    print("in temporally_composite_dataarray()")
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

    temp_comp_da = da.isel(time=da.time.dt.date == target_date).copy()

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

    """ I think this is obsolete...
    # Update the new .nc file with new temporal_interpolation_flag
    if tried_temporal_interp:
        if n_missing > 0:
            # Tried, but could not find replacement values
            where_missing = temp_comp_2d == missing_value
            temporal_flags[where_missing] = FLAGVAL_STILLMISSING
    """

    temp_comp_da.data[0, :, :] = temp_comp_2d[:, :]

    return temp_comp_da, temporal_flags


def gen_temporal_composite_daily(
    date,
    hemisphere,
    resolution,
):
    """Create a temporally composited daily data set."""
    print("NOTE: gen_temporal_composite_daily() only partially implemented...")

    # Load data from all contributing files
    # Is it possible to make this "lazy" evaluation so we don't create/read
    # these fields until they are needed?
    # Though...I guess they are always needed in the NH because we try
    # to fill in data near the North Pole (hole).
    init_datasets = {}
    for date in iter_dates_near_date(date, day_range=3):
        # Read in or create the data set
        # ds = read_with_create_initial_daily_ecdr(date, hemisphere, resolution)

        # Drop unnecessary fields, and assert existence of needed fields
        # Question: does it make sense to temporally interpolate
        #   unfiltered fields such as bt_raw and nt_raw?  Perhaps need
        #   to apply filter fields to those....
        init_datasets[date] = date

    # This is a placeholder showing that dates were looped through...
    for ds in init_datasets:
        print(f"ds: {ds}")

    # Loop over all desired each desired output field
    # potentially including associated fields such as interp flag fields
    # Write out the composited file


def get_idecdr_filename(
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: AU_SI_RESOLUTIONS,
    idecdr_dir: Path,
) -> Path:
    idecdr_fn = standard_output_filename(
        hemisphere=hemisphere,
        date=date,
        sat="ausi",
        algorithm="idecdr",
        resolution=f"{resolution}km",
    )
    idecdr_path = Path(idecdr_dir) / Path(idecdr_fn)

    return idecdr_path


def read_or_create_and_read_idecdr_ds(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: AU_SI_RESOLUTIONS,
    ide_dir: Path,
) -> xr.Dataset:
    """Read an idecdr netCDF file, creating it if it doesn't exist."""
    ide_filepath = get_idecdr_filename(date, hemisphere, resolution, idecdr_dir=ide_dir)
    if not ide_filepath.exists():
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
            output_dir=ide_dir,
            excluded_fields=excluded_idecdr_fields,
        )
    ide_ds = xr.open_dataset(ide_filepath)

    return ide_ds


def temporally_interpolated_ecdr_dataset_for_au_si_tbs(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: AU_SI_RESOLUTIONS,
    interp_range: int = 5,
    ide_dir: Path,
) -> tuple[xr.Dataset, npt.NDArray]:
    """Create xr dataset containing the second pass of daily enhanced CDR.

    This function returns two fields:
      - a Dataset containing the temporally interpolated field
        - this is 3d: (time, y, x)
      - a numpy array with the temporal interpolation flags that are
          determined during the temporal interpolation process
    """
    # Read in the idecdr file for this date
    ide_ds = read_or_create_and_read_idecdr_ds(
        date=date, hemisphere=hemisphere, resolution=resolution, ide_dir=ide_dir
    )

    # Copy ide_ds to a new xr tiecdr dataset
    ide_ds.copy(deep=True)
    # ds_varlist = [name for name in tie_ds.data_vars]

    # Update the cdr_conc var with temporally interpolated cdr_conc field
    #   by creating a DataArray with conc fields +/- interp_range around date
    interp_varname = "conc"
    var_stack = ide_ds.data_vars[interp_varname].copy()
    for interp_date in iter_dates_near_date(target_date=date, day_range=interp_range):
        print(f"interp_date: {interp_date}")
        if interp_date != date:
            interp_ds = read_or_create_and_read_idecdr_ds(
                date=interp_date,
                hemisphere=hemisphere,
                resolution=resolution,
                ide_dir=ide_dir,
            )
            this_var = interp_ds.data_vars[interp_varname].copy()
            var_stack = xr.concat([var_stack, this_var], "time")

    var_stack = var_stack.sortby("time")

    ti_var, ti_flags = temporally_composite_dataarray(
        target_date=date,
        da=var_stack,
        interp_range=interp_range,
    )

    # Return the tiecdr dataset
    return ti_var, ti_flags


def make_tiecdr_netcdf(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: AU_SI_RESOLUTIONS,
    output_dir: Path,
) -> None:
    logger.info(f"Creating tiecdr for {date=}, {hemisphere=}, {resolution=}")
    temporally_interpolated_ecdr_dataset_for_au_si_tbs(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        ide_dir=output_dir,
    )
    """
    output_fn = standard_output_filename(
        hemisphere=hemisphere,
        date=date,
        sat="ausi",
        algorithm="tiecdr",
        resolution=f"{resolution}km",
    )
    output_path = Path(output_dir) / Path(output_fn)

    written_tie_ncfile = write_tie_netcdf(
        tie_ds=tie_ds,
        output_filepath=output_path,
    )
    logger.info(f"Wrote intermed daily ncfile: {written_tie_ncfile}")
    """


if __name__ == "__main__":
    date = dt.datetime(2021, 2, 16).date()
    hemisphere = NORTH
    resolution = "12"

    gen_temporal_composite_daily(date, hemisphere, resolution)
