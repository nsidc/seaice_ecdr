"""Routines for generating temporally composited file.

"""

import click
import traceback
import datetime as dt
import sys
import numpy as np
import numpy.typing as npt
import xarray as xr
from loguru import logger
from pathlib import Path
from typing import get_args, Iterable, cast
from pm_icecon.util import date_range, standard_output_filename
from pm_icecon.fill_polehole import fill_pole_hole
from seaice_ecdr.masks import psn_125_near_pole_hole_mask
from pm_tb_data._types import Hemisphere, NORTH
from pm_tb_data.fetch.au_si import AU_SI_RESOLUTIONS

from seaice_ecdr.initial_daily_ecdr import (
    initial_daily_ecdr_dataset_for_au_si_tbs,
    make_idecdr_netcdf,
    write_ide_netcdf,
)
from seaice_ecdr.cli.util import datetime_to_date
from seaice_ecdr.constants import (
    INITIAL_DAILY_OUTPUT_DIR,
    TEMPORAL_INTERP_DAILY_OUTPUT_DIR,
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
    """Return name of sample initial daily ecdr file."""
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
    output_directory: Path,
):
    """Return standard ide file name."""
    # TODO: Perhaps this function should come from seaice_ecdr, not pm_icecon?
    #       Specifically, the conventions specified in open Trello card
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
    ide_dir: Path,
    force_ide_file_creation=False,
):
    """Return init daily ecdr field, creating it if necessary."""
    ide_filepath = get_standard_initial_daily_ecdr_filename(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        output_directory=ide_dir,
    )

    if not ide_filepath.is_file() or force_ide_file_creation:
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

    temp_comp_da.data[0, :, :] = temp_comp_2d[:, :]

    return temp_comp_da, temporal_flags


def get_idecdr_filename(
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: AU_SI_RESOLUTIONS,
    idecdr_dir: Path,
) -> Path:
    """Yields the name of the pass1 -- idecdr -- intermediate file."""

    # TODO: Perhaps this function should come from seaice_ecdr, not pm_icecon?
    idecdr_fn = standard_output_filename(
        hemisphere=hemisphere,
        date=date,
        sat="ausi",
        algorithm="idecdr",
        resolution=f"{resolution}km",
    )
    idecdr_path = idecdr_dir / idecdr_fn

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
            output_dir=ide_dir,
            excluded_fields=excluded_idecdr_fields,
        )
    logger.info(f"Reading ideCDR file from: {ide_filepath}")
    ide_ds = xr.open_dataset(ide_filepath)

    return ide_ds


def grid_is_psn125(hemisphere, gridshape):
    """Return True if this is the 12.5km NSIDC NH polar stereo grid."""
    is_nh = hemisphere == NORTH
    is_125 = gridshape == (896, 608)
    return is_nh and is_125


def temporally_interpolated_ecdr_dataset_for_au_si_tbs(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: AU_SI_RESOLUTIONS,
    interp_range: int = 5,
    ide_dir: Path,
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
        date=date, hemisphere=hemisphere, resolution=resolution, ide_dir=ide_dir
    )

    # Copy ide_ds to a new xr tiecdr dataset
    tie_ds = ide_ds.copy(deep=True)
    # ds_varlist = [name for name in tie_ds.data_vars]

    # Update the cdr_conc var with temporally interpolated cdr_conc field
    #   by creating a DataArray with conc fields +/- interp_range around date
    interp_varname = "conc"
    var_stack = ide_ds.data_vars[interp_varname].copy()
    for interp_date in iter_dates_near_date(target_date=date, day_range=interp_range):
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

    tie_ds["cdr_conc_ti"] = ti_var

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
                tb_spatint_bitmask_map = {
                    "v18": 1,
                    "h18": 2,
                    "v23": 4,
                    "v36": 8,
                    "h36": 16,
                    "pole_filled": 32,
                }
                tie_ds["spatint_bitmask"].data[
                    is_pole_filled
                ] += tb_spatint_bitmask_map["pole_filled"]
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

    # Here, we should specify details about the initial daily eCDF file, eg:
    #  exclude unwanted fields
    #  ensure that fields are compressed
    # Set netCDF encoding to compress all except excluded fields
    for excluded_field in excluded_fields:
        if excluded_field in tie_ds.variables.keys():
            tie_ds = tie_ds.drop_vars(excluded_field)

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
    resolution: AU_SI_RESOLUTIONS,
    output_dir: Path,
    ide_dir: Path,
    interp_range: int = 5,
    fill_the_pole_hole: bool = True,
) -> None:
    logger.info(f"Creating tiecdr for {date=}, {hemisphere=}, {resolution=}")
    tie_ds = temporally_interpolated_ecdr_dataset_for_au_si_tbs(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        interp_range=interp_range,
        ide_dir=ide_dir,
        fill_the_pole_hole=fill_the_pole_hole,
    )
    # TODO: Perhaps this function should come from seaice_ecdr, not pm_icecon?
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
    logger.info(f"Wrote temporally interpolated daily ncfile: {written_tie_ncfile}")


def create_tiecdr_for_date_range(
    *,
    hemisphere: Hemisphere,
    start_date: dt.date,
    end_date: dt.date,
    resolution: AU_SI_RESOLUTIONS,
    output_dir: Path,
    ide_dir: Path,
) -> None:
    """Generate the temporally composited daily ecdr files for a range of dates."""
    for date in date_range(start_date=start_date, end_date=end_date):
        try:
            make_tiecdr_netcdf(
                date=date,
                hemisphere=hemisphere,
                resolution=resolution,
                output_dir=output_dir,
                ide_dir=ide_dir,
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
            err_filename = standard_output_filename(
                hemisphere=hemisphere,
                date=date,
                sat="u2",
                algorithm="tiecdr",
                resolution=f"{resolution}km",
            )
            err_filename += ".error"
            logger.info(f"Writing error info to {err_filename}")
            with open(output_dir / err_filename, "w") as f:
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
    "-h",
    "--hemisphere",
    required=True,
    type=click.Choice(get_args(Hemisphere)),
)
@click.option(
    "-o",
    "--output-dir",
    required=True,
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
    default=TEMPORAL_INTERP_DAILY_OUTPUT_DIR,
    show_default=True,
)
@click.option(
    "-r",
    "--resolution",
    required=True,
    type=click.Choice(get_args(AU_SI_RESOLUTIONS)),
)
@click.option(
    "--initial-daily-ecdr-dir",
    required=True,
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
    default=INITIAL_DAILY_OUTPUT_DIR,
    show_default=True,
)
def cli(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    output_dir: Path,
    resolution: AU_SI_RESOLUTIONS,
    initial_daily_ecdr_dir: Path,
) -> None:
    """Run the temporal composite daily ECDR algorithm with AMSR2 data.

    This requires the creation/existence of initial daily eCDR (idecdr) files.

    TODO: eventually we want to be able to specify: date, grid (grid includes
    projection, resolution, and bounds), and TBtype (TB type includes source and
    methodology for getting those TBs onto the grid)
    """
    create_tiecdr_for_date_range(
        hemisphere=hemisphere,
        start_date=date,
        end_date=date,
        resolution=resolution,
        output_dir=output_dir,
        ide_dir=initial_daily_ecdr_dir,
    )
