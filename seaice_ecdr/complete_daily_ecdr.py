"""Routines for generating completely filled daily eCDR files.

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
import xarray as xr
from loguru import logger
from pm_tb_data._types import NORTH, Hemisphere

from seaice_ecdr._types import ECDR_SUPPORTED_RESOLUTIONS
from seaice_ecdr.ancillary import (
    get_non_ocean_mask,
    get_surfacetype_da,
)
from seaice_ecdr.checksum import write_checksum_file
from seaice_ecdr.cli.util import datetime_to_date
from seaice_ecdr.constants import STANDARD_BASE_OUTPUT_DIR
from seaice_ecdr.melt import (
    MELT_ONSET_FILL_VALUE,
    MELT_SEASON_FIRST_DOY,
    MELT_SEASON_LAST_DOY,
    date_in_nh_melt_season,
    melting,
)
from seaice_ecdr.platforms import (
    get_platform_by_date,
)
from seaice_ecdr.set_daily_ncattrs import finalize_cdecdr_ds
from seaice_ecdr.temporal_composite_daily import get_tie_filepath, make_tiecdr_netcdf
from seaice_ecdr.util import date_range, get_ecdr_grid_shape, standard_daily_filename


@cache
def get_ecdr_dir(*, ecdr_data_dir: Path) -> Path:
    """Daily complete output dir for ECDR processing"""
    ecdr_dir = ecdr_data_dir / "complete_daily"
    ecdr_dir.mkdir(exist_ok=True)

    return ecdr_dir


def get_ecdr_filepath(
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ecdr_data_dir: Path,
) -> Path:
    """Return the complete daily eCDR file path."""
    platform = get_platform_by_date(date)
    ecdr_filename = standard_daily_filename(
        hemisphere=hemisphere,
        date=date,
        sat=platform,
        resolution=resolution,
    )
    ecdr_dir = get_ecdr_dir(ecdr_data_dir=ecdr_data_dir)

    ecdr_filepath = ecdr_dir / ecdr_filename

    return ecdr_filepath


# TODO: this should be in the temporal interpolation module.
def read_tiecdr_ds(
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ecdr_data_dir: Path,
) -> xr.Dataset:
    tie_filepath = get_tie_filepath(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        ecdr_data_dir=ecdr_data_dir,
    )
    logger.info(f"Reading tieCDR file from: {tie_filepath}")
    tie_ds = xr.load_dataset(tie_filepath)

    return tie_ds


# TODO: this should be in the temporal interpolation module.
def read_or_create_and_read_standard_tiecdr_ds(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ecdr_data_dir: Path,
    overwrite_tie: bool = False,
) -> xr.Dataset:
    """Read an tiecdr netCDF file, creating it if it doesn't exist.

    Uses standard input sources (non-NRT).
    """
    make_tiecdr_netcdf(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        ecdr_data_dir=ecdr_data_dir,
        overwrite=overwrite_tie,
    )

    tie_ds = read_tiecdr_ds(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        ecdr_data_dir=ecdr_data_dir,
    )

    return tie_ds


def _empty_melt_onset_field(
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
) -> npt.NDArray:
    """Return an array of the shape for this hem/res filled with fill_value."""
    # `grid_shape` is 2D.
    grid_shape = get_ecdr_grid_shape(
        hemisphere=hemisphere,
        resolution=resolution,
    )
    # Add a dim for `time`, making the melt onset field 3D.
    melt_onset_field_shape = (1, *grid_shape)
    empty_melt_onset_field = np.full(
        melt_onset_field_shape,
        MELT_ONSET_FILL_VALUE,
        dtype=np.uint8,
    )

    return empty_melt_onset_field


def read_melt_onset_field_from_complete_daily(
    *,
    date,
    hemisphere,
    resolution,
    ecdr_data_dir: Path,
) -> npt.NDArray:
    cde_ds = read_cdecdr_ds(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        ecdr_data_dir=ecdr_data_dir,
    )

    # TODO: Perhaps these field names should be in a dictionary somewhere?
    melt_onset_from_ds = cde_ds["melt_onset_day_cdr_seaice_conc"].to_numpy()

    return melt_onset_from_ds


def read_melt_elements_from_tiecdr(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ecdr_data_dir: Path,
):
    tie_ds = read_tiecdr_ds(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        ecdr_data_dir=ecdr_data_dir,
    )

    return (
        np.squeeze(tie_ds["cdr_conc"].to_numpy()),
        tie_ds["h19_day_si"].to_numpy(),
        tie_ds["h37_day_si"].to_numpy(),
    )


def update_melt_onset_for_day(
    *,
    prior_melt_onset_field: npt.NDArray,
    date: dt.date,
    cdr_conc: npt.NDArray,
    tb_h19: npt.NDArray,
    tb_h37: npt.NDArray,
    non_ocean_mask: npt.NDArray[np.bool_],
) -> npt.NDArray[np.uint8]:
    """Return an updated melt onset field given data for the given day."""
    is_melted_today = melting(
        concentrations=cdr_conc,
        tb_h19=tb_h19,
        tb_h37=tb_h37,
    )
    # Apply non-ocean mask
    is_melted_today[0, non_ocean_mask] = False

    have_prior_melt_values = prior_melt_onset_field != MELT_ONSET_FILL_VALUE
    is_missing_prior = prior_melt_onset_field == MELT_ONSET_FILL_VALUE
    has_new_melt = is_missing_prior & is_melted_today

    melt_onset_field = np.zeros(prior_melt_onset_field.shape, dtype=np.uint8)
    melt_onset_field[:] = MELT_ONSET_FILL_VALUE
    melt_onset_field[have_prior_melt_values] = prior_melt_onset_field[
        have_prior_melt_values
    ]

    day_of_year = int(date.strftime("%j"))
    melt_onset_field[has_new_melt] = day_of_year

    return melt_onset_field


def create_melt_onset_field(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ecdr_data_dir: Path,
) -> npt.NDArray[np.uint8] | None:
    """Return a uint8 melt onset field (NH only).

    Note: this routine creates the melt onset field using input data
    from today (and yesterday if needed).

    For dates with day of year prior to first_melt_doy or after last_melt_doy,
    this will be an uint8 array with all values set to no_melt_flag.

    For date with day of year equal to first_melt_doy, this field will be
    computed.

    For date between first_melt_doy and last_melt_doy, this will read the
    melt onset field from the prior day's cdecdr file and compute melt
    for the current day.
    """
    if hemisphere != NORTH:
        return None

    day_of_year = int(date.strftime("%j"))
    # Determine if the given day of year is within the melt season. If it's not,
    # return an empty melt onset field.
    if not date_in_nh_melt_season(date=date):
        logger.info(f"returning empty melt_onset_field for {day_of_year}")
        return _empty_melt_onset_field(
            hemisphere=hemisphere,
            resolution=resolution,
        )

    is_first_day_of_melt = day_of_year == MELT_SEASON_FIRST_DOY
    if is_first_day_of_melt:
        # The first day of the melt seasion should start with an empty melt
        # onset field.
        prior_melt_onset_field = _empty_melt_onset_field(
            hemisphere=hemisphere,
            resolution=resolution,
        )
        logger.info(f"using empty melt_onset_field for prior for {day_of_year}")
    else:
        # During the melt season, try to read the previous day's input as a
        # starting point. Use an empty melt onset field if no data for the
        # previous day is available.
        try:
            prior_melt_onset_field = read_melt_onset_field_from_complete_daily(
                date=date - dt.timedelta(days=1),
                hemisphere=hemisphere,
                resolution=resolution,
                ecdr_data_dir=ecdr_data_dir,
            )
            logger.info(f"using read melt_onset_field for prior for {day_of_year}")
        except FileNotFoundError:
            logger.warning(
                f"Tried to read previous melt field for {day_of_year} but the file was not found."
            )
            prior_melt_onset_field = _empty_melt_onset_field(
                hemisphere=hemisphere,
                resolution=resolution,
            )
            logger.info(f"using empty melt_onset_field for prior for {day_of_year}")

    cdr_conc_ti, tb_h19, tb_h37 = read_melt_elements_from_tiecdr(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        ecdr_data_dir=ecdr_data_dir,
    )

    non_ocean_mask = get_non_ocean_mask(
        hemisphere=hemisphere,
        resolution=resolution,
    )

    updated_melt_onset = update_melt_onset_for_day(
        prior_melt_onset_field=prior_melt_onset_field,
        cdr_conc=cdr_conc_ti,
        tb_h19=tb_h19,
        tb_h37=tb_h37,
        non_ocean_mask=non_ocean_mask.data,
        date=date,
    )

    return updated_melt_onset


def complete_daily_ecdr_ds(
    *,
    tie_ds: xr.Dataset,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ecdr_data_dir: Path,
) -> xr.Dataset:
    """Create xr dataset containing the complete daily enhanced CDR.

    This function returns
    - a Dataset containing
      - The melt onset field
      - All appropriate QA and QC fields
    """
    cde_ds = tie_ds.copy()

    melt_onset_field = create_melt_onset_field(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        ecdr_data_dir=ecdr_data_dir,
    )

    # Add the surface-type field
    # TODO: Setting a DataArray directly into a Dataset changes the
    #       coordinate variables of the Dataset.  Eg, here, if the
    #       surfacetype dataarray is set directly, the x, y, and time
    #       variables/coords/dims of the cde_ds are reset.
    #       The methodology here should be reviewed to see if there is
    #       a "better" way to add a geo-referenced dataarray to an existing
    #       xr Dataset.
    platform = get_platform_by_date(date)
    surfacetype_da = get_surfacetype_da(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        platform=platform,
    )
    # Force use of the cde_ds coords instead of the x, y, time vars
    # from the ancillary file (which *should* be compatible...but we
    # don't want coords changing in cde_ds as a result of external files).
    surfacetype_da.assign_coords(
        {
            "time": cde_ds.time,
            "y": cde_ds.y,
            "x": cde_ds.x,
        }
    )
    cde_ds = cde_ds.merge(surfacetype_da)

    # TODO: Need to ensure that the cdr_seaice_conc field does not have values
    #       where seaice cannot occur, eg over land or lakes

    # If no melt_onset_field, then finalize nc attrs and return dataset
    if melt_onset_field is None:
        cde_ds = finalize_cdecdr_ds(cde_ds, hemisphere)
        return cde_ds

    # Update cde_ds with melt onset info
    cde_ds["melt_onset_day_cdr_seaice_conc"] = (
        ("time", "y", "x"),
        melt_onset_field,
        {
            "grid_mapping": "crs",
            "standard_name": "status_flag",
            "comment": (
                "Value of 255 means no melt detected yet or the date is"
                " outside the melt season.  Other values indicate the day"
                " of year when melt was first detected at this location."
            ),
        },
        {
            "zlib": True,
        },
    )

    # the np.squeeze() function here is to remove the time dim so that
    # this becomes a 2d array for updating the qa_... field
    is_melt_has_occurred = np.squeeze(
        (MELT_SEASON_FIRST_DOY <= cde_ds["melt_onset_day_cdr_seaice_conc"].data)
        & (cde_ds["melt_onset_day_cdr_seaice_conc"].data <= MELT_SEASON_LAST_DOY)
    )
    # TODO: the flag value being "or"ed to the bitmask should be looked
    #       up as the temporally-interpolation-has-occured value
    #       rather than hardcoded as '128'.
    cde_ds["qa_of_cdr_seaice_conc"] = cde_ds["qa_of_cdr_seaice_conc"].where(
        ~is_melt_has_occurred,
        other=np.bitwise_or(cde_ds["qa_of_cdr_seaice_conc"], 128),
    )

    cde_ds = finalize_cdecdr_ds(cde_ds, hemisphere)

    return cde_ds


def write_cde_netcdf(
    *,
    cde_ds: xr.Dataset,
    output_filepath: Path,
    ecdr_data_dir: Path,
    uncompressed_fields: Iterable[str] = ("crs", "time", "y", "x"),
    excluded_fields: Iterable[str] = [],
    conc_fields: Iterable[str] = [
        "raw_bt_seaice_conc",
        "raw_nt_seaice_conc",
        "cdr_seaice_conc",
    ],
) -> Path:
    """Write the complete, temporally interpolated ECDR to a netCDF file.

    This function also creates a checksum file for the complete daily netcdf.
    """
    logger.info(f"Writing netCDF of initial_daily eCDR file to: {output_filepath}")
    for excluded_field in excluded_fields:
        if excluded_field in cde_ds.variables.keys():
            cde_ds = cde_ds.drop_vars(excluded_field)

    nc_encoding = {}
    for varname in cde_ds.variables.keys():
        varname = cast(str, varname)
        if varname in conc_fields:
            nc_encoding[varname] = {
                "zlib": True,
                "dtype": "uint8",
                "scale_factor": 0.01,
                "add_offset": 0.0,
                "_FillValue": 255,
            }
        elif varname not in uncompressed_fields:
            nc_encoding[varname] = {"zlib": True}

    cde_ds.to_netcdf(
        output_filepath,
        encoding=nc_encoding,
        unlimited_dims=[
            "time",
        ],
    )

    # Write checksum file for the complete daily output.
    write_checksum_file(
        input_filepath=output_filepath,
        ecdr_data_dir=ecdr_data_dir,
        product_type="complete_daily",
    )

    return output_filepath


def make_standard_cdecdr_netcdf(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ecdr_data_dir: Path,
    overwrite_cde: bool = False,
) -> Path:
    """Create a 'standard', complete (ready for prod) daily CDR NetCDF file.

    'standard' files are those that use the non-NRT input sources.

    This function is recursive. It will attempt to create earlier complete daily
    files if they do not exist.
    """
    cde_filepath = get_ecdr_filepath(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        ecdr_data_dir=ecdr_data_dir,
    )

    if overwrite_cde or not cde_filepath.is_file():
        logger.info(f"Creating cdecdr for {date=}, {hemisphere=}, {resolution=}")

        tie_ds = read_or_create_and_read_standard_tiecdr_ds(
            date=date,
            hemisphere=hemisphere,
            resolution=resolution,
            ecdr_data_dir=ecdr_data_dir,
        )

        # Ensure the previous day's complete daily field exists for the melt
        # onset calculation/update.
        # This is a recursive function! Ideally we'd just have code that
        # generates the necessary intermediate files for the target date, and
        # then this code would be solely responsible for reading the previous
        # day's complete field.
        # TODO: this computation is only necessary or NH files which have
        #       the melt onset field.  SH never has the melt onset field
        #       (or at least not for the current version of the CDR).
        if date_in_nh_melt_season(date=date - dt.timedelta(days=1)):
            make_standard_cdecdr_netcdf(
                date=date - dt.timedelta(days=1),
                hemisphere=hemisphere,
                resolution=resolution,
                ecdr_data_dir=ecdr_data_dir,
                overwrite_cde=overwrite_cde,
            )

        cde_ds = complete_daily_ecdr_ds(
            tie_ds=tie_ds,
            date=date,
            hemisphere=hemisphere,
            resolution=resolution,
            ecdr_data_dir=ecdr_data_dir,
        )

        written_cde_ncfile = write_cde_netcdf(
            cde_ds=cde_ds,
            output_filepath=cde_filepath,
            ecdr_data_dir=ecdr_data_dir,
        )
        logger.info(f"Wrote complete daily ncfile: {written_cde_ncfile}")

    return cde_filepath


def read_cdecdr_ds(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ecdr_data_dir: Path,
) -> xr.Dataset:
    cde_filepath = get_ecdr_filepath(
        date,
        hemisphere,
        resolution,
        ecdr_data_dir=ecdr_data_dir,
    )
    logger.info(f"Reading cdeCDR file from: {cde_filepath}")
    cde_ds = xr.load_dataset(cde_filepath)

    return cde_ds


def read_or_create_and_read_standard_cdecdr_ds(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ecdr_data_dir: Path,
    overwrite_cde: bool = False,
) -> xr.Dataset:
    """Read an cdecdr netCDF file, creating it if it doesn't exist.

    Uses standard input sources (non-NRT).

    Note: this can be recursive because the melt onset field calculation
    requires the prior day's field values during the melt season.
    """
    make_standard_cdecdr_netcdf(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        ecdr_data_dir=ecdr_data_dir,
        overwrite_cde=overwrite_cde,
    )

    cde_ds = read_cdecdr_ds(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        ecdr_data_dir=ecdr_data_dir,
    )

    return cde_ds


def create_standard_cdecdr_for_date(
    date: dt.date,
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ecdr_data_dir: Path,
    overwrite_cde: bool = False,
) -> None:
    try:
        make_standard_cdecdr_netcdf(
            date=date,
            hemisphere=hemisphere,
            resolution=resolution,
            ecdr_data_dir=ecdr_data_dir,
            overwrite_cde=overwrite_cde,
        )

    # TODO: either catch and re-throw this exception or throw an error after
    # attempting to make the netcdf for each date. The exit code should be
    # non-zero in such a case.
    except Exception:
        logger.error(
            "Failed to create complete daily NetCDF for"
            f" {hemisphere=}, {date=}, {resolution=}."
        )
        # TODO: These error logs should be written to e.g.,
        # `/share/apps/logs/seaice_ecdr`. The `logger` module should be able
        # to handle automatically logging error details to such a file.
        err_filepath = get_ecdr_filepath(
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


def create_standard_cdecdr_for_date_range(
    *,
    hemisphere: Hemisphere,
    start_date: dt.date,
    end_date: dt.date,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ecdr_data_dir: Path,
    overwrite_cde: bool = False,
) -> None:
    """Generate the complete daily ecdr files for a range of dates."""
    for date in date_range(start_date=start_date, end_date=end_date):
        create_standard_cdecdr_for_date(
            hemisphere=hemisphere,
            date=date,
            resolution=resolution,
            ecdr_data_dir=ecdr_data_dir,
            overwrite_cde=overwrite_cde,
        )


@click.command(name="cdecdr")
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

    This requires the creation/existence of temporally interpolated eCDR
    (tiecdr) files.

    TODO: eventually we want to be able to specify: date, grid (grid includes
    projection, resolution, and bounds), and TBtype (TB type includes source and
    methodology for getting those TBs onto the grid)
    """
    if end_date is None:
        end_date = copy.copy(date)

    create_standard_cdecdr_for_date_range(
        hemisphere=hemisphere,
        start_date=date,
        end_date=end_date,
        resolution=resolution,
        ecdr_data_dir=ecdr_data_dir,
    )
