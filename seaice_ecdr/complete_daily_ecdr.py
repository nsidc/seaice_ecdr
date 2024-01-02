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
import xarray as xr
from loguru import logger
from pm_icecon.util import date_range
from pm_tb_data._types import NORTH, SOUTH, Hemisphere

from seaice_ecdr._types import ECDR_SUPPORTED_RESOLUTIONS
from seaice_ecdr.ancillary import (
    get_land_mask,
    get_surfacetype_da,
)
from seaice_ecdr.cli.util import datetime_to_date
from seaice_ecdr.constants import STANDARD_BASE_OUTPUT_DIR
from seaice_ecdr.melt import (
    MELT_ONSET_FILL_VALUE,
    MELT_SEASON_FIRST_DOY,
    MELT_SEASON_LAST_DOY,
    melting,
)
from seaice_ecdr.set_daily_ncattrs import finalize_cdecdr_ds
from seaice_ecdr.temporal_composite_daily import get_tie_filepath, make_tiecdr_netcdf
from seaice_ecdr.util import standard_daily_filename


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
    ecdr_filename = standard_daily_filename(
        hemisphere=hemisphere,
        date=date,
        # TODO: extract this to kwarg!!!
        sat="am2",
        resolution=resolution,
    )
    ecdr_dir = get_ecdr_dir(ecdr_data_dir=ecdr_data_dir)

    ecdr_filepath = ecdr_dir / ecdr_filename

    return ecdr_filepath


def read_or_create_and_read_tiecdr_ds(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ecdr_data_dir: Path,
) -> xr.Dataset:
    """Read an tiecdr netCDF file, creating it if it doesn't exist."""
    tie_filepath = get_tie_filepath(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        ecdr_data_dir=ecdr_data_dir,
    )
    # TODO: This only creates if file is missing.  We may want an overwrite opt
    if not tie_filepath.is_file():
        make_tiecdr_netcdf(
            date=date,
            hemisphere=hemisphere,
            resolution=resolution,
            ecdr_data_dir=ecdr_data_dir,
        )
    logger.info(f"Reading tieCDR file from: {tie_filepath}")
    tie_ds = xr.load_dataset(tie_filepath)

    return tie_ds


def filled_ndarray(
    *,
    hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    fill_value,
    dtype=np.uint8,
) -> np.ndarray:
    """Return an array of the shape for this hem/res filled with fill_value."""
    if hemisphere == NORTH and resolution == "12.5":
        array_shape = (1, 896, 608)
    elif hemisphere == SOUTH and resolution == "12.5":
        array_shape = (1, 664, 632)
    else:
        raise RuntimeError(
            f"Could not determine array shape for {hemisphere}" f" and {resolution}"
        )
    array = np.full(array_shape, fill_value, dtype=dtype)

    return array


def read_melt_onset_field(
    *,
    date,
    hemisphere,
    resolution,
    ecdr_data_dir: Path,
) -> np.ndarray:
    """Return the melt onset field for this complete daily eCDR file."""
    cde_ds = read_or_create_and_read_cdecdr_ds(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        ecdr_data_dir=ecdr_data_dir,
    )

    # TODO: Perhaps these field names should be in a dictionary somewhere?
    melt_onset_from_ds = cde_ds["melt_onset_day_cdr_seaice_conc"].to_numpy()

    return melt_onset_from_ds


def read_melt_elements(
    *,
    date,
    hemisphere,
    resolution,
    ecdr_data_dir,
):
    """Return the elements from tiecdr needed to calculate melt."""
    tie_ds = read_or_create_and_read_tiecdr_ds(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        ecdr_data_dir=ecdr_data_dir,
    )
    return (
        np.squeeze(tie_ds["cdr_conc"].to_numpy()),
        tie_ds["h18_day_si"].to_numpy(),
        tie_ds["h36_day_si"].to_numpy(),
    )


def create_melt_onset_field(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ecdr_data_dir: Path,
    first_melt_doy: int = MELT_SEASON_FIRST_DOY,
    last_melt_doy: int = MELT_SEASON_LAST_DOY,
    no_melt_flag: int = MELT_ONSET_FILL_VALUE,
) -> np.ndarray | None:
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
    if (day_of_year < first_melt_doy) or (day_of_year > last_melt_doy):
        melt_onset_field = filled_ndarray(
            hemisphere=hemisphere,
            resolution=resolution,
            fill_value=no_melt_flag,
            dtype=np.uint8,
        )
        logger.info(f"returning empty melt_onset_field for {day_of_year}")
        return melt_onset_field
    elif day_of_year == first_melt_doy:
        # This is the first day with melt onset
        prior_melt_onset_field = filled_ndarray(
            hemisphere=hemisphere,
            resolution=resolution,
            fill_value=no_melt_flag,
            dtype=np.uint8,
        )
        logger.info(f"using empty melt_onset_field for prior for {day_of_year}")
    elif (date.year == 2012) and (date <= dt.date(2012, 7, 2)):
        # These are melt season days in first year of AMSR2 data
        prior_melt_onset_field = filled_ndarray(
            hemisphere=hemisphere,
            resolution=resolution,
            fill_value=no_melt_flag,
            dtype=np.uint8,
        )
        logger.info(f"using empty melt_onset_field for prior for {day_of_year}")
    else:
        prior_melt_onset_field = read_melt_onset_field(
            date=date - dt.timedelta(days=1),
            hemisphere=hemisphere,
            resolution=resolution,
            ecdr_data_dir=ecdr_data_dir,
        )
        logger.info(f"using read melt_onset_field for prior for {day_of_year}")

    cdr_conc_ti, tb_h19, tb_h37 = read_melt_elements(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        ecdr_data_dir=ecdr_data_dir,
    )
    is_melted_today = melting(
        concentrations=cdr_conc_ti,
        tb_h19=tb_h19,
        tb_h37=tb_h37,
    )
    # Apply land mask
    land_mask = get_land_mask(
        hemisphere=hemisphere,
        resolution=resolution,
    )
    is_melted_today[0, land_mask.data] = False

    have_prior_melt_values = prior_melt_onset_field != no_melt_flag
    is_missing_prior = prior_melt_onset_field == no_melt_flag
    has_new_melt = is_missing_prior & is_melted_today

    melt_onset_field = np.zeros(prior_melt_onset_field.shape, dtype=np.uint8)
    melt_onset_field[:] = no_melt_flag
    melt_onset_field[have_prior_melt_values] = prior_melt_onset_field[
        have_prior_melt_values
    ]

    melt_onset_field[has_new_melt] = day_of_year

    return melt_onset_field


def complete_daily_ecdr_dataset_for_au_si_tbs(
    *,
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
    tie_ds = read_or_create_and_read_tiecdr_ds(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        ecdr_data_dir=ecdr_data_dir,
    )
    cde_ds = tie_ds.copy()

    melt_onset_field = create_melt_onset_field(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        ecdr_data_dir=ecdr_data_dir,
    )

    # Add the surface-type field
    cde_ds["surface_type"] = get_surfacetype_da(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
    )

    # TODO: Need to ensure that the cdr_seaice_conc field does not have values
    #       where seaice cannot occur, eg over land or lakes

    # Update cde_ds with melt onset info
    if melt_onset_field is None:
        return cde_ds

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

    return cde_ds


def write_cde_netcdf(
    *,
    cde_ds: xr.Dataset,
    output_filepath: Path,
    uncompressed_fields: Iterable[str] = ("crs", "time", "y", "x"),
    excluded_fields: Iterable[str] = [],
    conc_fields: Iterable[str] = [
        "raw_bt_seaice_conc",
        "raw_nt_seaice_conc",
        "cdr_seaice_conc",
    ],
) -> Path:
    """Write the temporally interpolated ECDR to a netCDF file."""
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

    return output_filepath


def make_cdecdr_netcdf(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ecdr_data_dir: Path,
) -> Path:
    logger.info(f"Creating cdecdr for {date=}, {hemisphere=}, {resolution=}")
    cde_ds = complete_daily_ecdr_dataset_for_au_si_tbs(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        ecdr_data_dir=ecdr_data_dir,
    )

    cde_ds = finalize_cdecdr_ds(cde_ds)

    output_path = get_ecdr_filepath(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        ecdr_data_dir=ecdr_data_dir,
    )

    written_cde_ncfile = write_cde_netcdf(
        cde_ds=cde_ds,
        output_filepath=output_path,
    )
    logger.info(f"Wrote complete daily ncfile: {written_cde_ncfile}")

    return output_path


def read_or_create_and_read_cdecdr_ds(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ecdr_data_dir: Path,
    overwrite_cde: bool = False,
) -> xr.Dataset:
    """Read an cdecdr netCDF file, creating it if it doesn't exist.

    Note: this can be recursive because the melt onset field calculation
    requires the prior day's field values during the melt season.
    """
    cde_filepath = get_ecdr_filepath(
        date,
        hemisphere,
        resolution,
        ecdr_data_dir=ecdr_data_dir,
    )

    if overwrite_cde or not cde_filepath.is_file():
        make_cdecdr_netcdf(
            date=date,
            hemisphere=hemisphere,
            resolution=resolution,
            ecdr_data_dir=ecdr_data_dir,
        )
    logger.info(f"Reading cdeCDR file from: {cde_filepath}")
    cde_ds = xr.load_dataset(cde_filepath)

    return cde_ds


def create_cdecdr_for_date_range(
    *,
    hemisphere: Hemisphere,
    start_date: dt.date,
    end_date: dt.date,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ecdr_data_dir: Path,
) -> None:
    """Generate the complete daily ecdr files for a range of dates."""
    for date in date_range(start_date=start_date, end_date=end_date):
        try:
            make_cdecdr_netcdf(
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
                "Failed to create complete daily NetCDF for"
                f" {hemisphere=}, {date=}, {resolution=}."
            )
            # TODO: These error logs should be written to e.g.,
            # `/share/apps/logs/seaice_ecdr`. The `logger` module should be able
            # to handle automatically logging error details to such a file.
            # TODO: Perhaps this function should come from seaice_ecdr
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

    create_cdecdr_for_date_range(
        hemisphere=hemisphere,
        start_date=date,
        end_date=end_date,
        resolution=resolution,
        ecdr_data_dir=ecdr_data_dir,
    )
