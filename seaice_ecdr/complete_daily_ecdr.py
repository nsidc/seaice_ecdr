"""Routines for generating completely filled daily eCDR files.

"""
import click
import traceback

import datetime as dt
import sys
from loguru import logger
import numpy as np

import xarray as xr
from pathlib import Path
from pm_icecon.util import date_range, standard_output_filename

from typing import get_args, Iterable, cast

from pm_tb_data._types import Hemisphere, NORTH, SOUTH
from pm_tb_data.fetch.au_si import AU_SI_RESOLUTIONS
from seaice_ecdr.cli.util import datetime_to_date
from seaice_ecdr.constants import INITIAL_DAILY_OUTPUT_DIR
from seaice_ecdr.temporal_composite_daily import make_tiecdr_netcdf

from seaice_ecdr.melt import (
    melting,
    MELT_SEASON_FIRST_DOY,
    MELT_SEASON_LAST_DOY,
    MELT_ONSET_FILL_VALUE,
)


def get_ecdr_filename(
    date,
    hemisphere,
    resolution,
    directory_name: Path | None = None,
    file_label: str | None = None,
) -> Path:
    """Return the initial daily eCDR file path."""
    if directory_name is None:
        raise RuntimeError("No directory_name provided")
    if file_label is None:
        raise RuntimeError("No file_label provided")
    if "km" not in resolution:
        resolution = f"{resolution}km"
    ecdr_filename = standard_output_filename(
        algorithm=file_label,
        hemisphere=hemisphere,
        date=date,
        sat="ausi",
        resolution=resolution,
    )
    ecdr_filepath = Path(
        directory_name,
        ecdr_filename,
    )

    return ecdr_filepath


# TODO: Perhaps this isn't needed?  The necessary precursor files can
#       be computed recursively rather than iterated from the beginning
#       of the calendar year.
def iter_cdecdr_dates(
    target_date: dt.date,
    date_step: int = 1,
):
    """Return iterator of dates from start of year to a given date."""
    earliest_date = dt.date(target_date.year, 1, 1)

    date = earliest_date
    while date <= target_date:
        yield date
        date += dt.timedelta(days=date_step)


def read_or_create_and_read_tiecdr_ds(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: AU_SI_RESOLUTIONS,
    tie_dir: Path,
) -> xr.Dataset:
    """Read an tiecdr netCDF file, creating it if it doesn't exist."""
    tie_filepath = get_ecdr_filename(
        date,
        hemisphere,
        resolution,
        directory_name=tie_dir,
        file_label="tiecdr",
    )
    # TODO: This only creates if file is missing.  We may want an overwrite opt
    if not tie_filepath.is_file():
        make_tiecdr_netcdf(
            date=date,
            hemisphere=hemisphere,
            resolution=resolution,
            output_dir=tie_dir,
            ide_dir=tie_dir,
        )
    logger.info(f"Reading tieCDR file from: {tie_filepath}")
    tie_ds = xr.open_dataset(tie_filepath)

    return tie_ds


def filled_ndarray(
    *,
    hemisphere,
    resolution,
    fill_value,
    dtype=np.uint8,
) -> np.ndarray:
    """Return an array of the shape for this hem/res filled with fill_value."""
    if hemisphere == NORTH and resolution == "12":
        array_shape = (896, 608)
    elif hemisphere == SOUTH and resolution == "12":
        array_shape = (664, 632)
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
    cde_dir,
) -> np.ndarray:
    """Return the melt onset field for this complete daily eCDR file."""
    cde_ds = read_or_create_and_read_cdecdr_ds(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        cde_dir=cde_dir,
        mask_and_scale=False,
    )

    return cde_ds["melt_onset"].to_numpy()


def read_melt_elements(
    *,
    date,
    hemisphere,
    resolution,
    tie_dir,
):
    """Return the elements from tiecdr needed to calculate melt."""
    tie_ds = read_or_create_and_read_tiecdr_ds(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        tie_dir=tie_dir,
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
    resolution: AU_SI_RESOLUTIONS,
    tie_dir: Path,
    cde_dir: Path,
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
    else:
        prior_melt_onset_field = read_melt_onset_field(
            date=date - dt.timedelta(days=1),
            hemisphere=hemisphere,
            resolution=resolution,
            cde_dir=cde_dir,
        )
        logger.info(f"using read melt_onset_field for prior for {day_of_year}")

    cdr_conc_ti, tb_h19, tb_h37 = read_melt_elements(
        date=date, hemisphere=hemisphere, resolution=resolution, tie_dir=tie_dir
    )
    is_melted_today = melting(
        concentrations=cdr_conc_ti,
        tb_h19=tb_h19,
        tb_h37=tb_h37,
    )

    have_prior_melt_values = prior_melt_onset_field != no_melt_flag
    is_missing_prior = prior_melt_onset_field == no_melt_flag
    has_new_melt = is_missing_prior & is_melted_today

    melt_onset_field = np.zeros(prior_melt_onset_field.shape, dtype=np.uint8)
    melt_onset_field[:] = no_melt_flag
    melt_onset_field[have_prior_melt_values] = prior_melt_onset_field[
        have_prior_melt_values
    ]

    melt_onset_field[has_new_melt] = day_of_year

    # TODO: Do we want to modify QA flag here too?
    return melt_onset_field


def complete_daily_ecdr_dataset_for_au_si_tbs(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: AU_SI_RESOLUTIONS,
    interp_range: int = 5,
    tie_dir: Path,
) -> xr.Dataset:
    """Create xr dataset containing the complete daily enhanced CDR.

    This function returns
    - a Dataset containing
      - The melt onset field
      - All appropriate QA and QC fields
    """
    tie_ds = read_or_create_and_read_tiecdr_ds(
        date=date, hemisphere=hemisphere, resolution=resolution, tie_dir=tie_dir
    )
    cde_ds = tie_ds.copy()
    melt_onset_field = create_melt_onset_field(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        tie_dir=tie_dir,
        cde_dir=tie_dir,
    )

    # Update cde_ds with melt onset info
    cde_ds["melt_onset"] = (
        ("y", "x"),
        melt_onset_field,
        {
            "_FillValue": 255,
            "grid_mapping": "crs",
            "standard_name": "status_flag",
            # Am removing valid range because it causes 255 to plot as NaN
            # "valid_range": [
            #   np.uint8(MELT_SEASON_FIRST_DOY),
            #   np.uint8(MELT_SEASON_LAST_DOY)
            # ],
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

    return cde_ds


def write_cde_netcdf(
    *,
    cde_ds: xr.Dataset,
    output_filepath: Path,
    uncompressed_fields: Iterable[str] = ("crs", "time", "y", "x"),
    excluded_fields: Iterable[str] = [],
) -> Path:
    """Write the temporally interpolated ECDR to a netCDF file."""
    logger.info(f"Writing netCDF of initial_daily eCDR file to: {output_filepath}")
    for excluded_field in excluded_fields:
        if excluded_field in cde_ds.variables.keys():
            cde_ds = cde_ds.drop_vars(excluded_field)

    nc_encoding = {}
    for varname in cde_ds.variables.keys():
        varname = cast(str, varname)
        if varname not in uncompressed_fields:
            nc_encoding[varname] = {"zlib": True}

    cde_ds.to_netcdf(
        output_filepath,
        encoding=nc_encoding,
    )

    return output_filepath


def make_cdecdr_netcdf(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: AU_SI_RESOLUTIONS,
    output_dir: Path,
    interp_range: int = 5,
    fill_the_pole_hole: bool = True,
) -> None:
    logger.info(f"Creating cdecdr for {date=}, {hemisphere=}, {resolution=}")
    cde_ds = complete_daily_ecdr_dataset_for_au_si_tbs(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        interp_range=interp_range,
        tie_dir=output_dir,
    )
    # TODO: Perhaps this function should come from seaice_ecdr, not pm_icecon?
    output_fn = get_ecdr_filename(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        file_label="cdecdr",
        directory_name=output_dir,
    )
    output_path = Path(output_dir) / Path(output_fn)

    written_cde_ncfile = write_cde_netcdf(
        cde_ds=cde_ds,
        output_filepath=output_path,
    )
    logger.info(f"Wrote complete daily ncfile: {written_cde_ncfile}")


def read_or_create_and_read_cdecdr_ds(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: AU_SI_RESOLUTIONS,
    cde_dir: Path,
    mask_and_scale: bool = True,
) -> xr.Dataset:
    """Read an cdecdr netCDF file, creating it if it doesn't exist.

    Note: this can be recursive because the melt onset field calculation
    requires the prior day's field values during the melt season.
    """
    cde_filepath = get_ecdr_filename(
        date,
        hemisphere,
        resolution,
        directory_name=cde_dir,
        file_label="cdecdr",
    )
    # TODO: This only creates if file is missing.  We may want an overwrite opt
    if not cde_filepath.is_file():
        make_cdecdr_netcdf(
            date=date,
            hemisphere=hemisphere,
            resolution=resolution,
            output_dir=cde_dir,
        )
    logger.info(f"Reading cdeCDR file from: {cde_filepath}")
    cde_ds = xr.open_dataset(cde_filepath, mask_and_scale=mask_and_scale)

    return cde_ds


def create_cdecdr_for_date_range(
    *,
    hemisphere: Hemisphere,
    start_date: dt.date,
    end_date: dt.date,
    resolution: AU_SI_RESOLUTIONS,
    output_dir: Path,
) -> None:
    """Generate the complete daily ecdr files for a range of dates."""
    for date in date_range(start_date=start_date, end_date=end_date):
        try:
            make_cdecdr_netcdf(
                date=date,
                hemisphere=hemisphere,
                resolution=resolution,
                output_dir=output_dir,
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
            err_filename = get_ecdr_filename(
                date=date,
                hemisphere=hemisphere,
                resolution=resolution,
                file_label="cdecdr",
            )
            err_filename = Path(err_filename, ".error")
            logger.info(f"Writing error info to {err_filename}")
            with open(output_dir / err_filename, "w") as f:
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
    default=INITIAL_DAILY_OUTPUT_DIR,
    show_default=True,
)
@click.option(
    "-r",
    "--resolution",
    required=True,
    type=click.Choice(get_args(AU_SI_RESOLUTIONS)),
)
def cli(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    output_dir: Path,
    resolution: AU_SI_RESOLUTIONS,
) -> None:
    """Run the temporal composite daily ECDR algorithm with AMSR2 data.

    This requires the creation/existence of temporally interpolated eCDR
    (tiecdr) files.

    TODO: eventually we want to be able to specify: date, grid (grid includes
    projection, resolution, and bounds), and TBtype (TB type includes source and
    methodology for getting those TBs onto the grid)
    """
    create_cdecdr_for_date_range(
        hemisphere=hemisphere,
        start_date=date,
        end_date=date,
        resolution=resolution,
        output_dir=output_dir,
    )
