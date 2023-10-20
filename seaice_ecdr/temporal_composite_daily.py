"""Routines for generating temporally composited file.

"""

import datetime as dt
import sys
import xarray as xr
from loguru import logger
from pathlib import Path
from pm_icecon.util import standard_output_filename

from seaice_ecdr.initial_daily_ecdr import (
    initial_daily_ecdr_dataset_for_au_si_tbs,
    write_ide_netcdf,
)


# Set the default minimum log notification to "info"
try:
    logger.remove(0)  # Removes previous logger info
    logger.add(sys.stderr, level="INFO")
except ValueError:
    logger.debug(sys.stderr, f"Started logging in {__name__}")
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
    seed_date: dt.date,
    day_range: int = 0,
    skip_future: bool = False,
):
    """Return iterator of dates near a given date."""
    first_date = seed_date - dt.timedelta(days=day_range)
    first_date = seed_date - dt.timedelta(days=day_range)
    if skip_future:
        last_date = seed_date
    else:
        last_date = seed_date + dt.timedelta(days=day_range)

    date = first_date
    while date <= last_date:
        yield date
        date += dt.timedelta(days=1)


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


if __name__ == "__main__":
    date = dt.datetime(2021, 2, 16).date()
    hemisphere = "north"
    resolution = "12"

    gen_temporal_composite_daily(date, hemisphere, resolution)
