"""Generate a sample ide file for use in development.

./gen_ide_sample.py

Modeled after code in the integration test case
"""

import datetime as dt
import sys

from loguru import logger

from seaice_ecdr.initial_daily_ecdr import (
    compute_initial_daily_ecdr_dataset as compute_idecdr_ds,
)

# Set the default minimum log notification to Warning
logger.remove(0)  # Removes previous logger info
logger.add(sys.stderr, level="INFO")


def gen_sample_idecdr_dataset(
    date,
    hemisphere,
    resolution,
    sample_filename=None,
):
    """Generate sample initial daily cdr file from seaice_ecdr repo."""
    if sample_filename is None:
        sample_filename = (
            f"sample_idecdr_{hemisphere}_{resolution}_"
            + f'{date.strftime("%Y%m%d")}.nc'
        )
    """Set up sample data set using pm_icecon."""
    log_string = f"""

    Creating sample seaice_ecdr initial daily data set with:
        date: {date}
        hemisphere: {hemisphere}
        resolution: {resolution}

    Output file name:
        {sample_filename}

    """
    logger.info(log_string)

    ide_ds = compute_idecdr_ds(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
    )
    ide_ds.to_netcdf(sample_filename)
    logger.info(
        f"""

    Wrote: {sample_filename}

    """
    )


if __name__ == "__main__":
    date = dt.datetime(2021, 4, 5).date()
    hemisphere = "north"
    resolution = "12"

    gen_sample_idecdr_dataset(date, hemisphere, resolution)
