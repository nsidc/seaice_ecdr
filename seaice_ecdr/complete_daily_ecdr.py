"""Routines for generating completely filled daily eCDR files.

"""

import datetime as dt
import sys
from loguru import logger


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
