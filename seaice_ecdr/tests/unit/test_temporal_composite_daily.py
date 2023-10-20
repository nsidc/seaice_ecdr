"""Tests for initial daily ECDR generation."""

# TODO: The tests should probably not require "real" data, but
#  should work with mock data.  Or else, they should be moved to
#  tests/integration/ directory.

import datetime as dt
import sys
from pathlib import Path

from loguru import logger


from seaice_ecdr.temporal_composite_daily import (
    get_sample_idecdr_filename,
    iter_dates_near_date,
    get_standard_initial_daily_ecdr_filename,
)


# Set the default minimum log notification to Warning
try:
    logger.remove(0)  # Removes previous logger info
    logger.add(sys.stderr, level="WARNING")
except ValueError:
    logger.debug(sys.stderr, f"Started logging in {__name__}")
    logger.add(sys.stderr, level="WARNING")


date = dt.date(2021, 2, 19)
hemisphere = "north"
resolution = "12"


def test_sample_filename_generation():
    """Verify creation of sample filename."""

    expected_filename = "sample_idecdr_north_12_20210219.nc"
    sample_filename = get_sample_idecdr_filename(date, hemisphere, resolution)

    assert sample_filename == expected_filename


def test_date_iterator():
    """Verify operation of date temporal composite's date iterator."""
    # Single argument to iterator should yield that date
    expected_date = date
    for iter_date in iter_dates_near_date(date):
        assert iter_date == expected_date

    # range of dates should start from day before and end with day after
    expected_dates = [date - dt.timedelta(days=1), date, date + dt.timedelta(days=1)]
    for idx, iter_date in enumerate(iter_dates_near_date(date, day_range=1)):
        assert iter_date == expected_dates[idx]

    # Skipping future means no dates after the seed_date are provided
    expected_dates = [date - dt.timedelta(days=2), date - dt.timedelta(days=1), date]
    for idx, iter_date in enumerate(
        iter_dates_near_date(date, day_range=2, skip_future=True)
    ):
        assert iter_date == expected_dates[idx]


def test_access_to_standard_output_filename():
    """Verify that standard output file names can be generated."""
    sample_ide_filepath = get_standard_initial_daily_ecdr_filename(
        date, hemisphere, resolution, output_directory=""
    )
    expected_filepath = Path("idecdr_NH_20210219_ausi_12km.nc")

    assert sample_ide_filepath == expected_filepath
