import datetime as dt
from pathlib import Path

from seaice_ecdr.complete_daily_ecdr import get_ecdr_filepath
from seaice_ecdr.daily_aggregate import (
    _get_daily_complete_filepaths_for_year,
)
from seaice_ecdr.platforms import PLATFORM_CONFIG
from seaice_ecdr.util import date_range


def test_get_daily_complete_filepaths_for_year(fs):
    complete_output_dir = Path("/path/to/data/dir/complete")
    fs.create_dir(complete_output_dir)

    # target year
    year = 2021

    # Before the target year
    for date in date_range(
        start_date=dt.date(year - 1, 12, 1), end_date=dt.date(year - 1, 12, 30)
    ):
        platform = PLATFORM_CONFIG.get_platform_by_date(date)
        fp = get_ecdr_filepath(
            platform_id=platform.id,
            date=date,
            hemisphere="north",
            resolution="12.5",
            # TODO: this isn't "right", although it works.
            intermediate_output_dir=complete_output_dir,
            is_nrt=False,
        )

        fs.create_file(fp)

    # During the target year
    expected_files = []
    for date in date_range(
        start_date=dt.date(year, 12, 1), end_date=dt.date(year, 12, 30)
    ):
        platform = PLATFORM_CONFIG.get_platform_by_date(date)
        fp = get_ecdr_filepath(
            date=date,
            platform_id=platform.id,
            hemisphere="north",
            resolution="12.5",
            intermediate_output_dir=complete_output_dir,
            is_nrt=False,
        )

        fs.create_file(fp)
        expected_files.append(fp)

    actual = _get_daily_complete_filepaths_for_year(
        year=year,
        complete_output_dir=complete_output_dir,
        hemisphere="north",
        resolution="12.5",
    )

    assert actual == expected_files
