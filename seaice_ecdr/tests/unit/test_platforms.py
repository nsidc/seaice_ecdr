"""Test the platforms.py routine for seaice_ecdr."""
import datetime as dt
from typing import get_args

from seaice_ecdr.platforms import (
    PLATFORM_AVAILABILITY,
    # PLATFORM_START_DATES,
    PLATFORM_START_DATES,
    PLATFORMS_FOR_SATS,
    SUPPORTED_SAT,
    _platform_available_for_date,
    _platform_start_dates_are_consistent,
    get_platform_by_date,
)

platform_test_dates = {
    "n07": dt.date(1980, 1, 1),
    "F08": dt.date(1990, 1, 1),
    "F11": dt.date(1992, 6, 1),
    "F13": dt.date(1998, 10, 1),
    "F17": dt.date(2011, 12, 25),
    "ame": dt.date(2005, 3, 15),
    "am2": dt.date(2019, 2, 14),
}


def test_SUPPORTED_SAT():
    cdrv5_sats = (
        "am2",
        "F17",
    )

    for sat in cdrv5_sats:
        assert sat in get_args(SUPPORTED_SAT)


def test_platforms_for_sats():
    for key in PLATFORMS_FOR_SATS.keys():
        assert key in get_args(SUPPORTED_SAT)


def test_default_platform_availability():
    for key in PLATFORM_AVAILABILITY.keys():
        assert key in str(SUPPORTED_SAT)

        pa_dict = PLATFORM_AVAILABILITY[key]
        assert "first_date" in pa_dict
        assert "last_date" in pa_dict


def test_default_platform_start_dates_are_consistent():
    assert _platform_start_dates_are_consistent(
        platform_start_dates=PLATFORM_START_DATES,
        platform_availability=PLATFORM_AVAILABILITY,
    )


def test_platform_availability_by_date():
    all_platforms = list(get_args(SUPPORTED_SAT))

    date_before_any_satellites = dt.date(1900, 1, 1)
    for platform in all_platforms:
        assert not _platform_available_for_date(
            date=date_before_any_satellites,
            platform=platform,
            platform_availability=PLATFORM_AVAILABILITY,
        )

    date_after_dead_satellites = dt.date(2100, 1, 1)
    dead_satellites = (
        "n07",
        "F08",
        "F11",
        "F13",
        "ame",
    )
    for platform in dead_satellites:
        assert not _platform_available_for_date(
            date=date_after_dead_satellites,
            platform=platform,
            platform_availability=PLATFORM_AVAILABILITY,
        )

    for platform in platform_test_dates.keys():
        assert _platform_available_for_date(
            date=platform_test_dates[platform],
            platform=platform,
            platform_availability=PLATFORM_AVAILABILITY,
        )


def test_get_platform_by_date():
    date_list = PLATFORM_START_DATES.keys()
    platform_list = PLATFORM_START_DATES.values()

    for date, expected_platform in zip(date_list, platform_list):
        print(f"testing {date} -> {expected_platform}")

        platform = get_platform_by_date(
            date=date,
            platform_start_dates=PLATFORM_START_DATES,
            platform_availability=PLATFORM_AVAILABILITY,
        )
        assert platform == expected_platform
