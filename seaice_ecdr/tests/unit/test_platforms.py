"""Test the platforms.py routine for seaice_ecdr."""
from typing import get_args

from seaice_ecdr.platforms import (
    PLATFORM_AVAILABILITY,
    PLATFORM_START_DATES,
    PLATFORMS_FOR_SATS,
    SUPPORTED_SAT,
    _platform_start_dates_are_consistent,
)


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
