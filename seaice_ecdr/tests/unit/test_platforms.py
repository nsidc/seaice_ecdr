"""Test the platforms.py routine for seaice_ecdr."""

import datetime as dt
from collections import OrderedDict
from typing import get_args

import yaml

from seaice_ecdr.platforms import (
    DEFAULT_PLATFORMS,
    PLATFORM_START_DATES,
    SUPPORTED_PLATFORM_ID,
    _get_platform_start_dates,
    _platform_available_for_date,
    _platform_start_dates_are_consistent,
    get_platform_by_date,
    platform_for_id,
)

platform_test_dates: OrderedDict[SUPPORTED_PLATFORM_ID, dt.date] = OrderedDict(
    {
        "n07": dt.date(1980, 1, 1),
        "F08": dt.date(1990, 1, 1),
        "F11": dt.date(1992, 6, 1),
        "F13": dt.date(1998, 10, 1),
        "F17": dt.date(2011, 12, 25),
        "ame": dt.date(2005, 3, 15),
        "am2": dt.date(2019, 2, 14),
    }
)


def test_SUPPORTED_PLATFORM_ID():
    cdrv5_sats = (
        "am2",
        "F17",
    )

    for sat in cdrv5_sats:
        assert sat in get_args(SUPPORTED_PLATFORM_ID)


def test_platforms_for_sats():
    for platform in DEFAULT_PLATFORMS:
        assert platform.id in get_args(SUPPORTED_PLATFORM_ID)


def test_default_platform_start_dates_are_consistent():
    assert _platform_start_dates_are_consistent(
        platform_start_dates=PLATFORM_START_DATES
    )


def test_platform_availability_by_date():
    all_platforms = list(get_args(SUPPORTED_PLATFORM_ID))

    date_before_any_satellites = dt.date(1900, 1, 1)
    for platform_id in all_platforms:
        platform = platform_for_id(platform_id)
        assert not _platform_available_for_date(
            date=date_before_any_satellites,
            platform=platform,
        )

    date_after_dead_satellites = dt.date(2100, 1, 1)
    dead_satellites: tuple[SUPPORTED_PLATFORM_ID] = (  # type: ignore[assignment]
        "n07",
        "F08",
        "F11",
        "F13",
        "ame",
    )
    for platform_id in dead_satellites:
        platform = platform_for_id(platform_id)
        assert not _platform_available_for_date(
            date=date_after_dead_satellites,
            platform=platform,
        )

    for platform_id in platform_test_dates.keys():
        platform = platform_for_id(platform_id)
        assert _platform_available_for_date(
            date=platform_test_dates[platform_id],
            platform=platform,
        )


def test_get_platform_by_date():
    date_list = PLATFORM_START_DATES.keys()
    platform_list = PLATFORM_START_DATES.values()

    for date, expected_platform in zip(date_list, platform_list):
        print(f"testing {date} -> {expected_platform}")

        platform = get_platform_by_date(
            date=date,
        )
        assert platform == expected_platform


def test_override_platform_by_date(monkeypatch, tmpdir):
    override_file = tmpdir / "override_platform_dates.yaml"
    expected_platform_dates = {
        dt.date(1987, 7, 10): "F08",
        dt.date(1991, 12, 3): "F11",
        dt.date(1995, 10, 1): "F13",
        dt.date(2002, 6, 1): "ame",
    }

    with open(override_file, "w") as yaml_file:
        yaml.safe_dump(expected_platform_dates, yaml_file)

    monkeypatch.setenv("PLATFORM_START_DATES_CFG_OVERRIDE_FILE", str(override_file))

    platform_dates = _get_platform_start_dates()

    assert platform_dates == {
        date: platform_for_id(id)  # type: ignore[arg-type]
        for date, id in expected_platform_dates.items()
    }
