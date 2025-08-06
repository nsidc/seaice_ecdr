"""Test the platforms.py routine for seaice_ecdr."""

import datetime as dt
from collections import OrderedDict
from pathlib import Path
from typing import get_args

import pytest
import yaml
from pydantic import ValidationError

from seaice_ecdr.platforms import (
    PLATFORM_CONFIG,
    SUPPORTED_PLATFORM_ID,
)
from seaice_ecdr.platforms.config import (
    SUPPORTED_PLATFORMS,
    _get_platform_config,
    is_dmsp_platform,
)
from seaice_ecdr.platforms.models import (
    DateRange,
    Platform,
    PlatformConfig,
    PlatformStartDate,
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
    cdrv5_platform_ids = (
        "am2",
        "F17",
    )

    for platform_id in cdrv5_platform_ids:
        assert platform_id in get_args(SUPPORTED_PLATFORM_ID)


def test_platforms_for_sats():
    for platform in SUPPORTED_PLATFORMS:
        assert platform.id in get_args(SUPPORTED_PLATFORM_ID)


def test_get_platform_by_date():
    date_before_any_satellites = dt.date(1900, 1, 1)
    with pytest.raises(RuntimeError):
        PLATFORM_CONFIG.get_platform_by_date(date_before_any_satellites)

    expected_f13_date = dt.date(1995, 11, 1)
    platform = PLATFORM_CONFIG.get_platform_by_date(expected_f13_date)
    assert platform.id == "F13"


def test_override_platform_by_date(monkeypatch, tmpdir):
    override_file = Path(tmpdir / "override_platform_dates.yaml")
    expected_platform_dates = {
        "cdr_platform_start_dates": [
            {"platform_id": "F08", "start_date": dt.date(1987, 8, 12)},
            {"platform_id": "F11", "start_date": dt.date(1992, 6, 15)},
        ],
    }

    with open(override_file, "w") as yaml_file:
        yaml.safe_dump(expected_platform_dates, yaml_file)

    monkeypatch.setenv("PLATFORM_START_DATES_CONFIG_FILEPATH", str(override_file))
    platform_config = _get_platform_config()

    assert len(platform_config.cdr_platform_start_dates) == 2
    assert platform_config.cdr_platform_start_dates[0].platform_id == "F08"
    assert platform_config.cdr_platform_start_dates[0].start_date == dt.date(
        1987, 8, 12
    )
    assert platform_config.cdr_platform_start_dates[1].platform_id == "F11"
    assert platform_config.cdr_platform_start_dates[1].start_date == dt.date(
        1992, 6, 15
    )


def test__get_platform_config():
    platform_cfg = _get_platform_config()

    assert len(platform_cfg.platforms) >= 1
    assert len(platform_cfg.cdr_platform_start_dates) >= 1
    assert PLATFORM_CONFIG == platform_cfg


def test_platform_config_validation_error():
    # Tests `validate_platform_start_dates_platform_in_platforms`
    with pytest.raises(ValidationError, match=r"Did not find am2 in platform list.*"):
        PlatformConfig(
            platforms=[
                Platform(
                    id="ame",
                    name="fooname",
                    sensor="sensorname",
                    date_range=DateRange(
                        first_date=dt.date(2012, 7, 2),
                        last_date=None,
                    ),
                )
            ],
            cdr_platform_start_dates=[
                PlatformStartDate(
                    platform_id="am2",
                    start_date=dt.date(2022, 1, 1),
                )
            ],
        )

    # tests `validate_platform_start_dates_in_order`
    with pytest.raises(
        ValidationError, match=r"Platform start dates are not sequential.*"
    ):
        PlatformConfig(
            platforms=[
                Platform(
                    id="F13",
                    name="fooname",
                    sensor="sensorname",
                    date_range=DateRange(
                        first_date=dt.date(1991, 1, 1),
                        last_date=dt.date(1992, 1, 1),
                    ),
                ),
                Platform(
                    id="am2",
                    name="fooname",
                    sensor="sensorname",
                    date_range=DateRange(
                        first_date=dt.date(2012, 7, 2),
                        last_date=None,
                    ),
                ),
            ],
            cdr_platform_start_dates=[
                PlatformStartDate(platform_id="am2", start_date=dt.date(2022, 1, 1)),
                PlatformStartDate(platform_id="F13", start_date=dt.date(1991, 1, 1)),
            ],
        )

    # tests `validate_platform_start_date_in_platform_date_range`
    # First error date is before the date range, and the second is after.
    for err_start_date in (dt.date(2000, 1, 1), dt.date(2015, 1, 1)):
        with pytest.raises(
            ValidationError, match=r".*is outside of the platform's date range.*"
        ):
            PlatformConfig(
                platforms=[
                    Platform(
                        id="ame",
                        name="fooname",
                        sensor="sensorname",
                        date_range=DateRange(
                            first_date=dt.date(2012, 7, 2),
                            last_date=dt.date(2012, 12, 31),
                        ),
                    )
                ],
                cdr_platform_start_dates=[
                    PlatformStartDate(
                        platform_id="ame",
                        # Start date is before the first available date
                        start_date=err_start_date,
                    )
                ],
            )


def test_platform_date_range_validation_error():
    # tests `validate_date_range`
    with pytest.raises(ValidationError, match=r".*First date.*is after last date.*"):
        DateRange(
            first_date=dt.date(2021, 1, 1),
            last_date=dt.date(2020, 1, 1),
        )


def test_get_first_platform_start_date():
    actual = PLATFORM_CONFIG.get_first_platform_start_date()

    min_date = min(
        [
            platform_start_date.start_date
            for platform_start_date in PLATFORM_CONFIG.cdr_platform_start_dates
        ]
    )

    assert actual == min_date


def test_platform_available_for_date():
    config = PlatformConfig(
        platforms=[
            Platform(
                id="ame",
                name="fooname",
                sensor="sensorname",
                date_range=DateRange(
                    first_date=dt.date(2012, 7, 2),
                    last_date=dt.date(2012, 12, 31),
                ),
            )
        ],
        cdr_platform_start_dates=[
            PlatformStartDate(
                platform_id="ame",
                start_date=dt.date(2012, 7, 2),
            )
        ],
    )

    assert config.platform_available_for_date(
        platform_id="ame", date=dt.date(2012, 7, 2)
    )
    assert config.platform_available_for_date(
        platform_id="ame", date=dt.date(2012, 12, 31)
    )
    assert not config.platform_available_for_date(
        platform_id="ame", date=dt.date(2012, 6, 30)
    )
    assert not config.platform_available_for_date(
        platform_id="ame", date=dt.date(2013, 6, 30)
    )


def test_is_dmsp_platform():
    for platform_id in ("F18", "F17", "F13", "F11", "F08", "n07"):
        assert is_dmsp_platform(platform_id)

    assert not is_dmsp_platform("am2")
    assert not is_dmsp_platform("ame")
