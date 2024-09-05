"""platforms.py.

Routines for dealing with the platform (satellite) and sensors
for CDRv5


TODO: There are a couple of date ranges for which we do not want
        to produce data files:
          Aug 12-24, 1984 because there is no SMMR data
          Dec 3, 1987 - Jan 12, 1988 because no F08 data
        Also, anything prior to the start of the data record,
          eg prior to Oct 25, 1978
"""

import datetime as dt
import os
from collections import OrderedDict
from typing import cast, get_args

import yaml
from loguru import logger
from pydantic import BaseModel

from seaice_ecdr._types import SUPPORTED_SAT


class DateRange(BaseModel):
    first_date: dt.date
    last_date: dt.date | None


class Platform(BaseModel):
    # E.g., "DMSP 5D-3/F17 > Defense Meteorological Satellite Program-F17"
    name: str
    # GCMD sensor name. E.g., SSMIS > Special Sensor Microwave Imager/Sounder
    sensor: str
    # E.g., "F17"
    id: SUPPORTED_SAT
    date_range: DateRange


AM2_PLATFORM = Platform(
    name="GCOM-W1 > Global Change Observation Mission 1st-Water",
    sensor="AMSR2 > Advanced Microwave Scanning Radiometer 2",
    # TODO: rename as ID? Each platform should have a unique ID to identify it
    # in other parts of the code.
    id="am2",
    date_range=DateRange(
        first_date=dt.date(2012, 7, 2),
        last_date=None,
    ),
)

AME_PLATFORM = Platform(
    name="Aqua > Earth Observing System, Aqua",
    sensor="AMSR-E > Advanced Microwave Scanning Radiometer-EOS",
    id="ame",
    date_range=DateRange(
        first_date=dt.date(2002, 6, 1),
        last_date=dt.date(2011, 10, 3),
    ),
)

F17_PLATFORM = Platform(
    name="DMSP 5D-3/F17 > Defense Meteorological Satellite Program-F17",
    sensor="SSMIS > Special Sensor Microwave Imager/Sounder",
    id="F17",
    date_range=DateRange(
        first_date=dt.date(2008, 1, 1),
        last_date=None,
    ),
)

F13_PLATFORM = Platform(
    name="DMSP 5D-2/F13 > Defense Meteorological Satellite Program-F13",
    sensor="SSM/I > Special Sensor Microwave/Imager",
    id="F13",
    date_range=DateRange(
        first_date=dt.date(1995, 10, 1),
        last_date=dt.date(2007, 12, 31),
    ),
)
F11_PLATFORM = Platform(
    name="DMSP 5D-2/F11 > Defense Meteorological Satellite Program-F11",
    sensor="SSM/I > Special Sensor Microwave/Imager",
    id="F11",
    date_range=DateRange(
        first_date=dt.date(1991, 12, 3),
        last_date=dt.date(1995, 9, 30),
    ),
)
F08_PLATFORM = Platform(
    name="DMSP 5D-2/F8 > Defense Meteorological Satellite Program-F8",
    sensor="SSM/I > Special Sensor Microwave/Imager",
    id="F08",
    date_range=DateRange(
        first_date=dt.date(1987, 7, 10),
        last_date=dt.date(1991, 12, 2),
    ),
)

N07_PLATFORM = Platform(
    name="Nimbus-7",
    sensor="SMMR > Scanning Multichannel Microwave Radiometer",
    id="n07",
    date_range=DateRange(
        first_date=dt.date(1978, 10, 25),
        last_date=dt.date(1987, 7, 9),
    ),
)

DEFAULT_PLATFORMS = [
    AM2_PLATFORM,
    AME_PLATFORM,
    F17_PLATFORM,
    F13_PLATFORM,
    F11_PLATFORM,
    F08_PLATFORM,
    N07_PLATFORM,
]


DEFAULT_PLATFORM_START_DATES: OrderedDict[dt.date, Platform] = OrderedDict(
    {
        dt.date(1978, 10, 25): N07_PLATFORM,
        dt.date(1987, 7, 10): F08_PLATFORM,
        dt.date(1991, 12, 3): F11_PLATFORM,
        dt.date(1995, 10, 1): F13_PLATFORM,
        dt.date(2002, 6, 1): AME_PLATFORM,  # AMSR-E is first AMSR sat
        # F17 starts while AMSR-E is up, on 2008-01-01. We don't use
        # F17 until 2011-10-04.
        dt.date(2011, 10, 4): F17_PLATFORM,
        dt.date(2012, 7, 3): AM2_PLATFORM,  # AMSR2
    }
)


def platform_for_id(id: SUPPORTED_SAT) -> Platform:
    for platform in DEFAULT_PLATFORMS:
        if platform.id == id:
            return platform

    err_msg = f"Failed to find platform for {id=}"
    raise RuntimeError(err_msg)


def read_platform_start_dates_cfg_override(
    start_dates_cfg_filename,
) -> OrderedDict[dt.date, Platform]:
    """The "platform_start_dates" dictionary is an OrderedDict
    of keys (dates) with corresponding platform ids (values)

    Note: It seems like yaml can't safe_load() an OrderedDict.
    """
    try:
        with open(start_dates_cfg_filename, "r") as config_file:
            file_dict = yaml.safe_load(config_file)
    except FileNotFoundError:
        raise RuntimeError(
            f"Could not find specified start_dates config file: {start_dates_cfg_filename}"
        )

    platform_start_dates = OrderedDict(file_dict)

    # Assert that the keys are ordered.
    assert sorted(platform_start_dates.keys()) == list(platform_start_dates.keys())
    # Assert that the platforms are in our list of supported sats.
    assert all(
        [
            platform in get_args(SUPPORTED_SAT)
            for platform in platform_start_dates.values()
        ]
    )

    platform_start_dates_with_platform = {
        date: platform_for_id(id) for date, id in platform_start_dates.items()
    }

    platform_start_dates_with_platform = cast(
        OrderedDict[dt.date, Platform], platform_start_dates_with_platform
    )

    return platform_start_dates_with_platform


def _platform_available_for_date(
    *,
    date: dt.date,
    platform: Platform,
) -> bool:
    """Determine if platform is available on this date."""
    # First, verify the values of the first listed platform
    first_available_date = platform.date_range.first_date
    if date < first_available_date:
        print(
            f"""
            Satellite {platform.id} is not available on date {date}.
            {date} is before first_available_date {first_available_date}
            Date info: {platform.date_range}
            """
        )
        return False

    last_available_date = platform.date_range.last_date
    # If the last available date is `None`, then there is no end date and we
    # should treat the date as if the platform is available.
    if last_available_date is None:
        return True

    if date > last_available_date:
        print(
            f"""
            Satellite {platform} is not available on date {date}.
            {date} is after last_available_date {last_available_date}
            Date info: {platform.date_range}
            """
        )
        return False

    return True


def _platform_start_dates_are_consistent(
    *,
    platform_start_dates: OrderedDict[dt.date, Platform],
) -> bool:
    """Return whether the provided start date structure is valid."""
    date_list = list(platform_start_dates.keys())
    platform_list = list(platform_start_dates.values())
    try:
        date = date_list[0]
        platform = platform_list[0]
        assert _platform_available_for_date(
            date=date,
            platform=platform,
        )

        for idx in range(1, len(date_list)):
            date = date_list[idx]
            platform = platform_list[idx]

            # Check the end of the prior platform's date range
            prior_date = date - dt.timedelta(days=1)
            prior_platform = platform_list[idx - 1]
            assert _platform_available_for_date(
                date=prior_date,
                platform=prior_platform,
            )

            # Check this platform's first available date
            assert _platform_available_for_date(
                date=date,
                platform=platform,
            )
    except AssertionError:
        raise RuntimeError(
            f"""
        platform start dates are not consistent
        platform_start_dates: {platform_start_dates}
        platforms: {DEFAULT_PLATFORMS}
        """
        )

    return True


def _get_platform_start_dates() -> OrderedDict[dt.date, Platform]:
    """Return dict of start dates for differnt platforms.

    Platform start dates can be overridden via a YAML override file specified by
    the `PLATFORM_START_DATES_CFG_OVERRIDE_FILE` envvar.
    """

    if override_file := os.environ.get("PLATFORM_START_DATES_CFG_OVERRIDE_FILE"):
        _platform_start_dates = read_platform_start_dates_cfg_override(override_file)
        logger.info(f"Read platform start dates from {override_file}")
    # TODO: it's clear that we should refactor to support passing in custom
    # platform start dates programatically. This is essentially global state and
    # it makes it very difficult to test out different combinations as a result.
    elif forced_platform_id := os.environ.get("FORCE_PLATFORM"):
        if forced_platform_id not in get_args(SUPPORTED_SAT):
            raise RuntimeError(
                f"The forced platform ({forced_platform_id}) is not a supported platform."
            )

        forced_platform_id = cast(SUPPORTED_SAT, forced_platform_id)

        forced_platform = platform_for_id(forced_platform_id)
        first_date_of_forced_platform = forced_platform.date_range.first_date
        _platform_start_dates = OrderedDict(
            {
                first_date_of_forced_platform: forced_platform,
            }
        )

    else:
        _platform_start_dates = DEFAULT_PLATFORM_START_DATES

    _platform_start_dates = cast(OrderedDict[dt.date, Platform], _platform_start_dates)

    assert _platform_start_dates_are_consistent(
        platform_start_dates=_platform_start_dates
    )

    return _platform_start_dates


PLATFORM_START_DATES = _get_platform_start_dates()


def get_platform_by_date(
    date: dt.date,
) -> Platform:
    """Return the platform for this date."""
    start_date_list = list(PLATFORM_START_DATES.keys())
    platform_list = list(PLATFORM_START_DATES.values())

    if date < start_date_list[0]:
        raise RuntimeError(
            f"""
           date {date} too early.
           First start_date: {start_date_list[0]}
           """
        )

    if date >= start_date_list[-1]:
        return platform_list[-1]

    return_platform = platform_list[0]
    for start_date, latest_platform in zip(start_date_list[1:], platform_list[1:]):
        if date >= start_date:
            return_platform = latest_platform
            continue
        else:
            break

    return return_platform


def get_first_platform_start_date() -> dt.date:
    """Return the start date of the first platform."""
    earliest_date = min(PLATFORM_START_DATES.keys())

    return earliest_date
