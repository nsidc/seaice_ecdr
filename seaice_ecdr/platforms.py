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
from functools import cache
from typing import cast, get_args

import yaml
from loguru import logger

from seaice_ecdr._types import SUPPORTED_SAT

# TODO: De-dup with nc_attrs.py
# Here’s what the GCMD platform long name should be based on sensor/platform short name:
PLATFORMS_FOR_SATS: dict[SUPPORTED_SAT, str] = dict(
    am2="GCOM-W1 > Global Change Observation Mission 1st-Water",
    ame="Aqua > Earth Observing System, Aqua",
    F17="DMSP 5D-3/F17 > Defense Meteorological Satellite Program-F17",
    F13="DMSP 5D-2/F13 > Defense Meteorological Satellite Program-F13",
    F11="DMSP 5D-2/F11 > Defense Meteorological Satellite Program-F11",
    F08="DMSP 5D-2/F8 > Defense Meteorological Satellite Program-F8",
    n07="Nimbus-7",
)


# TODO: De-dup with nc_attrs.py
# Here’s what the GCMD sensor name should be based on sensor short name:
SENSORS_FOR_SATS: dict[SUPPORTED_SAT, str] = dict(
    am2="AMSR2 > Advanced Microwave Scanning Radiometer 2",
    ame="AMSR-E > Advanced Microwave Scanning Radiometer-EOS",
    F17="SSMIS > Special Sensor Microwave Imager/Sounder",
    # TODO: de-dup SSM/I text?
    F13="SSM/I > Special Sensor Microwave/Imager",
    F11="SSM/I > Special Sensor Microwave/Imager",
    F08="SSM/I > Special Sensor Microwave/Imager",
    n07="SMMR > Scanning Multichannel Microwave Radiometer",
)


# These first and last dates were adapted from the cdrv4 file
#   https://bitbucket.org/nsidc/seaice_cdr/src/master/source/config/cdr.yml
# of commit:
#   https://bitbucket.org/nsidc/seaice_cdr/commits/c9c632e73530554d8acfac9090baeb1e35755897
PLATFORM_AVAILABILITY: OrderedDict[SUPPORTED_SAT, dict] = OrderedDict(
    n07={"first_date": dt.date(1978, 10, 25), "last_date": dt.date(1987, 7, 9)},
    F08={"first_date": dt.date(1987, 7, 10), "last_date": dt.date(1991, 12, 2)},
    F11={"first_date": dt.date(1991, 12, 3), "last_date": dt.date(1995, 9, 30)},
    F13={"first_date": dt.date(1995, 10, 1), "last_date": dt.date(2007, 12, 31)},
    F17={"first_date": dt.date(2008, 1, 1), "last_date": None},
    # ame={"first_date": dt.date(2002, 6, 1), "last_date": dt.date(2011, 10, 3)},
    # am2={"first_date": dt.date(2012, 7, 2), "last_date": None},
)


def read_platform_start_dates_cfg_override(
    start_dates_cfg_filename,
) -> OrderedDict[dt.date, SUPPORTED_SAT]:
    """The "platform_start_dates" dictionary is an OrderedDict
    of keys (dates) with corresponding platforms/sats (values)

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

    return platform_start_dates


@cache
def get_platform_start_dates() -> OrderedDict[dt.date, SUPPORTED_SAT]:
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
    elif forced_platform := os.environ.get("FORCE_PLATFORM"):
        if forced_platform not in get_args(SUPPORTED_SAT):
            raise RuntimeError(
                f"The forced platform ({forced_platform}) is not a supported platform."
            )

        forced_platform = cast(SUPPORTED_SAT, forced_platform)

        first_date_of_forced_platform = PLATFORM_AVAILABILITY[forced_platform][
            "first_date"
        ]
        _platform_start_dates = OrderedDict(
            {
                first_date_of_forced_platform: forced_platform,
            }
        )

    else:
        _platform_start_dates = OrderedDict(
            {
                dt.date(1978, 10, 25): "n07",
                dt.date(1987, 7, 10): "F08",
                dt.date(1991, 12, 3): "F11",
                dt.date(1995, 10, 1): "F13",
                # dt.date(2002, 6, 1): "ame",  # AMSR-E is first AMSR sat
                # F17 starts while AMSR-E is up, on 2008-01-01. We don't use
                # F17 until 2011-10-04.
                dt.date(2008, 1, 1): "F17",
                # dt.date(2012, 7, 3): "am2",  # AMSR2
            }
        )

    _platform_start_dates = cast(
        OrderedDict[dt.date, SUPPORTED_SAT], _platform_start_dates
    )

    assert _platform_start_dates_are_consistent(
        platform_start_dates=_platform_start_dates
    )

    return _platform_start_dates


def _platform_available_for_date(
    *,
    date: dt.date,
    platform: SUPPORTED_SAT,
    platform_availability: OrderedDict = PLATFORM_AVAILABILITY,
) -> bool:
    """Determine if platform is available on this date."""
    # First, verify the values of the first listed platform
    first_available_date = platform_availability[platform]["first_date"]
    if date < first_available_date:
        print(
            f"""
            Satellite {platform} is not available on date {date}
            {date} is before first_available_date {first_available_date}
            Date info: {platform_availability[platform]}
            """
        )
        return False

    try:
        last_available_date = platform_availability[platform]["last_date"]
        try:
            if date > last_available_date:
                print(
                    f"""
                    Satellite {platform} is not available on date {date}
                    {date} is after last_available_date {last_available_date}
                    Date info: {platform_availability[platform]}
                    """
                )
                return False
        except TypeError as e:
            if last_available_date is None:
                pass
            else:
                raise e
    except IndexError as e:
        # last_date is set to None if platform is still providing new data
        if last_available_date is None:
            pass
        else:
            raise e

    return True


def _platform_start_dates_are_consistent(
    *,
    platform_start_dates: OrderedDict[dt.date, SUPPORTED_SAT],
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
            platform_availability=PLATFORM_AVAILABILITY,
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
                platform_availability=PLATFORM_AVAILABILITY,
            )

            # Check this platform's first available date
            assert _platform_available_for_date(
                date=date,
                platform=platform,
                platform_availability=PLATFORM_AVAILABILITY,
            )
    except AssertionError:
        raise RuntimeError(
            f"""
        platform start dates are not consistent
        platform_start_dates: {platform_start_dates}
        platform_availability: {PLATFORM_AVAILABILITY}
        """
        )

    return True


def get_platform_by_date(
    date: dt.date,
) -> SUPPORTED_SAT:
    """Return the platform for this date."""
    platform_start_dates = get_platform_start_dates()

    start_date_list = list(platform_start_dates.keys())
    platform_list = list(platform_start_dates.values())

    if date < start_date_list[0]:
        raise RuntimeError(
            f"""
           date {date} too early.
           First start_date: {start_date_list[0]}
           """
        )

    return_platform = None
    if date >= start_date_list[-1]:
        return_platform = platform_list[-1]

    if return_platform is None:
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
    platform_start_dates = get_platform_start_dates()
    earliest_date = min(platform_start_dates.keys())

    return earliest_date
