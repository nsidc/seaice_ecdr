"""platforms.py.

Routines for dealing with the platform (satellite) and sensors
for CDRv5
"""

import datetime as dt
from collections import OrderedDict

# from functools import cache
# from typing import Any, Final, Literal, get_args
from seaice_ecdr._types import SUPPORTED_SAT

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
    ame={"first_date": dt.date(2002, 6, 1), "last_date": dt.date(2011, 10, 3)},
    am2={"first_date": dt.date(2012, 7, 2), "last_date": None},
)


PLATFORM_START_DATES: OrderedDict[dt.date, str] = OrderedDict(
    {
        dt.date(1978, 10, 25): "n07",
        dt.date(1987, 7, 10): "f08",
        dt.date(1991, 12, 3): "f11",
        dt.date(1995, 10, 1): "f13",
        dt.date(2008, 1, 1): "f17",
        # dt.date(2002, 6, 1): 'ame',   # AMSR-E...
        # dt.date(2011, 10, 4): 'f17',  # followed by f17 (again)
        dt.date(2012, 7, 3): "am2",
    }
)


def _platform_available_for_date(
    *,
    date: dt.date,
    platform: SUPPORTED_SAT,
    platform_availability: OrderedDict = PLATFORM_AVAILABILITY,
) -> bool:
    """Determine if platform is available on this date."""
    first_available_date = platform_availability["platform"]["first_date"]
    if date < first_available_date:
        raise RuntimeError(
            f"""
        Satellite {platform} is not available on date {date}
        {date} is before first_available_date {first_available_date}
        Date info: {platform_availability["platform"]}
        """
        )

    try:
        last_available_date = platform_availability["platform"]["last_date"]
        if date > last_available_date:
            raise RuntimeError(
                f"""
            Satellite {platform} is not available on date {date}
            {date} is after last_available_date {last_available_date}
            Date info: {platform_availability["platform"]}
            """
            )
    except IndexError as e:
        # last_date is set to None if platform is still providing new data
        if last_available_date is None:
            pass
        else:
            raise e

    return True


# @cache
def _platform_start_dates_are_consistent(
    *,
    platform_start_dates: OrderedDict = PLATFORM_START_DATES,
    platform_availability: OrderedDict = PLATFORM_AVAILABILITY,
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
            platform_availability=platform_availability,
        )

        for idx in range(1, len(date_list) + 1):
            date = date_list[idx]
            platform = platform_list[idx]

            # Check the end of the prior platform's date range
            prior_date = date - dt.timedelta(days=1)
            prior_platform = platform[idx - 1]
            assert _platform_available_for_date(
                date=prior_date,
                platform=prior_platform,
                platform_availability=platform_availability,
            )

            # Check this platform's first available date
            assert _platform_available_for_date(
                date=date,
                platform=platform,
                platform_availability=platform_availability,
            )
    except AssertionError:
        raise RuntimeError(
            f"""
        platform start dates are not consistent
        platform_start_dates: {platform_start_dates}
        platform_availability: {platform_availability}
        """
        )

    return True


def get_platform_by_date(
    date: dt.date,
    platform_start_dates: OrderedDict = PLATFORM_START_DATES,
    platform_availability: OrderedDict = PLATFORM_AVAILABILITY,
) -> str:
    """Return the platform for this date."""
    assert _platform_start_dates_are_consistent(
        platform_start_dates=platform_start_dates,
        platform_availability=platform_availability,
    )

    for start_date in platform_start_dates.keys():
        if date > start_date:
            continue
        platform = platform_start_dates[start_date]
        assert _platform_available_for_date(
            date=date,
            platform=platform,
            platform_availability=platform_availability,
        )
        break

    return platform
