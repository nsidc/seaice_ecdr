"""Platform config

Contains configuration for platforms supported by this code (e.g., AMSR2, F17, etc.).

Platform start dates are read from a yaml file in this directory
(`default_platform_start_dates.yml`) unless overridden by the
`PLATFORM_START_DATES_CONFIG_FILEPATH` envvar.
"""

import datetime as dt
import os
from pathlib import Path
from typing import cast, get_args

import yaml
from loguru import logger

from seaice_ecdr.platforms.models import (
    SUPPORTED_PLATFORM_ID,
    DateRange,
    Platform,
    PlatformConfig,
    PlatformStartDate,
    platform_for_id,
)

_this_dir = Path(__file__).parent
DEFAULT_PLATFORM_START_DATES_CONFIG_FILEPATH = Path(
    _this_dir / "../config/default_platform_start_dates.yml"
).resolve()
PROTOTYPE_PLATFORM_START_DATES_CONFIG_FILEPATH = Path(
    _this_dir / "../config/prototype_platform_start_dates.yml"
).resolve()
NRT_PLATFORM_START_DATES_CONFIG_FILEPATH = Path(
    _this_dir / "../config/nrt_platform_start_dates.yml"
).resolve()


AM2_PLATFORM = Platform(
    name="GCOM-W1 > Global Change Observation Mission 1st-Water",
    sensor="AMSR2 > Advanced Microwave Scanning Radiometer 2",
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


F18_PLATFORM = Platform(
    name="DMSP 5D-3/F18 > Defense Meteorological Satellite Program-F18",
    sensor="SSMIS > Special Sensor Microwave Imager/Sounder",
    id="F18",
    date_range=DateRange(
        # TODO: is this accurate? This value from NSIDC-0001 docs.
        first_date=dt.date(2017, 1, 1),
        last_date=None,
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

SUPPORTED_PLATFORMS = [
    AM2_PLATFORM,
    AME_PLATFORM,
    F17_PLATFORM,
    F18_PLATFORM,
    F13_PLATFORM,
    F11_PLATFORM,
    F08_PLATFORM,
    N07_PLATFORM,
]


def _get_platform_config() -> PlatformConfig:
    """Gets the platform config given a start dates filepath.

    This function is not intended to be used outside of this module, as it sets
    a global variable accessed from other parts of the code (`PLATFORM_CONFIG`).
    """

    if platform_override_filepath_str := os.environ.get(
        "PLATFORM_START_DATES_CONFIG_FILEPATH"
    ):
        platform_start_dates_config_filepath = Path(platform_override_filepath_str)
    else:
        platform_start_dates_config_filepath = (
            DEFAULT_PLATFORM_START_DATES_CONFIG_FILEPATH
        )

    logger.info(
        f"Using platform start dates from {platform_start_dates_config_filepath}"
    )

    if not platform_start_dates_config_filepath.is_file():
        raise RuntimeError(
            f"Could not find platform config file: {platform_start_dates_config_filepath}"
        )

    # TODO: drop support for "FORCE_PLATFORM" in favor of a platform start dates
    # config override file.
    if forced_platform_id := os.environ.get("FORCE_PLATFORM"):
        if forced_platform_id not in get_args(SUPPORTED_PLATFORM_ID):
            raise RuntimeError(
                f"The forced platform ({forced_platform_id}) is not a supported platform."
            )

        forced_platform_id = cast(SUPPORTED_PLATFORM_ID, forced_platform_id)
        forced_platform = platform_for_id(
            platforms=SUPPORTED_PLATFORMS, platform_id=forced_platform_id
        )
        first_date_of_forced_platform = forced_platform.date_range.first_date
        forced_cdr_platform_start_dates = [
            PlatformStartDate(
                platform_id=forced_platform_id,
                start_date=first_date_of_forced_platform,
            )
        ]

        return PlatformConfig(
            platforms=SUPPORTED_PLATFORMS,
            cdr_platform_start_dates=forced_cdr_platform_start_dates,
        )

    with open(platform_start_dates_config_filepath, "r") as config_file:
        start_dates_cfg = yaml.safe_load(config_file)
        platform_cfg = PlatformConfig(platforms=SUPPORTED_PLATFORMS, **start_dates_cfg)

    return platform_cfg


PLATFORM_CONFIG = _get_platform_config()
