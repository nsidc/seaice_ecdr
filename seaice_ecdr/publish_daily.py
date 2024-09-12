import datetime as dt
from pathlib import Path

from pm_tb_data._types import Hemisphere

from seaice_ecdr._types import ECDR_SUPPORTED_RESOLUTIONS
from seaice_ecdr.platforms import PLATFORM_CONFIG
from seaice_ecdr.util import (
    nrt_daily_filename,
    standard_daily_filename,
)


# TODO: this and `get_complete_daily_filepath` are identical (aside from var
# names) to `get_ecdr_filepath` and `get_ecdr_dir`.
def get_complete_daily_dir(
    *,
    complete_output_dir: Path,
    year: int,
    # TODO: extract nrt handling and make responsiblity for defining the output
    # dir a higher-level concern.
    is_nrt: bool,
) -> Path:
    if is_nrt:
        # NRT daily data just lives under the complete output dir.
        ecdr_dir = complete_output_dir
    else:
        ecdr_dir = complete_output_dir / "daily" / str(year)
    ecdr_dir.mkdir(parents=True, exist_ok=True)

    return ecdr_dir


def get_complete_daily_filepath(
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    complete_output_dir: Path,
    is_nrt: bool,
):
    platform = PLATFORM_CONFIG.get_platform_by_date(date)
    if is_nrt:
        ecdr_filename = nrt_daily_filename(
            hemisphere=hemisphere,
            date=date,
            platform_id=platform.id,
            resolution=resolution,
        )
    else:
        ecdr_filename = standard_daily_filename(
            hemisphere=hemisphere,
            date=date,
            platform_id=platform.id,
            resolution=resolution,
        )

    ecdr_dir = get_complete_daily_dir(
        complete_output_dir=complete_output_dir,
        year=date.year,
        is_nrt=is_nrt,
    )

    ecdr_filepath = ecdr_dir / ecdr_filename

    return ecdr_filepath
