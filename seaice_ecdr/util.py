import datetime as dt
import re
from typing import cast, get_args

from pm_tb_data._types import Hemisphere

from seaice_ecdr._types import ECDR_SUPPORTED_RESOLUTIONS, SUPPORTED_SAT
from seaice_ecdr.constants import ECDR_PRODUCT_VERSION


def standard_daily_filename(
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    sat: SUPPORTED_SAT,
    date: dt.date,
) -> str:
    """Return standard daily NetCDF filename.

    North Daily files: sic_psn12.5_YYYYMMDD_sat_v05r00.nc
    South Daily files: sic_pss12.5_YYYYMMDD_sat_v05r00.nc
    """
    fn = f"sic_ps{hemisphere[0]}{resolution}_{date:%Y%m%d}_{sat}_{ECDR_PRODUCT_VERSION}.nc"

    return fn


def standard_daily_aggregate_filename(
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    start_date: dt.date,
    end_date: dt.date,
) -> str:
    """Return standard daily aggregate NetCDF filename.

    North Daily aggregate files: sic_psn12.5_YYYYMMDD-YYYYMMDD_v05r00.nc
    South Daily aggregate files: sic_pss12.5_YYYYMMDD-YYYYMMDD_v05r00.nc
    """
    fn = f"sic_ps{hemisphere[0]}{resolution}_{start_date:%Y%m%d}-{end_date:%Y%m%d}_{ECDR_PRODUCT_VERSION}.nc"

    return fn


def standard_monthly_filename(
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    sat: SUPPORTED_SAT,
    year: int,
    month: int,
) -> str:
    """Return standard monthly NetCDF filename.

    North Monthly files: sic_psn12.5_YYYYMM_sat_v05r00.nc
    South Monthly files: sic_pss12.5_YYYYMM_sat_v05r00.nc
    """
    fn = f"sic_ps{hemisphere[0]}{resolution}_{year}{month:02}_{sat}_{ECDR_PRODUCT_VERSION}.nc"

    return fn


def standard_monthly_aggregate_filename(
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    start_year: int,
    start_month: int,
    end_year: int | None = None,
    end_month: int | None = None,
) -> str:
    """Return standard monthly aggregate NetCDF filename.

    North Monthly aggregate files: sic_psn12.5_YYYYMM-YYYYMM_v05r00.nc
    South Monthly aggregate files: sic_pss12.5_YYYYMM-YYYYMM_v05r00.nc
    """
    date_str = f"{start_year}{start_month:02}-{end_year}{end_month:02}"

    fn = f"sic_ps{hemisphere[0]}{resolution}_{date_str}_{ECDR_PRODUCT_VERSION}.nc"

    return fn


# This regex works for both daily and monthly filenames.
STANDARD_FN_REGEX = re.compile(r"sic_ps.*_.*_(?P<sat>.*)_.*.nc")


def sat_from_filename(filename: str) -> SUPPORTED_SAT:
    match = STANDARD_FN_REGEX.match(filename)

    if not match:
        raise RuntimeError(f"Failed to parse satellite from {filename}")

    sat = match.group("sat")

    # Ensure the sat is expected.
    assert sat in get_args(SUPPORTED_SAT)
    sat = cast(SUPPORTED_SAT, sat)

    return sat
