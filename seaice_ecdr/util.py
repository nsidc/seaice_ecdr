import datetime as dt

from pm_tb_data._types import Hemisphere

from seaice_ecdr.constants import ECDR_PRODUCT_VERSION


def standard_daily_filename(
    *,
    hemisphere: Hemisphere,
    resolution: str,
    sat: str,
    date: dt.date,
) -> str:
    """Return standard daily NetCDF filename.

    North Daily files: sic_psn12.5_YYYYMMDD_sat_v05r00.nc
    South Daily files: sic_pss12.5_YYYYMMDD_sat_v05r00.nc
    """
    fn = f"sic_ps{hemisphere[0]}{resolution}_{date:%Y%m%d}_{sat}_{ECDR_PRODUCT_VERSION}.nc"

    return fn


def standard_monthly_filename(
    *,
    hemisphere: Hemisphere,
    resolution: str,
    sat: str,
    year: int,
    month: int,
) -> str:
    """Return standard monthly NetCDF filename.

    North Monthly files: sic_psn12.5_YYYYMM_sat_v05r00.nc
    South Monthly files: sic_pss12.5_YYYYMM_sat_v05r00.nc
    """
    fn = f"sic_ps{hemisphere[0]}{resolution}_{year}{month:02}_{sat}_{ECDR_PRODUCT_VERSION}.nc"

    return fn
