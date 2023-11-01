"""Code for producing the monthly ECDR.

Follows the same procedure as CDR v4.

Variables:

* `nsidc_nt_seaice_conc_monthly`: Average the daily NASA Team sea ice concentration
  values over each month of data
* `nsidc_bt_seaice_conc_monthly: Average the daily Bootstrap sea ice concentration
  values over each month of data
* `cdr_seaice_conc_monthly`: Create combined monthly sea ice concentration
* `stdev_of_cdr_seaice_conc_monthly`: Calculate standard deviation of sea ice
  concentration
* `qa_of_cdr_seaice_conc_monthly`: QA Fields (Weather filters, land
  spillover, valid ice mask, spatial and temporal interpolation, melt onset)
* `melt_onset_day_cdr_seaice_conc_monthly`: Melt onset day (Value from the last
  day of the month)

Notes about CDR v4:

* Requires a minimum of 20 days for monthly calculations, unless the
  platform is 'n07', in which case we use a minimum of 10 days.
* Skips 1987-12 and 1988-01.
* Only determines monthly melt onset day for NH.
* CDR is not re-calculated from the monthly nt and bt fields. Just the average
  of the CDR conc fields.
"""

from pathlib import Path

import xarray as xr

from seaice_ecdr._types import SUPPORTED_SAT
from seaice_ecdr.complete_daily_ecdr import get_ecdr_dir
from seaice_ecdr.constants import STANDARD_BASE_OUTPUT_DIR


def check_min_days_for_valid_month(
    *,
    daily_ds_for_month: xr.Dataset,
    sat: SUPPORTED_SAT,
) -> None:
    days_in_ds = len(daily_ds_for_month.time)
    if sat == "n07":
        min_days = 10
    else:
        min_days = 20

    has_min_days = days_in_ds >= min_days

    if not has_min_days:
        raise RuntimeError(
            "Failed to make monthly dataset: not enough days in daily dataset."
            f" Has {days_in_ds} but expected at least {min_days}."
        )


def _get_daily_complete_filepaths_for_month(
    *,
    year: int,
    month: int,
    ecdr_data_dir: Path,
    sat: SUPPORTED_SAT,
) -> list[Path]:
    data_dir = get_ecdr_dir(ecdr_data_dir=ecdr_data_dir)
    # TODO: use `get_ecdr_filepath` and iterate over dates in year, month? This
    # would allow us to log when a date is missing, for example.
    data_list = list(data_dir.glob(f"*_{year}{month:02}*_{sat}_*.nc"))

    return data_list


def get_daily_ds_for_month(
    *,
    year: int,
    month: int,
    sat: SUPPORTED_SAT,
    ecdr_data_dir: Path,
) -> xr.Dataset:
    data_list = _get_daily_complete_filepaths_for_month(
        year=year,
        month=month,
        sat=sat,
        ecdr_data_dir=ecdr_data_dir,
    )
    # Read all of the complete daily data for the given year and month.
    ds = xr.open_mfdataset(data_list)

    return ds


def make_monthly_ds(
    *,
    daily_ds_for_month: xr.Dataset,
    sat: SUPPORTED_SAT,
) -> xr.Dataset:
    # TODO: some kind of check that `daily_ds_for_month` only has data for one year & month?
    # Min-day check
    check_min_days_for_valid_month(
        daily_ds_for_month=daily_ds_for_month,
        sat=sat,
    )

    return ds


if __name__ == "__main__":
    from typing import Final

    year = 2022
    month = 3
    sat: Final = "am2"

    daily_ds_for_month = get_daily_ds_for_month(
        year=year,
        month=month,
        ecdr_data_dir=STANDARD_BASE_OUTPUT_DIR,
        sat=sat,
    )
    ds = make_monthly_ds(
        daily_ds_for_month=daily_ds_for_month,
        sat=sat,
    )
