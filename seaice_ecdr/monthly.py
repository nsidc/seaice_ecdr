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
* Uses masks from the first file in the month to apply to monthly fields.
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


QA_OF_CDR_SEAICE_CONC_DAILY_FLAGS = dict(
    bt_weather_filter_applied=1,
    nt_weather_filter_applied=2,
    land_spillover_applied=4,
    no_input_data=8,
    valid_ice_mask_applied=16,
    spatial_interpolation_applied=32,
    temporal_interpolation_applied=64,
    # This flag value only occurs in the Arctic.
    start_of_melt_detected=128,
)

# TODO: rename. This is actually a bit mask (except the fill_value)
QA_OF_CDR_SEAICE_CONC_MONTHLY_FLAGS = dict(
    average_concentration_exeeds_15=1,
    average_concentration_exeeds_30=2,
    at_least_half_the_days_have_sea_ice_conc_exceeds_15=4,
    at_least_half_the_days_have_sea_ice_conc_exceeds_30=8,
    region_masked_by_ocean_climatology=16,
    at_least_one_day_during_month_has_spatial_interpolation=32,
    at_least_one_day_during_month_has_temporal_interpolation=64,
    at_least_one_day_during_month_has_melt_detected=128,
    fill_value=255,
)


def _monthly_qa_field(
    *,
    daily_ds_for_month: xr.Dataset,
    is_sic: xr.DataArray,
    cdr_seaice_conc_monthly: xr.DataArray,
) -> xr.DataArray:
    """Create `qa_of_cdr_seaice_conc_monthly`."""
    # initialize the variable
    qa_of_cdr_seaice_conc_monthly = xr.full_like(
        cdr_seaice_conc_monthly,
        fill_value=0,
    )
    qa_of_cdr_seaice_conc_monthly.name = "qa_of_cdr_seaice_conc_monthly"

    average_exceeds_15 = is_sic & (cdr_seaice_conc_monthly > 15)
    qa_of_cdr_seaice_conc_monthly = qa_of_cdr_seaice_conc_monthly.where(
        ~average_exceeds_15,
        qa_of_cdr_seaice_conc_monthly
        + QA_OF_CDR_SEAICE_CONC_MONTHLY_FLAGS["average_concentration_exeeds_15"],
    )

    average_exceeds_30 = is_sic & (cdr_seaice_conc_monthly > 30)
    qa_of_cdr_seaice_conc_monthly = qa_of_cdr_seaice_conc_monthly.where(
        ~average_exceeds_30,
        qa_of_cdr_seaice_conc_monthly
        + QA_OF_CDR_SEAICE_CONC_MONTHLY_FLAGS["average_concentration_exeeds_30"],
    )

    days_in_ds = len(daily_ds_for_month.time)
    majority_of_days = (days_in_ds + 1) // 2

    at_least_half_have_sic_gt_15 = (
        is_sic & (daily_ds_for_month.cdr_seaice_conc > 15)
    ).sum(dim="time") >= majority_of_days
    qa_of_cdr_seaice_conc_monthly = qa_of_cdr_seaice_conc_monthly.where(
        ~at_least_half_have_sic_gt_15,
        qa_of_cdr_seaice_conc_monthly
        + QA_OF_CDR_SEAICE_CONC_MONTHLY_FLAGS[
            "at_least_half_the_days_have_sea_ice_conc_exceeds_15"
        ],
    )

    at_least_half_have_sic_gt_30 = (
        is_sic & (daily_ds_for_month.cdr_seaice_conc > 30)
    ).sum(dim="time") >= majority_of_days
    qa_of_cdr_seaice_conc_monthly = qa_of_cdr_seaice_conc_monthly.where(
        ~at_least_half_have_sic_gt_30,
        qa_of_cdr_seaice_conc_monthly
        + QA_OF_CDR_SEAICE_CONC_MONTHLY_FLAGS[
            "at_least_half_the_days_have_sea_ice_conc_exceeds_30"
        ],
    )

    # Use "valid_ice_mask_applied", which is actually the invalid ice mask.
    region_masked_by_ocean_climatology = (
        daily_ds_for_month.qa_of_cdr_seaice_conc
        == QA_OF_CDR_SEAICE_CONC_DAILY_FLAGS["valid_ice_mask_applied"]
    ).any(dim="time")
    qa_of_cdr_seaice_conc_monthly = qa_of_cdr_seaice_conc_monthly.where(
        ~region_masked_by_ocean_climatology,
        qa_of_cdr_seaice_conc_monthly
        + QA_OF_CDR_SEAICE_CONC_MONTHLY_FLAGS["region_masked_by_ocean_climatology"],
    )

    at_least_one_day_during_month_has_spatial_interpolation = (
        daily_ds_for_month.qa_of_cdr_seaice_conc
        == QA_OF_CDR_SEAICE_CONC_DAILY_FLAGS["spatial_interpolation_applied"]
    ).any(dim="time")
    qa_of_cdr_seaice_conc_monthly = qa_of_cdr_seaice_conc_monthly.where(
        ~at_least_one_day_during_month_has_spatial_interpolation,
        qa_of_cdr_seaice_conc_monthly
        + QA_OF_CDR_SEAICE_CONC_MONTHLY_FLAGS[
            "at_least_one_day_during_month_has_spatial_interpolation"
        ],
    )

    at_least_one_day_during_month_has_temporal_interpolation = (
        daily_ds_for_month.qa_of_cdr_seaice_conc
        == QA_OF_CDR_SEAICE_CONC_DAILY_FLAGS["temporal_interpolation_applied"]
    ).any(dim="time")
    qa_of_cdr_seaice_conc_monthly = qa_of_cdr_seaice_conc_monthly.where(
        ~at_least_one_day_during_month_has_temporal_interpolation,
        qa_of_cdr_seaice_conc_monthly
        + QA_OF_CDR_SEAICE_CONC_MONTHLY_FLAGS[
            "at_least_one_day_during_month_has_temporal_interpolation"
        ],
    )

    at_least_one_day_during_month_has_melt_detected = (
        daily_ds_for_month.qa_of_cdr_seaice_conc
        == QA_OF_CDR_SEAICE_CONC_DAILY_FLAGS["start_of_melt_detected"]
    ).any(dim="time")
    qa_of_cdr_seaice_conc_monthly = qa_of_cdr_seaice_conc_monthly.where(
        ~at_least_one_day_during_month_has_melt_detected,
        qa_of_cdr_seaice_conc_monthly
        + QA_OF_CDR_SEAICE_CONC_MONTHLY_FLAGS[
            "at_least_one_day_during_month_has_melt_detected"
        ],
    )

    qa_of_cdr_seaice_conc_monthly = qa_of_cdr_seaice_conc_monthly.where(
        qa_of_cdr_seaice_conc_monthly != 0,
        QA_OF_CDR_SEAICE_CONC_MONTHLY_FLAGS["fill_value"],
    )

    return qa_of_cdr_seaice_conc_monthly


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

    # create `nsidc_{nt|bt}_seaice_conc_monthly`. These are averages of the
    # daily NT and BT values. These 'raw' fields do not have any flags.
    nsidc_nt_seaice_conc_monthly = daily_ds_for_month.nasateam_seaice_conc_raw.mean(
        dim="time"
    )
    nsidc_nt_seaice_conc_monthly.name = "nsidc_nt_seaice_conc_monthly"
    nsidc_bt_seaice_conc_monthly = daily_ds_for_month.bootstrap_seaice_conc_raw.mean(
        dim="time"
    )
    nsidc_bt_seaice_conc_monthly.name = "nsidc_bt_seaice_conc_monthly"

    # create `cdr_seaice_conc_monthly`. This is the combined monthly SIC.
    # The `cdr_seaice_conc` variable has temporally filled data and flags.
    # TODO: what's `conc`?
    # TODO: should we handle flags separately? Land mask should be consistent
    # between files (so the average will still be the land flag), and this code
    # assumes `NaN` is missing data. CDR v4 copies the flags from the first
    # daily data file it gets and applies it to the average for the month after
    # the mean calculation.
    cdr_seaice_conc_monthly = daily_ds_for_month.cdr_seaice_conc.sel(
        time=daily_ds_for_month.time.min()
    ).copy()
    cdr_seaice_conc_monthly = cdr_seaice_conc_monthly.drop_vars("time")
    cdr_seaice_conc_monthly.name = "cdr_seaice_conc_monthly"
    is_sic = (cdr_seaice_conc_monthly >= 0) & (cdr_seaice_conc_monthly <= 100)
    cdr_seaice_conc_monthly = cdr_seaice_conc_monthly.where(
        # Locations at which to preserve this object’s values.
        ~is_sic,
        # TODO: use `drop_attrs=True`?
        daily_ds_for_month.cdr_seaice_conc.mean(dim="time"),
    )

    # Create `stdev_of_cdr_seaice_conc_monthly`, the standard deviation of the
    # sea ice concentration.
    stdv_of_cdr_seaice_conc_monthly = daily_ds_for_month.cdr_seaice_conc.sel(
        time=daily_ds_for_month.time.min()
    ).copy()
    stdv_of_cdr_seaice_conc_monthly = stdv_of_cdr_seaice_conc_monthly.drop_vars("time")
    stdv_of_cdr_seaice_conc_monthly.name = "stdv_of_cdr_seaice_conc_monthly"
    stdv_of_cdr_seaice_conc_monthly = stdv_of_cdr_seaice_conc_monthly.where(
        # Locations at which to preserve this object’s values.
        ~is_sic,
        # cdr v4 uses a ddof of 1.
        daily_ds_for_month.cdr_seaice_conc.std(dim="time", ddof=1),
    )

    qa_of_cdr_seaice_conc_monthly = _monthly_qa_field(
        daily_ds_for_month=daily_ds_for_month,
        is_sic=is_sic,
        cdr_seaice_conc_monthly=cdr_seaice_conc_monthly,
    )

    # Create `melt_onset_day_cdr_seaice_conc_monthly`. This is the value from
    # the last day of the month.
    melt_onset_day_cdr_seaice_conc_monthly = (
        daily_ds_for_month.melt_onset_day_cdr_seaice_conc.sel(
            time=daily_ds_for_month.time.max()
        )
    )
    melt_onset_day_cdr_seaice_conc_monthly.name = (
        "melt_onset_day_cdr_seaice_conc_monthly"
    )
    melt_onset_day_cdr_seaice_conc_monthly = (
        melt_onset_day_cdr_seaice_conc_monthly.drop_vars("time")
    )

    monthly_ds = xr.Dataset(
        data_vars=dict(
            cdr_seaice_conc_monthly=cdr_seaice_conc_monthly,
            nsidc_nt_seaice_conc_monthly=nsidc_nt_seaice_conc_monthly,
            nsidc_bt_seaice_conc_monthly=nsidc_bt_seaice_conc_monthly,
            stdv_of_cdr_seaice_conc_monthly=stdv_of_cdr_seaice_conc_monthly,
            melt_onset_day_cdr_seaice_conc_monthly=melt_onset_day_cdr_seaice_conc_monthly,
            qa_of_cdr_seaice_conc_monthly=qa_of_cdr_seaice_conc_monthly,
        )
    )

    return monthly_ds


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
    monthly_ds = make_monthly_ds(
        daily_ds_for_month=daily_ds_for_month,
        sat=sat,
    )

    breakpoint()