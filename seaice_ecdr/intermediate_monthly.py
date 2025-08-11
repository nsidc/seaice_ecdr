"""Code for producing the monthly ECDR.

Follows the same procedure as CDR v4.

Variables:

* `cdr_seaice_conc_monthly`: Create combined monthly sea ice concentration
* `cdr_seaice_conc_monthly_stdev`: Calculate standard deviation of sea ice
  concentration
* `cdr_seaice_conc_monthly_qa_flag`: QA Fields (Weather filters, land
  spillover, valid ice mask, spatial and temporal interpolation, melt onset)
* `cdr_melt_onset_day_monthly`: Melt onset day (Value from the last
  day of the month)

Notes about CDR v4:

* Requires a minimum of 20 days for monthly calculations, unless the
  platform is 'n07', in which case we use a minimum of 10 days.
* Skips 1987-12 and 1988-01.
* Only determines monthly melt onset day for NH.
* CDR is not re-calculated from the monthly nt and bt fields. Just the average
  of the CDR conc fields.
"""

import calendar
import datetime as dt
from collections import Counter, OrderedDict
from pathlib import Path
from typing import get_args

import click
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from pm_tb_data._types import NORTH, Hemisphere

from seaice_ecdr._types import ECDR_SUPPORTED_RESOLUTIONS
from seaice_ecdr.ancillary import (
    ANCILLARY_SOURCES,
    flag_value_for_meaning,
    get_monthly_cdr_conc_threshold,
    remove_FillValue_from_coordinate_vars,
)
from seaice_ecdr.constants import DEFAULT_BASE_OUTPUT_DIR
from seaice_ecdr.days_treated_differently import months_of_cdr_missing_data
from seaice_ecdr.intermediate_daily import get_ecdr_filepath
from seaice_ecdr.melt import MELT_SEASON_LAST_DOY
from seaice_ecdr.nc_attrs import get_global_attrs
from seaice_ecdr.platforms import PLATFORM_CONFIG, SUPPORTED_PLATFORM_ID
from seaice_ecdr.tb_data import get_hemisphere_from_crs_da
from seaice_ecdr.util import (
    get_intermediate_output_dir,
    platform_id_from_filename,
    standard_monthly_filename,
)


def check_min_days_for_valid_month(
    *,
    daily_ds_for_month: xr.Dataset,
    platform_id: SUPPORTED_PLATFORM_ID,
) -> None:
    days_in_ds = len(daily_ds_for_month.time)
    if platform_id == "n07":
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
    intermediate_output_dir: Path,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    is_nrt: bool,
) -> list[Path]:
    """Return a list of paths to ECDR daily complete filepaths for the given year and month."""
    data_list = []
    _, last_day_of_month = calendar.monthrange(year, month)
    for period in pd.period_range(
        start=dt.date(year, month, 1),
        end=dt.date(year, month, last_day_of_month),
        freq="D",
    ):

        platform = PLATFORM_CONFIG.get_platform_by_date(period.to_timestamp().date())
        expected_fp = get_ecdr_filepath(
            date=period.to_timestamp().date(),
            hemisphere=hemisphere,
            resolution=resolution,
            intermediate_output_dir=intermediate_output_dir,
            platform_id=platform.id,
            is_nrt=is_nrt,
        )
        if expected_fp.is_file():
            data_list.append(expected_fp)
        else:
            logger.warning(f"Expected to find {expected_fp} but found none.")

    if len(data_list) == 0:
        raise RuntimeError(
            f"No daily data files found looking in {intermediate_output_dir=}"
        )

    return data_list


def _platform_id_for_month(
    *, platform_ids: list[SUPPORTED_PLATFORM_ID]
) -> SUPPORTED_PLATFORM_ID:
    """Returns the platform ID from this month given a list of input platforms.

    The platform for monthly files is based on which platform contributes most to the
    month. If two platforms contribute equally, use the latest platform in the series.

    Function assumes the list of platform ids is already sorted (i.e., the latest
    platform is `platform_ids[-1]`).
    """
    # More than one platform, we need to choose the most common/latest in the series.
    # `Counter` returns a dict keyed by `platform` with counts as values:
    count = Counter(platform_ids)
    most_common_platform_ids = count.most_common()
    most_common_and_latest_platform_id = most_common_platform_ids[-1][0]

    return most_common_and_latest_platform_id


def get_daily_ds_for_month(
    *,
    year: int,
    month: int,
    intermediate_output_dir: Path,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    is_nrt: bool,
) -> xr.Dataset:
    """Create an xr.Dataset wtih ECDR complete daily data for a given year and month.

    The resulting xr.Dataset includes:
        * `year` and `month` attribtues.
        * The filepaths of the source data are included in a `filepaths` variable.
    """
    data_list = _get_daily_complete_filepaths_for_month(
        year=year,
        month=month,
        intermediate_output_dir=intermediate_output_dir,
        hemisphere=hemisphere,
        resolution=resolution,
        is_nrt=is_nrt,
    )
    # Read all of the complete daily data for the given year and month.
    ds = xr.open_mfdataset(data_list)

    # Assert that we have the year and month that we want.
    assert np.all([pd.Timestamp(t.values).year == year for t in ds.time])
    assert np.all([pd.Timestamp(t.values).month == month for t in ds.time])

    ds.attrs["year"] = year
    ds.attrs["month"] = month

    ds["filepaths"] = xr.DataArray(
        data=data_list, dims=("time",), coords=dict(time=ds.time)
    )

    # Extract `platform_id` from the filenames contributing to this
    # dataset. Ideally, we would use a custom `combine_attrs` when reading the
    # data with `xr.open_mfdataset` in order to get the platform/sensor from global
    # attrs in each of the contributing files. Unfortunately this interface is
    # poorly documented and seems to have limited support. E.g., see
    # https://github.com/pydata/xarray/issues/6679
    platform_ids = []
    for filepath in data_list:
        platform_ids.append(platform_id_from_filename(filepath.name))

    platform_id = _platform_id_for_month(platform_ids=platform_ids)

    ds.attrs["platform_id"] = platform_id

    return ds


# TODO: utilize these in the daily complete processing. Or use constant from
# that module!
CDR_SEAICE_CONC_QA_FLAG_DAILY_BITMASKS = OrderedDict(
    bt_weather_filter_applied=1,
    nt_weather_filter_applied=2,
    land_spillover_applied=4,
    no_input_data=8,
    invalid_ice_mask_applied=16,
    spatial_interpolation_applied=32,
    temporal_interpolation_applied=64,
    # This flag value only occurs in the Arctic.
    start_of_melt_detected=128,
)

# TODO: rename. This is actually a bit mask (except the fill_value)
CDR_SEAICE_CONC_MONTHLY_QA_FLAG_BITMASKS_NORTH = OrderedDict(
    {
        "average_concentration_exceeds_0.15": 1,
        "average_concentration_exceeds_0.30": 2,
        "at_least_half_the_days_have_sea_ice_conc_exceeds_0.15": 4,
        "at_least_half_the_days_have_sea_ice_conc_exceeds_0.30": 8,
        "invalid_ice_mask_applied": 16,
        "at_least_one_day_during_month_has_spatial_interpolation": 32,
        "at_least_one_day_during_month_has_temporal_interpolation": 64,
        "at_least_one_day_during_month_has_melt_detected": 128,
    }
)

CDR_SEAICE_CONC_MONTHLY_QA_FLAG_BITMASKS_SOUTH = (
    CDR_SEAICE_CONC_MONTHLY_QA_FLAG_BITMASKS_NORTH.copy()
)
CDR_SEAICE_CONC_MONTHLY_QA_FLAG_BITMASKS_SOUTH.pop(
    "at_least_one_day_during_month_has_melt_detected"
)


def _qa_field_has_flag(*, qa_field: xr.DataArray, flag_value: int) -> xr.DataArray:
    """Returns a boolean DataArray indicating where a flag value occurs in the given `qa_field`."""
    qa_field_contains_flag = (
        qa_field.where(qa_field.isnull(), qa_field.astype(int) & flag_value)
        == flag_value
    )

    return qa_field_contains_flag


def calc_cdr_seaice_conc_monthly_qa_flag(
    *,
    daily_ds_for_month: xr.Dataset,
    cdr_seaice_conc_monthly: xr.DataArray,
    hemisphere: Hemisphere,
) -> xr.DataArray:
    """Create `cdr_seaice_conc_monthly_qa_flag`."""
    if hemisphere == "north":
        qa_flag_bitmasks = CDR_SEAICE_CONC_MONTHLY_QA_FLAG_BITMASKS_NORTH
    else:
        qa_flag_bitmasks = CDR_SEAICE_CONC_MONTHLY_QA_FLAG_BITMASKS_SOUTH

    # initialize the variable
    cdr_seaice_conc_monthly_qa_flag = xr.full_like(
        cdr_seaice_conc_monthly,
        fill_value=0,
        dtype=np.uint8,
    )
    cdr_seaice_conc_monthly_qa_flag.name = "cdr_seaice_conc_monthly_qa_flag"

    average_exceeds_15 = cdr_seaice_conc_monthly > 0.15
    cdr_seaice_conc_monthly_qa_flag = cdr_seaice_conc_monthly_qa_flag.where(
        ~average_exceeds_15,
        cdr_seaice_conc_monthly_qa_flag
        + qa_flag_bitmasks["average_concentration_exceeds_0.15"],
    )

    average_exceeds_30 = cdr_seaice_conc_monthly > 0.30
    cdr_seaice_conc_monthly_qa_flag = cdr_seaice_conc_monthly_qa_flag.where(
        ~average_exceeds_30,
        cdr_seaice_conc_monthly_qa_flag
        + qa_flag_bitmasks["average_concentration_exceeds_0.30"],
    )

    days_in_ds = len(daily_ds_for_month.time)
    majority_of_days = (days_in_ds + 1) // 2

    at_least_half_have_sic_gt_15 = (daily_ds_for_month.cdr_seaice_conc > 0.15).sum(
        dim="time"
    ) >= majority_of_days
    cdr_seaice_conc_monthly_qa_flag = cdr_seaice_conc_monthly_qa_flag.where(
        ~at_least_half_have_sic_gt_15,
        cdr_seaice_conc_monthly_qa_flag
        + qa_flag_bitmasks["at_least_half_the_days_have_sea_ice_conc_exceeds_0.15"],
    )

    at_least_half_have_sic_gt_30 = (daily_ds_for_month.cdr_seaice_conc > 0.30).sum(
        dim="time"
    ) >= majority_of_days
    cdr_seaice_conc_monthly_qa_flag = cdr_seaice_conc_monthly_qa_flag.where(
        ~at_least_half_have_sic_gt_30,
        cdr_seaice_conc_monthly_qa_flag
        + qa_flag_bitmasks["at_least_half_the_days_have_sea_ice_conc_exceeds_0.30"],
    )

    # Use "invalid_ice_mask_applied", which is actually the invalid ice mask.
    invalid_ice_mask_applied = _qa_field_has_flag(
        qa_field=daily_ds_for_month.cdr_seaice_conc_qa_flag,
        flag_value=CDR_SEAICE_CONC_QA_FLAG_DAILY_BITMASKS["invalid_ice_mask_applied"],
    ).any(dim="time")
    cdr_seaice_conc_monthly_qa_flag = cdr_seaice_conc_monthly_qa_flag.where(
        ~invalid_ice_mask_applied,
        cdr_seaice_conc_monthly_qa_flag + qa_flag_bitmasks["invalid_ice_mask_applied"],
    )

    at_least_one_day_during_month_has_spatial_interpolation = _qa_field_has_flag(
        qa_field=daily_ds_for_month.cdr_seaice_conc_qa_flag,
        flag_value=CDR_SEAICE_CONC_QA_FLAG_DAILY_BITMASKS[
            "spatial_interpolation_applied"
        ],
    ).any(dim="time")
    cdr_seaice_conc_monthly_qa_flag = cdr_seaice_conc_monthly_qa_flag.where(
        ~at_least_one_day_during_month_has_spatial_interpolation,
        cdr_seaice_conc_monthly_qa_flag
        + qa_flag_bitmasks["at_least_one_day_during_month_has_spatial_interpolation"],
    )

    at_least_one_day_during_month_has_temporal_interpolation = _qa_field_has_flag(
        qa_field=daily_ds_for_month.cdr_seaice_conc_qa_flag,
        flag_value=CDR_SEAICE_CONC_QA_FLAG_DAILY_BITMASKS[
            "temporal_interpolation_applied"
        ],
    ).any(dim="time")
    cdr_seaice_conc_monthly_qa_flag = cdr_seaice_conc_monthly_qa_flag.where(
        ~at_least_one_day_during_month_has_temporal_interpolation,
        cdr_seaice_conc_monthly_qa_flag
        + qa_flag_bitmasks["at_least_one_day_during_month_has_temporal_interpolation"],
    )

    if hemisphere == "north":
        at_least_one_day_during_month_has_melt_detected = _qa_field_has_flag(
            qa_field=daily_ds_for_month.cdr_seaice_conc_qa_flag,
            flag_value=CDR_SEAICE_CONC_QA_FLAG_DAILY_BITMASKS["start_of_melt_detected"],
        ).any(dim="time")
        cdr_seaice_conc_monthly_qa_flag = cdr_seaice_conc_monthly_qa_flag.where(
            ~at_least_one_day_during_month_has_melt_detected,
            cdr_seaice_conc_monthly_qa_flag
            + qa_flag_bitmasks["at_least_one_day_during_month_has_melt_detected"],
        )

    monthly_qa_values = [np.uint8(value) for value in qa_flag_bitmasks.values()]

    cdr_seaice_conc_monthly_qa_flag = cdr_seaice_conc_monthly_qa_flag.assign_attrs(
        long_name="NOAA/NSIDC CDR of Passive Microwave Monthly Northern Hemisphere Sea Ice Concentration QA flags",
        standard_name="status_flag",
        flag_meanings=" ".join(k for k in qa_flag_bitmasks.keys()),
        flag_masks=monthly_qa_values,
        grid_mapping="crs",
        valid_range=(np.uint8(0), np.uint8(sum(monthly_qa_values))),
    )

    cdr_seaice_conc_monthly_qa_flag.encoding = dict(
        dtype=np.uint8,
        zlib=True,
    )

    return cdr_seaice_conc_monthly_qa_flag


def is_empty_month(
    platform_id: SUPPORTED_PLATFORM_ID,
    hemisphere: Hemisphere,
    day_in_month: dt.date,
) -> bool:
    """Determine if this is a month that should have
    all of its values set to missing"""
    year_month_str = f"{day_in_month.year:4d}-{day_in_month.month:02d}"
    empty_year_months = months_of_cdr_missing_data(platform_id, hemisphere)
    return year_month_str in empty_year_months


def calc_cdr_seaice_conc_monthly(
    *,
    daily_ds_for_month: xr.Dataset,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ancillary_source: ANCILLARY_SOURCES,
) -> xr.DataArray:
    """Create the `cdr_seaice_conc_monthly` variable."""
    daily_conc_for_month = daily_ds_for_month.cdr_seaice_conc
    conc_monthly = daily_conc_for_month.mean(dim="time", skipna=True)

    # Fill with empty if this is an empty month
    # NOTE: Need to do this here because num_missing_conc_pixels is in this function
    platform_id = daily_ds_for_month.platform_id
    hemisphere = get_hemisphere_from_crs_da(daily_ds_for_month.crs)
    day_in_month = daily_ds_for_month.time[0].dt.date.data.tolist()

    if is_empty_month(
        platform_id=platform_id,
        hemisphere=hemisphere,
        day_in_month=day_in_month,
    ):
        conc_monthly.data[:] = np.nan

    # Clamp lower bound to threshold. note that `conc_monthly` are fractions
    # (e.g., 0.1 is 10% SIC.)
    conc_threshold_percent = get_monthly_cdr_conc_threshold()
    conc_threshold_frac = conc_threshold_percent / 100.0
    conc_monthly = conc_monthly.where(
        np.isnan(conc_monthly) | (conc_monthly >= conc_threshold_frac),
        other=0,
    )

    conc_monthly.name = "cdr_seaice_conc_monthly"
    conc_monthly = conc_monthly.assign_attrs(
        long_name="NOAA/NSIDC CDR of Passive Microwave Monthly Northern Hemisphere Sea Ice Concentration",
        standard_name="sea_ice_area_fraction",
        coverage_content_type="image",
        units="1",
        valid_range=(np.uint8(0), np.uint8(100)),
        grid_mapping="crs",
    )

    conc_monthly.encoding.update(
        scale_factor=np.float32(0.01),
        add_offset=np.float32(0.0),
        dtype=np.uint8,
        _FillValue=255,
        zlib=True,
    )

    return conc_monthly


def calc_cdr_seaice_conc_monthly_stdev(
    *,
    daily_cdr_seaice_conc: xr.DataArray,
) -> xr.DataArray:
    """
    Create the `cdr_seaice_conc_monthly_stdev` variable.

    Note: Using np.std() instead of DataArray.std() eliminates
          a div by zero warning from but means that the DataArray
          must be set up explicitly instead of resulting from
          the DataArray.std() operation.  Attributes and encoding
          are explicitly specified here just as they need to be
          when using functions on a DataArray.
    Note: In numpy array terms, "axis=0" refers to the time axis
          because the dimensions of the DataArray are ("time", "y", "x").
    """
    cdr_seaice_conc_monthly_stdev_np = np.nanstd(
        np.array(daily_cdr_seaice_conc),
        axis=daily_cdr_seaice_conc.get_axis_num("time"),
        ddof=1,
    )

    # Extract non-'time' dims and coords from DataArray
    dims_without_time = [dim for dim in daily_cdr_seaice_conc.dims if dim != "time"]
    coords_without_time = [
        daily_cdr_seaice_conc[dim] for dim in dims_without_time if dim != "time"
    ]

    cdr_seaice_conc_monthly_stdev = xr.DataArray(
        data=cdr_seaice_conc_monthly_stdev_np,
        name="cdr_seaice_conc_monthly_stdev",
        coords=coords_without_time,
        dims=dims_without_time,
        attrs=dict(
            long_name="NOAA/NSIDC CDR of Passive Microwave Monthly Northern Hemisphere Sea Ice Concentration Source Estimated Standard Deviation",
            valid_range=(np.float32(0.0), np.float32(1.0)),
            grid_mapping="crs",
            units="1",
        ),
    )

    cdr_seaice_conc_monthly_stdev.encoding = dict(
        _FillValue=-1,
        zlib=True,
    )

    return cdr_seaice_conc_monthly_stdev


def calc_cdr_melt_onset_day_monthly(
    *,
    daily_melt_onset_for_month: xr.DataArray,
) -> xr.DataArray:
    """Create the `cdr_melt_onset_day_monthly` variable."""
    # Create `cdr_melt_onset_day_monthly`. This is the value from
    # the last day of the month unless the month is incomplete.
    # xarray uses np.datetime64[ns] for time
    doy_list: list[int] = [
        int(date.strftime("%j"))
        for date in daily_melt_onset_for_month.time.dt.date.values
    ]
    if MELT_SEASON_LAST_DOY in doy_list:
        day_244_idx = doy_list.index(MELT_SEASON_LAST_DOY)
        max_date = daily_melt_onset_for_month.time[day_244_idx].values
        logger.info(f"Found non-max date with melt: {max_date}")
    else:
        max_date = daily_melt_onset_for_month.time.max()

    cdr_melt_onset_day_monthly = daily_melt_onset_for_month.sel(time=max_date)
    cdr_melt_onset_day_monthly.name = "cdr_melt_onset_day_monthly"
    cdr_melt_onset_day_monthly = cdr_melt_onset_day_monthly.drop_vars("time")

    cdr_melt_onset_day_monthly = cdr_melt_onset_day_monthly.assign_attrs(
        long_name="NOAA/NSIDC CDR Monthly Day of Snow Melt Onset Over Sea Ice",
        units="1",
        valid_range=(np.ubyte(0), np.ubyte(255)),
        grid_mapping="crs",
        comment="Value of 0 indicates sea ice concentration less than 50%"
        " at start of melt season; values of 60-244 indicate day"
        " of year of snow melt onset on sea ice detected during"
        " melt season; value of 255 indicates no melt detected"
        " during melt season, including non-ocean grid cells.",
    )
    cdr_melt_onset_day_monthly.encoding = dict(
        _FillValue=None,
        dtype=np.uint8,
        zlib=True,
    )

    return cdr_melt_onset_day_monthly


def _assign_time_to_monthly_ds(
    *,
    monthly_ds: xr.Dataset,
    year: int,
    month: int,
) -> xr.Dataset:
    """Add the time coordinate/dimension to a monthly dataset."""
    # TODO: should this step be done in the `calc_*` functions?
    # assign the time dimension
    with_time = monthly_ds.copy()

    with_time = with_time.expand_dims(
        dim=dict(
            # Time is the first of the month
            time=[dt.datetime(year, month, 1)],
        ),
        # Time should be the first dim.
        axis=0,
    )

    with_time["time"] = with_time.time.assign_attrs(
        dict(
            long_name="ANSI date",
            axis="T",
            coverage_content_type="coordinate",
            standard_name="time",
        )
    )

    with_time.time.encoding = dict(
        units="days since 1970-01-01 00:00:00",
        calendar="standard",
    )

    return with_time


def calc_surface_type_mask_monthly(
    *,
    daily_ds_for_month: xr.Dataset,
    hemisphere: Hemisphere,
) -> xr.DataArray:
    daily_surf_mask = daily_ds_for_month.surface_type_mask

    # initialize the surface mask w/ the latest surface mask.
    monthly_surface_mask = daily_surf_mask.isel(time=-1)

    # The NH monthly surface mask should have a pole-hole that is the combination
    # of all contributing pole-holes.
    if hemisphere == NORTH:
        pole_hole_value = flag_value_for_meaning(
            var=daily_surf_mask, meaning="polehole_mask"
        )
        monthly_surface_mask = monthly_surface_mask.where(
            cond=~(daily_surf_mask == pole_hole_value).any(dim="time"),
            other=pole_hole_value,
        )

    monthly_surface_mask = monthly_surface_mask.drop_vars("time")

    return monthly_surface_mask


def make_intermediate_monthly_ds(
    *,
    daily_ds_for_month: xr.Dataset,
    platform_id: SUPPORTED_PLATFORM_ID,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ancillary_source: ANCILLARY_SOURCES,
) -> xr.Dataset:
    """Create a monthly dataset from daily data.

    Resulting monthly dataset is ready for writing to NetCDF file as a standard
    monthly data file.
    """
    # Min-day check
    check_min_days_for_valid_month(
        daily_ds_for_month=daily_ds_for_month,
        platform_id=platform_id,
    )

    # create `cdr_seaice_conc_monthly`. This is the combined monthly SIC.
    # The `cdr_seaice_conc` variable has temporally filled data and flags.
    cdr_seaice_conc_monthly = calc_cdr_seaice_conc_monthly(
        daily_ds_for_month=daily_ds_for_month,
        hemisphere=hemisphere,
        resolution=resolution,
        ancillary_source=ancillary_source,
    )

    # Create `cdr_seaice_conc_monthly_stdev`, the standard deviation of the
    # sea ice concentration.

    cdr_seaice_conc_monthly_stdev = calc_cdr_seaice_conc_monthly_stdev(
        daily_cdr_seaice_conc=daily_ds_for_month.cdr_seaice_conc,
    )

    cdr_seaice_conc_monthly_qa_flag = calc_cdr_seaice_conc_monthly_qa_flag(
        daily_ds_for_month=daily_ds_for_month,
        cdr_seaice_conc_monthly=cdr_seaice_conc_monthly,
        hemisphere=hemisphere,
    )

    # Set stdev to invalid and QA to all-missing if this is an empty month
    platform_id = daily_ds_for_month.platform_id
    hemisphere = get_hemisphere_from_crs_da(daily_ds_for_month.crs)
    day_in_month = daily_ds_for_month.time[0].dt.date.data.tolist()
    if is_empty_month(
        platform_id=platform_id,
        hemisphere=hemisphere,
        day_in_month=day_in_month,
    ):
        cdr_seaice_conc_monthly_stdev.data[:] = -1.0
        cdr_seaice_conc_monthly_qa_flag.data[:] = 0

    surface_type_mask_monthly = calc_surface_type_mask_monthly(
        daily_ds_for_month=daily_ds_for_month,
        hemisphere=hemisphere,
    )

    monthly_ds_data_vars = dict(
        cdr_seaice_conc_monthly=cdr_seaice_conc_monthly,
        cdr_seaice_conc_monthly_stdev=cdr_seaice_conc_monthly_stdev,
        cdr_seaice_conc_monthly_qa_flag=cdr_seaice_conc_monthly_qa_flag,
        surface_type_mask=surface_type_mask_monthly,
    )
    # Add monthly melt onset if the hemisphere is north. We don't detect melt in
    # the southern hemisphere.
    if hemisphere == NORTH:
        cdr_melt_onset_day_monthly = calc_cdr_melt_onset_day_monthly(
            daily_melt_onset_for_month=daily_ds_for_month.cdr_melt_onset_day,
        )
        monthly_ds_data_vars["cdr_melt_onset_day_monthly"] = cdr_melt_onset_day_monthly

    monthly_ds = xr.Dataset(
        data_vars=monthly_ds_data_vars,
    )

    monthly_ds = _assign_time_to_monthly_ds(
        monthly_ds=monthly_ds,
        year=daily_ds_for_month.year,
        month=daily_ds_for_month.month,
    )

    monthly_ds["crs"] = daily_ds_for_month.crs.isel(time=0).drop_vars("time")

    # Set global attributes
    monthly_ds_global_attrs = get_global_attrs(
        time=monthly_ds.time,
        temporality="monthly",
        aggregate=False,
        source=", ".join([fp.item().name for fp in daily_ds_for_month.filepaths]),
        # TODO: consider providing all platforms that went into month?
        # This would be consistent with how we handle the aggregate filenames.
        # Is it misleading to indicate that a month is a single platform
        # when it may not really be?
        platform_ids=[platform_id],
        resolution=resolution,
        hemisphere=hemisphere,
        ancillary_source=ancillary_source,
    )
    monthly_ds.attrs.update(monthly_ds_global_attrs)

    return monthly_ds.compute()


def get_intermediate_monthly_dir(*, intermediate_output_dir: Path) -> Path:
    monthly_dir = intermediate_output_dir / "monthly"
    monthly_dir.mkdir(parents=True, exist_ok=True)

    return monthly_dir


def get_intermediate_monthly_filepath(
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    platform_id: SUPPORTED_PLATFORM_ID,
    year: int,
    month: int,
    intermediate_output_dir: Path,
) -> Path:
    output_dir = get_intermediate_monthly_dir(
        intermediate_output_dir=intermediate_output_dir,
    )

    output_fn = standard_monthly_filename(
        hemisphere=hemisphere,
        resolution=resolution,
        platform_id=platform_id,
        year=year,
        month=month,
    )

    output_path = output_dir / output_fn

    return output_path


def make_intermediate_monthly_nc(
    *,
    year: int,
    month: int,
    hemisphere: Hemisphere,
    intermediate_output_dir: Path,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ancillary_source: ANCILLARY_SOURCES,
    is_nrt: bool,
) -> Path:
    daily_ds_for_month = get_daily_ds_for_month(
        year=year,
        month=month,
        intermediate_output_dir=intermediate_output_dir,
        hemisphere=hemisphere,
        resolution=resolution,
        is_nrt=is_nrt,
    )

    platform_id = daily_ds_for_month.platform_id

    output_path = get_intermediate_monthly_filepath(
        hemisphere=hemisphere,
        resolution=resolution,
        platform_id=platform_id,
        year=year,
        month=month,
        intermediate_output_dir=intermediate_output_dir,
    )

    monthly_ds = make_intermediate_monthly_ds(
        daily_ds_for_month=daily_ds_for_month,
        platform_id=platform_id,
        hemisphere=hemisphere,
        resolution=resolution,
        ancillary_source=ancillary_source,
    )

    monthly_ds = remove_FillValue_from_coordinate_vars(monthly_ds)
    monthly_ds.to_netcdf(
        output_path,
        unlimited_dims=[
            "time",
        ],
    )
    logger.info(
        f"Wrote intermediate monthly file for {year=} and {month=} using {len(daily_ds_for_month.time)} daily files to {output_path}"
    )

    return output_path


@click.command(name="intermediate-monthly")
@click.option(
    "--year",
    required=True,
    type=int,
    help="Year for which to create the monthly file.",
)
@click.option(
    "--month",
    required=True,
    type=int,
    help="Month for which to create the monthly file.",
)
@click.option(
    "--end-year",
    required=False,
    default=None,
    type=int,
    help="If given, the end year for which to create monthly files.",
)
@click.option(
    "--end-month",
    required=False,
    default=None,
    type=int,
    help="If given, the end year for which to create monthly files.",
)
@click.option(
    "-h",
    "--hemisphere",
    required=True,
    type=click.Choice(get_args(Hemisphere)),
)
@click.option(
    "--base-output-dir",
    required=True,
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
    default=DEFAULT_BASE_OUTPUT_DIR,
    help=(
        "Base output directory for standard ECDR outputs."
        " Subdirectories are created for outputs of"
        " different stages of processing."
    ),
    show_default=True,
)
@click.option(
    "-r",
    "--resolution",
    required=True,
    type=click.Choice(get_args(ECDR_SUPPORTED_RESOLUTIONS)),
)
@click.option(
    "--ancillary-source",
    required=True,
    type=click.Choice(get_args(ANCILLARY_SOURCES)),
)
@click.option(
    "--is-nrt",
    required=False,
    is_flag=True,
    help=("Create intermediate monthly file in NRT mode (uses NRT-stype filename)."),
)
def cli(
    *,
    year: int,
    month: int,
    end_year: int | None,
    end_month: int | None,
    hemisphere: Hemisphere,
    base_output_dir: Path,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ancillary_source: ANCILLARY_SOURCES,
    is_nrt: bool,
):
    if end_year is None:
        end_year = year
    if end_month is None:
        end_month = month

    intermediate_output_dir = get_intermediate_output_dir(
        base_output_dir=base_output_dir,
        hemisphere=hemisphere,
    )
    error_periods = []
    for period in pd.period_range(
        start=pd.Period(year=year, month=month, freq="M"),
        end=pd.Period(year=end_year, month=end_month, freq="M"),
        freq="M",
    ):
        try:
            make_intermediate_monthly_nc(
                year=period.year,
                month=period.month,
                intermediate_output_dir=intermediate_output_dir,
                hemisphere=hemisphere,
                resolution=resolution,
                ancillary_source=ancillary_source,
                is_nrt=is_nrt,
            )
        except Exception:
            error_periods.append(period)
            logger.exception(
                f"Failed to create monthly data for year={period.year} month={period.month}"
            )

    if error_periods:
        str_formatted_dates = "\n".join(
            period.strftime("%Y-%m") for period in error_periods
        )
        raise RuntimeError(
            f"Encountered {len(error_periods)} failures."
            f" Data for the following months were not created:\n{str_formatted_dates}"
        )
