"""Code for producing the monthly ECDR.

Follows the same procedure as CDR v4.

Variables:

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

from seaice_ecdr._types import ECDR_SUPPORTED_RESOLUTIONS, SUPPORTED_SAT
from seaice_ecdr.ancillary import flag_value_for_meaning
from seaice_ecdr.complete_daily_ecdr import get_ecdr_filepath
from seaice_ecdr.constants import STANDARD_BASE_OUTPUT_DIR
from seaice_ecdr.nc_attrs import get_global_attrs
from seaice_ecdr.util import sat_from_filename, standard_monthly_filename


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
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
) -> list[Path]:
    """Return a list of paths to ECDR daily complete filepaths for the given year and month."""
    data_list = []
    _, last_day_of_month = calendar.monthrange(year, month)
    for period in pd.period_range(
        start=dt.date(year, month, 1),
        end=dt.date(year, month, last_day_of_month),
        freq="D",
    ):
        expected_fp = get_ecdr_filepath(
            date=period.to_timestamp().date(),
            hemisphere=hemisphere,
            resolution=resolution,
            ecdr_data_dir=ecdr_data_dir,
        )
        if expected_fp.is_file():
            data_list.append(expected_fp)
        else:
            logger.warning(f"Expected to find {expected_fp} but found none.")

    if len(data_list) == 0:
        raise RuntimeError("No daily data files found.")

    return data_list


def _sat_for_month(*, sats: list[SUPPORTED_SAT]) -> SUPPORTED_SAT:
    """Returns the satellite from this month given a list of input satellites.

    The sat for monthly files is based on which sat contributes most to the
    month. If two sats contribute equally, use the latest sat in the series.

    Function assumes the list of satellites is already sorted (i.e., the latest
    satellite is `sats[-1]`).
    """
    # More than one sat, we need to choose the most common/latest in the series.
    # `Counter` returns a dict keyed by `sat` with counts as values:
    count = Counter(sats)
    most_common_sats = count.most_common()
    most_common_and_latest_sat = most_common_sats[-1][0]

    return most_common_and_latest_sat


def get_daily_ds_for_month(
    *,
    year: int,
    month: int,
    ecdr_data_dir: Path,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
) -> xr.Dataset:
    """Create an xr.Dataset wtih ECDR complete daily data for a given year and month.

    The resulting xr.Dataset includes:
        * `year` and `month` attribtues.
        * The filepaths of the source data are included in a `filepaths` variable.
    """
    data_list = _get_daily_complete_filepaths_for_month(
        year=year,
        month=month,
        ecdr_data_dir=ecdr_data_dir,
        hemisphere=hemisphere,
        resolution=resolution,
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

    # Extract `sat` from the filenames contributing to this
    # dataset. Ideally, we would use a custom `combine_attrs` when reading the
    # data with `xr.open_mfdataset` in order to get the sat/sensor from global
    # attrs in each of the contributing files. Unfortunately this interface is
    # poorly documented and seems to have limited support. E.g., see
    # https://github.com/pydata/xarray/issues/6679
    sats = []
    for filepath in data_list:
        sats.append(sat_from_filename(filepath.name))

    sat = _sat_for_month(sats=sats)

    ds.attrs["sat"] = sat

    logger.info(
        f"Created daily ds for {year=} {month=} from {len(ds.time)} complete daily files."
    )

    return ds


# TODO: utilize these in the daily complete processing. Or use constant from
# that module!
QA_OF_CDR_SEAICE_CONC_DAILY_BITMASKS = OrderedDict(
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
QA_OF_CDR_SEAICE_CONC_MONTHLY_BITMASKS = OrderedDict(
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


def _qa_field_has_flag(*, qa_field: xr.DataArray, flag_value: int) -> xr.DataArray:
    """Returns a boolean DataArray indicating where a flag value occurs in the given `qa_field`."""
    qa_field_contains_flag = (
        qa_field.where(qa_field.isnull(), qa_field.astype(int) & flag_value)
        == flag_value
    )

    return qa_field_contains_flag


def calc_qa_of_cdr_seaice_conc_monthly(
    *,
    daily_ds_for_month: xr.Dataset,
    cdr_seaice_conc_monthly: xr.DataArray,
) -> xr.DataArray:
    """Create `qa_of_cdr_seaice_conc_monthly`."""
    # initialize the variable
    qa_of_cdr_seaice_conc_monthly = xr.full_like(
        cdr_seaice_conc_monthly,
        fill_value=0,
        dtype=np.uint8,
    )
    qa_of_cdr_seaice_conc_monthly.name = "qa_of_cdr_seaice_conc_monthly"

    average_exceeds_15 = cdr_seaice_conc_monthly > 0.15
    qa_of_cdr_seaice_conc_monthly = qa_of_cdr_seaice_conc_monthly.where(
        ~average_exceeds_15,
        qa_of_cdr_seaice_conc_monthly
        + QA_OF_CDR_SEAICE_CONC_MONTHLY_BITMASKS["average_concentration_exceeds_0.15"],
    )

    average_exceeds_30 = cdr_seaice_conc_monthly > 0.30
    qa_of_cdr_seaice_conc_monthly = qa_of_cdr_seaice_conc_monthly.where(
        ~average_exceeds_30,
        qa_of_cdr_seaice_conc_monthly
        + QA_OF_CDR_SEAICE_CONC_MONTHLY_BITMASKS["average_concentration_exceeds_0.30"],
    )

    days_in_ds = len(daily_ds_for_month.time)
    majority_of_days = (days_in_ds + 1) // 2

    at_least_half_have_sic_gt_15 = (daily_ds_for_month.cdr_seaice_conc > 0.15).sum(
        dim="time"
    ) >= majority_of_days
    qa_of_cdr_seaice_conc_monthly = qa_of_cdr_seaice_conc_monthly.where(
        ~at_least_half_have_sic_gt_15,
        qa_of_cdr_seaice_conc_monthly
        + QA_OF_CDR_SEAICE_CONC_MONTHLY_BITMASKS[
            "at_least_half_the_days_have_sea_ice_conc_exceeds_0.15"
        ],
    )

    at_least_half_have_sic_gt_30 = (daily_ds_for_month.cdr_seaice_conc > 0.30).sum(
        dim="time"
    ) >= majority_of_days
    qa_of_cdr_seaice_conc_monthly = qa_of_cdr_seaice_conc_monthly.where(
        ~at_least_half_have_sic_gt_30,
        qa_of_cdr_seaice_conc_monthly
        + QA_OF_CDR_SEAICE_CONC_MONTHLY_BITMASKS[
            "at_least_half_the_days_have_sea_ice_conc_exceeds_0.30"
        ],
    )

    # Use "invalid_ice_mask_applied", which is actually the invalid ice mask.
    invalid_ice_mask_applied = _qa_field_has_flag(
        qa_field=daily_ds_for_month.qa_of_cdr_seaice_conc,
        flag_value=QA_OF_CDR_SEAICE_CONC_DAILY_BITMASKS["invalid_ice_mask_applied"],
    ).any(dim="time")
    qa_of_cdr_seaice_conc_monthly = qa_of_cdr_seaice_conc_monthly.where(
        ~invalid_ice_mask_applied,
        qa_of_cdr_seaice_conc_monthly
        + QA_OF_CDR_SEAICE_CONC_MONTHLY_BITMASKS["invalid_ice_mask_applied"],
    )

    at_least_one_day_during_month_has_spatial_interpolation = _qa_field_has_flag(
        qa_field=daily_ds_for_month.qa_of_cdr_seaice_conc,
        flag_value=QA_OF_CDR_SEAICE_CONC_DAILY_BITMASKS[
            "spatial_interpolation_applied"
        ],
    ).any(dim="time")
    qa_of_cdr_seaice_conc_monthly = qa_of_cdr_seaice_conc_monthly.where(
        ~at_least_one_day_during_month_has_spatial_interpolation,
        qa_of_cdr_seaice_conc_monthly
        + QA_OF_CDR_SEAICE_CONC_MONTHLY_BITMASKS[
            "at_least_one_day_during_month_has_spatial_interpolation"
        ],
    )

    at_least_one_day_during_month_has_temporal_interpolation = _qa_field_has_flag(
        qa_field=daily_ds_for_month.qa_of_cdr_seaice_conc,
        flag_value=QA_OF_CDR_SEAICE_CONC_DAILY_BITMASKS[
            "temporal_interpolation_applied"
        ],
    ).any(dim="time")
    qa_of_cdr_seaice_conc_monthly = qa_of_cdr_seaice_conc_monthly.where(
        ~at_least_one_day_during_month_has_temporal_interpolation,
        qa_of_cdr_seaice_conc_monthly
        + QA_OF_CDR_SEAICE_CONC_MONTHLY_BITMASKS[
            "at_least_one_day_during_month_has_temporal_interpolation"
        ],
    )

    at_least_one_day_during_month_has_melt_detected = _qa_field_has_flag(
        qa_field=daily_ds_for_month.qa_of_cdr_seaice_conc,
        flag_value=QA_OF_CDR_SEAICE_CONC_DAILY_BITMASKS["start_of_melt_detected"],
    ).any(dim="time")
    qa_of_cdr_seaice_conc_monthly = qa_of_cdr_seaice_conc_monthly.where(
        ~at_least_one_day_during_month_has_melt_detected,
        qa_of_cdr_seaice_conc_monthly
        + QA_OF_CDR_SEAICE_CONC_MONTHLY_BITMASKS[
            "at_least_one_day_during_month_has_melt_detected"
        ],
    )

    qa_of_cdr_seaice_conc_monthly = qa_of_cdr_seaice_conc_monthly.assign_attrs(
        long_name="Passive Microwave Monthly Northern Hemisphere Sea Ice Concentration QC flags",
        standard_name="status_flag",
        flag_meanings=" ".join(
            k for k in QA_OF_CDR_SEAICE_CONC_MONTHLY_BITMASKS.keys()
        ),
        flag_masks=[
            np.uint8(v) for v in QA_OF_CDR_SEAICE_CONC_MONTHLY_BITMASKS.values()
        ],
        grid_mapping="crs",
        valid_range=(np.uint8(0), np.uint8(255)),
    )

    qa_of_cdr_seaice_conc_monthly.encoding = dict(
        dtype=np.uint8,
        zlib=True,
    )

    return qa_of_cdr_seaice_conc_monthly


def _calc_conc_monthly(
    *,
    daily_conc_for_month: xr.DataArray,
    long_name: str,
    name: str,
) -> xr.DataArray:
    conc_monthly = daily_conc_for_month.mean(dim="time")

    conc_monthly.name = name
    conc_monthly = conc_monthly.assign_attrs(
        long_name=long_name,
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


def calc_cdr_seaice_conc_monthly(
    *,
    daily_ds_for_month: xr.Dataset,
) -> xr.DataArray:
    """Create the `cdr_seaice_conc_monthly` variable."""
    cdr_seaice_conc_monthly = _calc_conc_monthly(
        daily_conc_for_month=daily_ds_for_month.cdr_seaice_conc,
        long_name="NOAA/NSIDC Climate Data Record of Passive Microwave Monthly Northern Hemisphere Sea Ice Concentration",
        name="cdr_seaice_conc_monthly",
    )

    cdr_seaice_conc_monthly.attrs[
        "reference"
    ] = "https://nsidc.org/data/g02202/versions/5"

    return cdr_seaice_conc_monthly


def calc_stdv_of_cdr_seaice_conc_monthly(
    *,
    daily_cdr_seaice_conc: xr.DataArray,
) -> xr.DataArray:
    """Create the `stdv_of_cdr_seaice_conc_monthly` variable."""
    stdv_of_cdr_seaice_conc_monthly = daily_cdr_seaice_conc.std(
        dim="time",
        ddof=1,
    )
    stdv_of_cdr_seaice_conc_monthly.name = "stdv_of_cdr_seaice_conc_monthly"

    stdv_of_cdr_seaice_conc_monthly = stdv_of_cdr_seaice_conc_monthly.assign_attrs(
        long_name="Passive Microwave Monthly Northern Hemisphere Sea Ice Concentration Source Estimated Standard Deviation",
        valid_range=(np.float32(0.0), np.float32(1.0)),
        grid_mapping="crs",
        units="1",
    )

    stdv_of_cdr_seaice_conc_monthly.encoding = dict(
        _FillValue=-1,
        zlib=True,
    )

    return stdv_of_cdr_seaice_conc_monthly


def calc_melt_onset_day_cdr_seaice_conc_monthly(
    *,
    daily_melt_onset_for_month: xr.DataArray,
) -> xr.DataArray:
    """Create the `melt_onset_day_cdr_seaice_conc_monthly` variable."""
    # Create `melt_onset_day_cdr_seaice_conc_monthly`. This is the value from
    # the last day of the month.
    melt_onset_day_cdr_seaice_conc_monthly = daily_melt_onset_for_month.sel(
        time=daily_melt_onset_for_month.time.max()
    )
    melt_onset_day_cdr_seaice_conc_monthly.name = (
        "melt_onset_day_cdr_seaice_conc_monthly"
    )
    melt_onset_day_cdr_seaice_conc_monthly = (
        melt_onset_day_cdr_seaice_conc_monthly.drop_vars("time")
    )

    melt_onset_day_cdr_seaice_conc_monthly = (
        melt_onset_day_cdr_seaice_conc_monthly.assign_attrs(
            long_name="Monthly Day of Snow Melt Onset Over Sea Ice",
            units="1",
            valid_range=(np.ubyte(60), np.ubyte(255)),
            grid_mapping="crs",
        )
    )
    melt_onset_day_cdr_seaice_conc_monthly.encoding = dict(
        _FillValue=None,
        dtype=np.uint8,
        zlib=True,
    )

    return melt_onset_day_cdr_seaice_conc_monthly


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


def make_monthly_ds(
    *,
    daily_ds_for_month: xr.Dataset,
    sat: SUPPORTED_SAT,
    hemisphere: Hemisphere,
) -> xr.Dataset:
    """Create a monthly dataset from daily data.

    Resulting monthly dataset is ready for writing to NetCDF file as a standard
    monthly data file.
    """
    # Min-day check
    check_min_days_for_valid_month(
        daily_ds_for_month=daily_ds_for_month,
        sat=sat,
    )

    # create `cdr_seaice_conc_monthly`. This is the combined monthly SIC.
    # The `cdr_seaice_conc` variable has temporally filled data and flags.
    cdr_seaice_conc_monthly = calc_cdr_seaice_conc_monthly(
        daily_ds_for_month=daily_ds_for_month,
    )

    # Create `stdev_of_cdr_seaice_conc_monthly`, the standard deviation of the
    # sea ice concentration.
    stdv_of_cdr_seaice_conc_monthly = calc_stdv_of_cdr_seaice_conc_monthly(
        daily_cdr_seaice_conc=daily_ds_for_month.cdr_seaice_conc,
    )

    qa_of_cdr_seaice_conc_monthly = calc_qa_of_cdr_seaice_conc_monthly(
        daily_ds_for_month=daily_ds_for_month,
        cdr_seaice_conc_monthly=cdr_seaice_conc_monthly,
    )

    surface_type_mask_monthly = calc_surface_type_mask_monthly(
        daily_ds_for_month=daily_ds_for_month,
        hemisphere=hemisphere,
    )

    monthly_ds_data_vars = dict(
        cdr_seaice_conc_monthly=cdr_seaice_conc_monthly,
        stdv_of_cdr_seaice_conc_monthly=stdv_of_cdr_seaice_conc_monthly,
        qa_of_cdr_seaice_conc_monthly=qa_of_cdr_seaice_conc_monthly,
        surface_type_mask=surface_type_mask_monthly,
    )

    # Add monthly melt onset if the hemisphere is north. We don't detect melt in
    # the southern hemisphere.
    if hemisphere == NORTH:
        melt_onset_day_cdr_seaice_conc_monthly = calc_melt_onset_day_cdr_seaice_conc_monthly(
            daily_melt_onset_for_month=daily_ds_for_month.melt_onset_day_cdr_seaice_conc,
        )
        monthly_ds_data_vars[
            "melt_onset_day_cdr_seaice_conc_monthly"
        ] = melt_onset_day_cdr_seaice_conc_monthly

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
        # TODO: consider providing all sats that went into month? This would be
        # consistent with how we handle the aggregate filenames. Is it
        # misleading to indicate that a month is a single sat when it may not
        # really be?
        sats=[sat],
    )
    monthly_ds.attrs.update(monthly_ds_global_attrs)

    return monthly_ds.compute()


def get_monthly_dir(*, ecdr_data_dir: Path) -> Path:
    monthly_dir = ecdr_data_dir / "monthly"
    monthly_dir.mkdir(exist_ok=True)

    return monthly_dir


def get_monthly_filepath(
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    sat: SUPPORTED_SAT,
    year: int,
    month: int,
    ecdr_data_dir: Path,
) -> Path:
    output_dir = get_monthly_dir(ecdr_data_dir=ecdr_data_dir)

    output_fn = standard_monthly_filename(
        hemisphere=hemisphere,
        resolution=resolution,
        sat=sat,
        year=year,
        month=month,
    )

    output_path = output_dir / output_fn

    return output_path


def make_monthly_nc(
    *,
    year: int,
    month: int,
    hemisphere: Hemisphere,
    ecdr_data_dir: Path,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
) -> Path:
    daily_ds_for_month = get_daily_ds_for_month(
        year=year,
        month=month,
        ecdr_data_dir=ecdr_data_dir,
        hemisphere=hemisphere,
        resolution=resolution,
    )

    sat = daily_ds_for_month.sat

    monthly_ds = make_monthly_ds(
        daily_ds_for_month=daily_ds_for_month,
        sat=sat,
        hemisphere=hemisphere,
    )

    output_path = get_monthly_filepath(
        hemisphere=hemisphere,
        resolution=resolution,
        sat=sat,
        year=year,
        month=month,
        ecdr_data_dir=ecdr_data_dir,
    )

    # Set `x` and `y` `_FillValue` to `None`. Although unset initially, `xarray`
    # seems to default to `np.nan` for variables without a FillValue.
    monthly_ds.x.encoding["_FillValue"] = None
    monthly_ds.y.encoding["_FillValue"] = None

    monthly_ds.to_netcdf(
        output_path,
        unlimited_dims=[
            "time",
        ],
    )
    logger.info(f"Wrote monthly file for {year=} and {month=} to {output_path}")

    return output_path


@click.command(name="monthly")
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
    "--ecdr-data-dir",
    required=True,
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
    default=STANDARD_BASE_OUTPUT_DIR,
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
def cli(
    *,
    year: int,
    month: int,
    end_year: int | None,
    end_month: int | None,
    hemisphere: Hemisphere,
    ecdr_data_dir: Path,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
):
    if end_year is None:
        end_year = year
    if end_month is None:
        end_month = month

    for period in pd.period_range(
        start=pd.Period(year=year, month=month, freq="M"),
        end=pd.Period(year=end_year, month=end_month, freq="M"),
        freq="M",
    ):
        make_monthly_nc(
            year=period.year,
            month=period.month,
            ecdr_data_dir=ecdr_data_dir,
            hemisphere=hemisphere,
            resolution=resolution,
        )
