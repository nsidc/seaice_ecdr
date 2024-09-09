import datetime as dt
import re
from pathlib import Path
from typing import Iterator, Literal, cast, get_args

import numpy as np
import pandas as pd
import xarray as xr
from pm_tb_data._types import Hemisphere

from seaice_ecdr._types import ECDR_SUPPORTED_RESOLUTIONS
from seaice_ecdr.ancillary import get_ocean_mask
from seaice_ecdr.constants import ECDR_PRODUCT_VERSION
from seaice_ecdr.grid_id import get_grid_id
from seaice_ecdr.platforms import SUPPORTED_PLATFORM_ID


def standard_daily_filename(
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    sat: SUPPORTED_PLATFORM_ID,
    date: dt.date,
) -> str:
    """Return standard daily NetCDF filename.

    North Daily files: sic_psn12.5_YYYYMMDD_sat_v05r01.nc
    South Daily files: sic_pss12.5_YYYYMMDD_sat_v05r01.nc
    """
    grid_id = get_grid_id(
        hemisphere=hemisphere,
        resolution=resolution,
    )
    fn = f"sic_{grid_id}_{date:%Y%m%d}_{sat}_{ECDR_PRODUCT_VERSION}.nc"

    return fn


def nrt_daily_filename(
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    sat: SUPPORTED_PLATFORM_ID,
    date: dt.date,
) -> str:
    standard_fn = standard_daily_filename(
        hemisphere=hemisphere,
        resolution=resolution,
        sat=sat,
        date=date,
    )
    standard_fn_path = Path(standard_fn)

    fn_base = standard_fn_path.stem
    ext = standard_fn_path.suffix
    nrt_fn = fn_base + "_P" + ext

    return nrt_fn


def standard_daily_aggregate_filename(
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    start_date: dt.date,
    end_date: dt.date,
) -> str:
    """Return standard daily aggregate NetCDF filename.

    North Daily aggregate files: sic_psn12.5_YYYYMMDD-YYYYMMDD_v05r01.nc
    South Daily aggregate files: sic_pss12.5_YYYYMMDD-YYYYMMDD_v05r01.nc
    """
    grid_id = get_grid_id(
        hemisphere=hemisphere,
        resolution=resolution,
    )
    fn = (
        f"sic_{grid_id}_{start_date:%Y%m%d}-{end_date:%Y%m%d}_{ECDR_PRODUCT_VERSION}.nc"
    )

    return fn


def standard_monthly_filename(
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    sat: SUPPORTED_PLATFORM_ID,
    year: int,
    month: int,
) -> str:
    """Return standard monthly NetCDF filename.

    North Monthly files: sic_psn12.5_YYYYMM_sat_v05r01.nc
    South Monthly files: sic_pss12.5_YYYYMM_sat_v05r01.nc
    """
    grid_id = get_grid_id(
        hemisphere=hemisphere,
        resolution=resolution,
    )
    fn = f"sic_{grid_id}_{year}{month:02}_{sat}_{ECDR_PRODUCT_VERSION}.nc"

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

    North Monthly aggregate files: sic_psn12.5_YYYYMM-YYYYMM_v05r01.nc
    South Monthly aggregate files: sic_pss12.5_YYYYMM-YYYYMM_v05r01.nc
    """
    date_str = f"{start_year}{start_month:02}-{end_year}{end_month:02}"

    grid_id = get_grid_id(
        hemisphere=hemisphere,
        resolution=resolution,
    )

    fn = f"sic_{grid_id}_{date_str}_{ECDR_PRODUCT_VERSION}.nc"

    return fn


# This regex works for both daily and monthly filenames.
STANDARD_FN_REGEX = re.compile(r"sic_ps.*_.*_(?P<sat>.*)_.*.nc")


def sat_from_filename(filename: str) -> SUPPORTED_PLATFORM_ID:
    match = STANDARD_FN_REGEX.match(filename)

    if not match:
        raise RuntimeError(f"Failed to parse satellite from {filename}")

    sat = match.group("sat")

    # Ensure the sat is expected.
    assert sat in get_args(SUPPORTED_PLATFORM_ID)
    sat = cast(SUPPORTED_PLATFORM_ID, sat)

    return sat


def date_range(*, start_date: dt.date, end_date: dt.date) -> Iterator[dt.date]:
    """Yield a dt.date object representing each day between start_date and end_date."""
    for pd_timestamp in pd.date_range(start=start_date, end=end_date, freq="D"):
        yield pd_timestamp.date()


def get_dates_by_year(dates: list[dt.date]) -> list[list[dt.date]]:
    """Given a list of dates, return the dates grouped by year."""
    years = sorted(np.unique([date.year for date in dates]))
    dates_by_year = {}
    for year in years:
        dates_by_year[year] = sorted([date for date in dates if date.year == year])

    dates_by_year_list = list(dates_by_year.values())

    return dates_by_year_list


def get_ecdr_grid_shape(
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
) -> tuple[int, int]:
    grid_shapes = {
        "25": {"north": (448, 304), "south": (332, 316)},
        "12.5": {"north": (896, 608), "south": (664, 632)},
    }

    grid_shape = grid_shapes[resolution][hemisphere]

    return grid_shape


def get_num_missing_pixels(
    *,
    seaice_conc_var: xr.DataArray,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
) -> int:
    """The number of missing pixels is anywhere that there are nans over ocean."""
    ocean_mask = get_ocean_mask(
        hemisphere=hemisphere,
        resolution=resolution,
    )

    num_missing_pixels = int((seaice_conc_var.isnull() & ocean_mask).astype(int).sum())

    return num_missing_pixels


def raise_error_for_dates(*, error_dates: list[dt.date]) -> None:
    """If `error_dates` is non-empty, raise an error indicating those dates had a processing failure."""
    if error_dates:
        str_formatted_dates = "\n".join(
            date.strftime("%Y-%m-%d") for date in error_dates
        )
        raise RuntimeError(
            f"Encountered {len(error_dates)} failures."
            f" Data for the following dates were not created:\n{str_formatted_dates}"
        )


def _get_output_dir(
    *,
    base_output_dir: Path,
    hemisphere: Hemisphere,
    is_nrt: bool,
    data_type: Literal["intermediate", "complete"],
) -> Path:
    out_dir = base_output_dir / data_type / hemisphere
    if is_nrt:
        out_dir = out_dir / "nrt"

    out_dir.mkdir(exist_ok=True, parents=True)

    return out_dir


def get_intermediate_output_dir(
    *,
    base_output_dir: Path,
    hemisphere: Hemisphere,
    is_nrt: bool,
) -> Path:
    intermediate_dir = _get_output_dir(
        base_output_dir=base_output_dir,
        hemisphere=hemisphere,
        is_nrt=is_nrt,
        data_type="intermediate",
    )

    return intermediate_dir


def get_complete_output_dir(
    *,
    base_output_dir: Path,
    hemisphere: Hemisphere,
    is_nrt: bool,
) -> Path:
    complete_dir = _get_output_dir(
        base_output_dir=base_output_dir,
        hemisphere=hemisphere,
        is_nrt=is_nrt,
        data_type="complete",
    )

    return complete_dir
