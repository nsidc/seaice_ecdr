import datetime as dt
import re
from pathlib import Path
from typing import Iterable, Iterator, Literal, cast, get_args

import numpy as np
import pandas as pd
import xarray as xr
from pm_tb_data._types import Hemisphere

from seaice_ecdr._types import ECDR_SUPPORTED_RESOLUTIONS
from seaice_ecdr.ancillary import ANCILLARY_SOURCES, get_ocean_mask
from seaice_ecdr.constants import ECDR_NRT_PRODUCT_VERSION, ECDR_PRODUCT_VERSION
from seaice_ecdr.grid_id import get_grid_id
from seaice_ecdr.platforms import SUPPORTED_PLATFORM_ID


def standard_daily_filename(
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    platform_id: SUPPORTED_PLATFORM_ID,
    date: dt.date,
) -> str:
    """Return standard daily NetCDF filename.

    North Daily files: sic_psn12.5_{YYYYMMDD}_{platform_id}_{ECDR_PRODUCT_VERSION}.nc
    South Daily files: sic_pss12.5_{YYYYMMDD}_{platform_id}_{ECDR_PRODUCT_VERSION}.nc
    """
    grid_id = get_grid_id(
        hemisphere=hemisphere,
        resolution=resolution,
    )
    fn = f"sic_{grid_id}_{date:%Y%m%d}_{platform_id}_{ECDR_PRODUCT_VERSION}.nc"

    return fn


def nrt_daily_filename(
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    platform_id: SUPPORTED_PLATFORM_ID,
    date: dt.date,
) -> str:
    standard_fn = standard_daily_filename(
        hemisphere=hemisphere,
        resolution=resolution,
        platform_id=platform_id,
        date=date,
    )
    standard_fn_path = Path(standard_fn)

    fn_base = standard_fn_path.stem
    ext = standard_fn_path.suffix
    nrt_fn = fn_base + "_P" + ext

    # Replace the standard G02202 version number with the NRT version.
    nrt_fn = nrt_fn.replace(
        ECDR_PRODUCT_VERSION.version_str, ECDR_NRT_PRODUCT_VERSION.version_str
    )

    return nrt_fn


def standard_daily_aggregate_filename(
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    start_date: dt.date,
    end_date: dt.date,
) -> str:
    """Return standard daily aggregate NetCDF filename.

    North Daily aggregate files: sic_psn12.5_YYYYMMDD-YYYYMMDD_{ECDR_PRODUCT_VERSION}.nc
    South Daily aggregate files: sic_pss12.5_YYYYMMDD-YYYYMMDD_{ECDR_PRODUCT_VERSION}.nc
    """
    grid_id = get_grid_id(
        hemisphere=hemisphere,
        resolution=resolution,
    )
    fn = (
        f"sic_{grid_id}_{start_date:%Y%m%d}-{end_date:%Y%m%d}_{ECDR_PRODUCT_VERSION}.nc"
    )

    return fn


def _standard_monthly_filename(
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    platform_id: SUPPORTED_PLATFORM_ID | Literal["*"],
    year: int | Literal["*"],
    month: int | Literal["*"],
) -> str:
    """Function that has looser typing for wild-cardable (in a glob) kwargs than
    `standard_monthly_filename`.

    `standard_monthly_filename` is typed more strictly to ensure that output
    filenames conform to our expectations. This lets you pass in e.g., `"*"` for
    `platform_id`.
    """
    grid_id = get_grid_id(
        hemisphere=hemisphere,
        resolution=resolution,
    )
    if isinstance(month, int):
        month_str = f"{month:02}"
    else:
        month_str = month

    year_month = f"{year}{month_str}"
    # de-duplicate "**" if year and month are both a wildcard.

    year_month = year_month.replace("**", "*")
    fn = f"sic_{grid_id}_{year_month}_{platform_id}_{ECDR_PRODUCT_VERSION}.nc"

    return fn


def standard_monthly_filename(
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    platform_id: SUPPORTED_PLATFORM_ID,
    year: int,
    month: int,
) -> str:
    """Return standard monthly NetCDF filename.

    North Monthly files: sic_psn12.5_{YYYYMM}_{platform_id}_{ECDR_PRODUCT_VERSION}.nc
    South Monthly files: sic_pss12.5_{YYYYMM}_{platform_id}_{ECDR_PRODUCT_VERSION}.nc
    """
    return _standard_monthly_filename(
        hemisphere=hemisphere,
        resolution=resolution,
        platform_id=platform_id,
        year=year,
        month=month,
    )


def find_standard_monthly_netcdf_files(
    *,
    search_dir: Path,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    platform_id: SUPPORTED_PLATFORM_ID | Literal["*"],
    year: int | Literal["*"],
    month: int | Literal["*"],
) -> list[Path]:
    """Find standard monthly nc files matching the given params.

    `platform_id`, `year`, and `month` are wild-cardable (e.g., one can pass in
    a `*`) to search for files that match multiple platforms/years/months.
    """
    fn_glob = _standard_monthly_filename(
        hemisphere=hemisphere,
        resolution=resolution,
        platform_id=platform_id,
        year=year,
        month=month,
    )

    results = search_dir.glob(fn_glob)
    return list(sorted(results))


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

    North Monthly aggregate files: sic_psn12.5_YYYYMM-YYYYMM_{ECDR_PRODUCT_VERSION}.nc
    South Monthly aggregate files: sic_pss12.5_YYYYMM-YYYYMM_{ECDR_PRODUCT_VERSION}.nc
    """
    date_str = f"{start_year}{start_month:02}-{end_year}{end_month:02}"

    grid_id = get_grid_id(
        hemisphere=hemisphere,
        resolution=resolution,
    )

    fn = f"sic_{grid_id}_{date_str}_{ECDR_PRODUCT_VERSION}.nc"

    return fn


# This regex works for both daily and monthly filenames.
STANDARD_FN_REGEX = re.compile(r"sic_ps.*_.*_(?P<platform_id>.*)_.*.nc")


def platform_id_from_filename(filename: str) -> SUPPORTED_PLATFORM_ID:
    match = STANDARD_FN_REGEX.match(filename)

    if not match:
        raise RuntimeError(f"Failed to parse platform from {filename}")

    platform_id = match.group("platform_id")

    # Ensure the platform is expected.
    assert platform_id in get_args(SUPPORTED_PLATFORM_ID)
    platform_id = cast(SUPPORTED_PLATFORM_ID, platform_id)

    return platform_id


def date_range(*, start_date: dt.date, end_date: dt.date) -> Iterator[dt.date]:
    """Yield a dt.date object representing each day between start_date and end_date."""
    for pd_timestamp in pd.date_range(start=start_date, end=end_date, freq="D"):
        yield pd_timestamp.date()


def get_dates_by_year(dates: Iterable[dt.date]) -> list[list[dt.date]]:
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
    ancillary_source: ANCILLARY_SOURCES,
) -> int:
    """The number of missing pixels is anywhere that there are nans over ocean."""
    ocean_mask = get_ocean_mask(
        hemisphere=hemisphere,
        resolution=resolution,
        ancillary_source=ancillary_source,
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
    data_type: Literal["intermediate", "complete"],
) -> Path:
    out_dir = base_output_dir / data_type / hemisphere

    out_dir.mkdir(exist_ok=True, parents=True)

    return out_dir


def get_intermediate_output_dir(
    *,
    base_output_dir: Path,
    hemisphere: Hemisphere,
) -> Path:
    intermediate_dir = _get_output_dir(
        base_output_dir=base_output_dir,
        hemisphere=hemisphere,
        data_type="intermediate",
    )

    return intermediate_dir


def get_complete_output_dir(
    *,
    base_output_dir: Path,
    hemisphere: Hemisphere,
) -> Path:
    complete_dir = _get_output_dir(
        base_output_dir=base_output_dir,
        hemisphere=hemisphere,
        data_type="complete",
    )

    return complete_dir
