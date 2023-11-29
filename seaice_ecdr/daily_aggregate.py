"""Code to produce daily aggregate files from daily complete data.
"""
import datetime as dt
from pathlib import Path
from typing import Final, get_args

import click
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from pm_tb_data._types import Hemisphere

from seaice_ecdr._types import ECDR_SUPPORTED_RESOLUTIONS, SUPPORTED_SAT
from seaice_ecdr.complete_daily_ecdr import get_ecdr_filepath
from seaice_ecdr.constants import STANDARD_BASE_OUTPUT_DIR
from seaice_ecdr.nc_attrs import get_global_attrs
from seaice_ecdr.util import standard_daily_filename


# TODO: very similar to `monthly._get_daily_complete_filepaths_for_month`. DRY
# out/move into module/subpackage related to daily data.
def _get_daily_complete_filepaths_for_year(
    *,
    year: int,
    ecdr_data_dir: Path,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
) -> list[Path]:
    data_list = []
    for period in pd.period_range(start=dt.date(year, 1, 1), end=dt.date(year, 12, 31)):
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


# TODO: this is also very similar to `monthly.get_daily_ds_for_month`.
def get_daily_ds_for_year(
    *,
    year: int,
    ecdr_data_dir: Path,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
) -> xr.Dataset:
    """Create an xr.Dataset wtih ECDR complete daily data for a given year.

    The resulting xr.Dataset includes:
        * `year` attribute.
        * The filepaths of the source data are included in a `filepaths` variable.
    """
    # Read all of the complete daily data for the given year and month.
    daily_filepaths = _get_daily_complete_filepaths_for_year(
        year=year,
        ecdr_data_dir=ecdr_data_dir,
        hemisphere=hemisphere,
        resolution=resolution,
    )
    ds = xr.open_mfdataset(daily_filepaths)

    assert np.all([pd.Timestamp(t.values).year == year for t in ds.time])

    # setup global attrs
    # Set global attributes
    daily_aggregate_ds_global_attrs = get_global_attrs(
        time=ds.time,
        temporality="daily",
        aggregate=True,
        source=", ".join([fp.name for fp in daily_filepaths]),
    )
    ds.attrs.update(daily_aggregate_ds_global_attrs)

    logger.info(
        f"Created daily ds for {year=} from {len(ds.time)} complete daily files."
    )

    return ds


def get_daily_aggregate_filepath(
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    sat: SUPPORTED_SAT,
    ecdr_data_dir: Path,
    start_date: dt.date,
    end_date: dt.date,
) -> Path:
    output_dir = ecdr_data_dir / "aggregate"
    output_dir.mkdir(exist_ok=True)

    output_fn = standard_daily_filename(
        hemisphere=hemisphere,
        resolution=resolution,
        sat=sat,
        date=start_date,
        end_date=end_date,
    )

    output_filepath = output_dir / output_fn

    return output_filepath


def make_daily_aggregate_netcdf_for_year(
    *,
    year: int,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ecdr_data_dir: Path,
    sat: SUPPORTED_SAT,
) -> None:
    daily_ds = get_daily_ds_for_year(
        year=year,
        ecdr_data_dir=ecdr_data_dir,
        hemisphere=hemisphere,
        resolution=resolution,
    )

    # TODO: add lat/lon fields

    output_path = get_daily_aggregate_filepath(
        hemisphere=hemisphere,
        resolution=resolution,
        sat=sat,
        start_date=pd.Timestamp(daily_ds.time.min().item()).date(),
        end_date=pd.Timestamp(daily_ds.time.max().item()).date(),
        ecdr_data_dir=ecdr_data_dir,
    )
    daily_ds.to_netcdf(output_path)

    logger.info(f"Wrote daily aggregate file for year={year} to {output_path}")


@click.command(name="daily-aggregate")
@click.option(
    "--year",
    required=True,
    type=int,
    help="Year for which to create the monthly file.",
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
    hemisphere: Hemisphere,
    ecdr_data_dir: Path,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
) -> None:
    # TODO: support different sat/platforms.
    sat: Final = "am2"

    make_daily_aggregate_netcdf_for_year(
        year=year,
        hemisphere=hemisphere,
        resolution=resolution,
        ecdr_data_dir=ecdr_data_dir,
        sat=sat,
    )
