"""Code to produce daily aggregate files from daily complete data.
"""

import datetime as dt
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import get_args

import click
import pandas as pd
import xarray as xr
from loguru import logger
from pm_tb_data._types import Hemisphere

from seaice_ecdr._types import ECDR_SUPPORTED_RESOLUTIONS
from seaice_ecdr.ancillary import get_ancillary_ds
from seaice_ecdr.checksum import write_checksum_file
from seaice_ecdr.complete_daily_ecdr import get_ecdr_filepath
from seaice_ecdr.constants import STANDARD_BASE_OUTPUT_DIR
from seaice_ecdr.nc_attrs import get_global_attrs
from seaice_ecdr.nc_util import concatenate_nc_files
from seaice_ecdr.platforms import get_first_platform_start_date
from seaice_ecdr.util import (
    sat_from_filename,
    standard_daily_aggregate_filename,
)


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
    start_date = max(dt.date(year, 1, 1), get_first_platform_start_date())
    for period in pd.period_range(start=start_date, end=dt.date(year, 12, 31)):
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


def get_daily_aggregate_filepath(
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ecdr_data_dir: Path,
    start_date: dt.date,
    end_date: dt.date,
) -> Path:
    output_dir = ecdr_data_dir / "aggregate"
    output_dir.mkdir(exist_ok=True)

    output_fn = standard_daily_aggregate_filename(
        hemisphere=hemisphere,
        resolution=resolution,
        start_date=start_date,
        end_date=end_date,
    )

    output_filepath = output_dir / output_fn

    return output_filepath


def _update_ncrcat_daily_ds(
    *,
    ds: xr.Dataset,
    daily_filepaths: list[Path],
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
):
    """Update the aggregate dataset created by `ncrcat`.

    Adds lat/lon fields and sets global attrs.
    """
    surf_geo_ds = get_ancillary_ds(
        hemisphere=hemisphere,
        resolution=resolution,
    )
    ds["latitude"] = surf_geo_ds.latitude
    ds["longitude"] = surf_geo_ds.longitude

    # Remove the "number of missing pixels" attr from the daily aggregate conc
    # variable.
    ds["cdr_seaice_conc"].attrs = {
        k: v
        for k, v in ds["cdr_seaice_conc"].attrs.items()
        if k != "number_of_missing_pixels"
    }

    # setup global attrs
    # Set global attributes
    daily_aggregate_ds_global_attrs = get_global_attrs(
        time=ds.time,
        temporality="daily",
        aggregate=True,
        source=", ".join([fp.name for fp in daily_filepaths]),
        sats=[sat_from_filename(fp.name) for fp in daily_filepaths],
    )
    ds.attrs = daily_aggregate_ds_global_attrs

    return ds


def make_daily_aggregate_netcdf_for_year(
    *,
    year: int,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ecdr_data_dir: Path,
) -> None:
    try:
        daily_filepaths = _get_daily_complete_filepaths_for_year(
            year=year,
            ecdr_data_dir=ecdr_data_dir,
            hemisphere=hemisphere,
            resolution=resolution,
        )

        # Create a temporary dir to store a WIP netcdf file. We do this because
        # using the `ncrcat` CLI tool (included with `nco` dep) is much faster than
        # xarray at concatenating the data. Then we do some modifications (e.g.,
        # adding `latitude` and `longitude`, global attrs, etc.) before saving the
        # data in it's final location.
        with TemporaryDirectory() as tmpdir:
            tmp_output_fp = Path(tmpdir) / "temp.nc"
            concatenate_nc_files(
                input_filepaths=daily_filepaths,
                output_filepath=tmp_output_fp,
            )
            daily_ds = xr.open_dataset(tmp_output_fp, chunks=dict(time=1))
            daily_ds = _update_ncrcat_daily_ds(
                ds=daily_ds,
                daily_filepaths=daily_filepaths,
                hemisphere=hemisphere,
                resolution=resolution,
            )

            output_path = get_daily_aggregate_filepath(
                hemisphere=hemisphere,
                resolution=resolution,
                start_date=pd.Timestamp(daily_ds.time.min().item()).date(),
                end_date=pd.Timestamp(daily_ds.time.max().item()).date(),
                ecdr_data_dir=ecdr_data_dir,
            )

            daily_ds.to_netcdf(
                output_path,
                unlimited_dims=[
                    "time",
                ],
            )

        logger.info(f"Wrote daily aggregate file for year={year} to {output_path}")

        # Write checksum file for the aggregate daily output.
        write_checksum_file(
            input_filepath=output_path,
            ecdr_data_dir=ecdr_data_dir,
            product_type="aggregate",
        )
    except Exception as e:
        logger.exception(f"Failed to create daily aggregate for {year=} {hemisphere=}")
        raise e


@click.command(name="daily-aggregate")
@click.option(
    "--year",
    required=True,
    type=int,
    help="Year for which to create the daily-aggregate file.",
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
@click.option(
    "--end-year",
    required=False,
    type=int,
    help="Last year for which to create the daily-aggregate file.",
    default=None,
)
def cli(
    *,
    year: int,
    hemisphere: Hemisphere,
    ecdr_data_dir: Path,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    end_year: int | None,
) -> None:
    if end_year is None:
        end_year = year

    failed_years = []
    for year_to_process in range(year, end_year + 1):
        try:
            make_daily_aggregate_netcdf_for_year(
                year=year_to_process,
                hemisphere=hemisphere,
                resolution=resolution,
                ecdr_data_dir=ecdr_data_dir,
            )
        except Exception:
            failed_years.append(year_to_process)

    if failed_years:
        str_formatted_years = "\n".join(str(year) for year in failed_years)
        raise RuntimeError(
            f"Encountered {len(failed_years)} failures."
            f" Daily aggregates for the following years were not created:\n{str_formatted_years}"
        )
