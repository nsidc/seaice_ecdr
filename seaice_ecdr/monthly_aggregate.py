"""Code to produce monthly aggregate files from monthly complete data.
"""
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
from seaice_ecdr.constants import STANDARD_BASE_OUTPUT_DIR
from seaice_ecdr.monthly import get_monthly_dir
from seaice_ecdr.nc_attrs import get_global_attrs
from seaice_ecdr.nc_util import concatenate_nc_files
from seaice_ecdr.util import sat_from_filename, standard_monthly_aggregate_filename


def _get_monthly_complete_filepaths(
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ecdr_data_dir: Path,
) -> list[Path]:
    monthly_dir = get_monthly_dir(
        ecdr_data_dir=ecdr_data_dir,
    )

    # TODO: the monthly filenames are encoded in the
    # `util.standard_monthly_filename` func. Can we adapt that to use wildcards
    # and return a glob-able string?
    # North Monthly files: sic_psn12.5_YYYYMM_sat_v05r00.nc
    # South Monthly files: sic_pss12.5_YYYYMM_sat_v05r00.nc
    filename_pattern = f"sic_ps{hemisphere[0]}{resolution}_*.nc"

    monthly_files = list(sorted(monthly_dir.glob(filename_pattern)))

    return monthly_files


def get_monthly_aggregate_filepath(
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
    ecdr_data_dir: Path,
) -> Path:
    output_dir = ecdr_data_dir / "aggregate"
    output_dir.mkdir(exist_ok=True)

    output_fn = standard_monthly_aggregate_filename(
        hemisphere=hemisphere,
        resolution=resolution,
        start_year=start_year,
        start_month=start_month,
        end_year=end_year,
        end_month=end_month,
    )

    output_filepath = output_dir / output_fn

    return output_filepath


def _update_ncrcat_monthly_ds(
    *,
    agg_ds: xr.Dataset,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    monthly_filepaths: list[Path],
) -> xr.Dataset:
    # Add latitude and longitude fields
    surf_geo_ds = get_ancillary_ds(
        hemisphere=hemisphere,
        resolution=resolution,
    )
    agg_ds["latitude"] = surf_geo_ds.latitude
    agg_ds["longitude"] = surf_geo_ds.latitude

    # setup global attrs
    # Set global attributes
    monthly_aggregate_ds_global_attrs = get_global_attrs(
        time=agg_ds.time,
        temporality="monthly",
        aggregate=True,
        source=", ".join([fp.name for fp in monthly_filepaths]),
        sats=[sat_from_filename(fp.name) for fp in monthly_filepaths],
    )
    agg_ds.attrs = monthly_aggregate_ds_global_attrs

    return agg_ds


@click.command(name="monthly-aggregate")
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
    hemisphere: Hemisphere,
    ecdr_data_dir: Path,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
) -> None:
    # ds = get_monthly_aggregate_ds(
    #     hemisphere=hemisphere,
    #     resolution=resolution,
    #     ecdr_data_dir=ecdr_data_dir,
    # )

    monthly_filepaths = _get_monthly_complete_filepaths(
        hemisphere=hemisphere,
        ecdr_data_dir=ecdr_data_dir,
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
            input_filepaths=monthly_filepaths,
            output_filepath=tmp_output_fp,
        )
        ds = xr.open_dataset(tmp_output_fp, chunks=dict(time=1))
        ds = _update_ncrcat_monthly_ds(
            agg_ds=ds,
            hemisphere=hemisphere,
            resolution=resolution,
            monthly_filepaths=monthly_filepaths,
        )

        # TODO: better way to extract start/end date? This same code is used above
        # in `get_monthly_aggregate_ds`.
        start_date = pd.Timestamp(ds.time.min().values).date()
        end_date = pd.Timestamp(ds.time.max().values).date()

        output_filepath = get_monthly_aggregate_filepath(
            hemisphere=hemisphere,
            resolution=resolution,
            start_year=start_date.year,
            start_month=start_date.month,
            end_year=end_date.year,
            end_month=end_date.month,
            ecdr_data_dir=ecdr_data_dir,
        )
        ds.to_netcdf(output_filepath)

    logger.info(f"Wrote monthly aggregate file to {output_filepath}")

    # Cleanup previously existing monthly aggregates.
    existing_fn_pattern = f"sic_ps{hemisphere[0]}{resolution}_??????-??????_*.nc"
    existing_filepaths = list((ecdr_data_dir / "aggregate").glob(existing_fn_pattern))
    for existing_filepath in existing_filepaths:
        if existing_filepath != output_filepath:
            existing_filepath.unlink()
            logger.info(f"Removed old monthly aggregate file {existing_filepath}")
