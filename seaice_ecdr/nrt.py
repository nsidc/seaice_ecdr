"""Code to run NRT ECDR processing."""
import datetime as dt
from pathlib import Path
from typing import get_args

import click
from pm_tb_data.fetch.lance_amsr2 import (
    download_latest_lance_files,
    access_local_lance_data,
)
from pm_tb_data._types import Hemisphere
from pm_tb_data.fetch.au_si import AU_SI_RESOLUTIONS

from seaice_ecdr.constants import LANCE_NRT_DATA_DIR
from seaice_ecdr.initial_daily_ecdr import make_cdr_netcdf
from seaice_ecdr.cli.util import datetime_to_date


@click.command(name="download-latest-nrt-data")
@click.option(
    "-o",
    "--output-dir",
    required=True,
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
    show_default=True,
    default=LANCE_NRT_DATA_DIR,
    help="Directory in which LANCE AMSR2 NRT files will be downloaded to.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    show_default=True,
    default=False,
    help="Overwrite existing LANCE files.",
)
def download_latest_nrt_data(*, output_dir: Path, overwrite: bool) -> None:
    """Download the latest NRT LANCE AMSR2 data to the specified output directory.

    Files are only downloaded if they are considered 'complete' and ready for
    NRT processing for the ECDR product. This means that the latest available
    date of data is never downloaded, as it is considered provisional/subject to
    change until a new day's worth of data is available.
    """
    download_latest_lance_files(output_dir=output_dir, overwrite=overwrite)


@click.command(name="initial-daily-ecdr")
@click.option(
    "-d",
    "--date",
    required=True,
    type=click.DateTime(formats=("%Y-%m-%d",)),
    callback=datetime_to_date,
)
@click.option(
    "-h",
    "--hemisphere",
    required=True,
    type=click.Choice(get_args(Hemisphere)),
)
# TODO: default output-dir for initial daily nrt files.
@click.option(
    "-o",
    "--output-dir",
    required=True,
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.option(
    "-r",
    "--resolution",
    required=True,
    type=click.Choice(get_args(AU_SI_RESOLUTIONS)),
)
@click.option(
    "--lance-amsr2-input-dir",
    required=True,
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
    show_default=True,
    default=LANCE_NRT_DATA_DIR,
    help="Directory in which LANCE AMSR2 NRT files are located.",
)
def nrt_initial_daily_ecdr(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    output_dir: Path,
    resolution: AU_SI_RESOLUTIONS,
    lance_amsr2_input_dir: Path,
):
    """Create an initial daily ECDR NetCDF using NRT LANCE AMSR2 data."""
    xr_tbs = access_local_lance_data(
        date=date,
        hemisphere=hemisphere,
        data_dir=lance_amsr2_input_dir,
    )

    make_cdr_netcdf(
        xr_tbs=xr_tbs,
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        output_dir=output_dir,
    )


@click.group(name="nrt")
def nrt_cli():
    """Run NRT Sea Ice ECDR."""
    ...


nrt_cli.add_command(download_latest_nrt_data)
nrt_cli.add_command(nrt_initial_daily_ecdr)
