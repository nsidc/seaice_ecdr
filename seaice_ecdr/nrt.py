"""Code to run NRT ECDR processing."""
import datetime as dt
from pathlib import Path
from typing import get_args

import click
from pm_tb_data._types import Hemisphere
from pm_tb_data.fetch.amsr.lance_amsr2 import (
    access_local_lance_data,
    download_latest_lance_files,
)

from seaice_ecdr._types import ECDR_SUPPORTED_RESOLUTIONS
from seaice_ecdr.cli.util import datetime_to_date
from seaice_ecdr.constants import LANCE_NRT_DATA_DIR, NRT_BASE_OUTPUT_DIR
from seaice_ecdr.initial_daily_ecdr import (
    compute_initial_daily_ecdr_dataset,
    get_idecdr_filepath,
    write_ide_netcdf,
)
from seaice_ecdr.platforms import (
    get_platform_by_date,
)
from seaice_ecdr.tb_data import EcdrTbData


def compute_nrt_initial_daily_ecdr_dataset(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    lance_amsr2_input_dir: Path,
):
    """Create an initial daily ECDR NetCDF using NRT LANCE AMSR2 data."""
    xr_tbs = access_local_lance_data(
        date=date,
        hemisphere=hemisphere,
        data_dir=lance_amsr2_input_dir,
    )

    tb_data = EcdrTbData(
        tbs=xr_tbs,
        resolution="12.5",
        data_source="LANCE AU_SI12",
        platform="am2",
    )

    nrt_initial_ecdr_ds = compute_initial_daily_ecdr_dataset(
        date=date,
        hemisphere=hemisphere,
        tb_data=tb_data,
    )

    return nrt_initial_ecdr_ds


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


@click.command(name="nrt-initial-daily-ecdr")
@click.option(
    "-d",
    "--date",
    required=True,
    type=click.DateTime(formats=("%Y-%m-%d", "%Y%m%d", "%Y.%m.%d")),
    callback=datetime_to_date,
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
    default=NRT_BASE_OUTPUT_DIR,
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
    ecdr_data_dir: Path,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    lance_amsr2_input_dir: Path,
):
    """Create an initial daily ECDR NetCDF using NRT LANCE AMSR2 data.

    TODO: Consider renaming this: nrt_initial_daily_ecdr_netcdf()
    """
    nrt_initial_ecdr_ds = compute_nrt_initial_daily_ecdr_dataset(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        lance_amsr2_input_dir=lance_amsr2_input_dir,
    )

    platform = get_platform_by_date(date)
    output_path = get_idecdr_filepath(
        hemisphere=hemisphere,
        date=date,
        platform=platform,
        resolution=resolution,
        ecdr_data_dir=ecdr_data_dir,
    )

    write_ide_netcdf(
        ide_ds=nrt_initial_ecdr_ds,
        output_filepath=output_path,
    )


@click.group(name="nrt")
def nrt_cli():
    """Run NRT Sea Ice ECDR."""
    ...


nrt_cli.add_command(download_latest_nrt_data)
nrt_cli.add_command(nrt_initial_daily_ecdr)
