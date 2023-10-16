"""Code to run NRT ECDR processing."""
from pathlib import Path

import click
from pm_tb_data.fetch.lance_amsr2 import download_latest_lance_files

from seaice_ecdr.constants import LANCE_NRT_DATA_DIR


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


if __name__ == "__main__":
    download_latest_nrt_data()
