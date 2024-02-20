"""Script for creating archive of code and ancillary files for NOAA

NOAA requires a copy of our code and ancillary data each time we do a data
release, to archive alongside the output data.

This script creates a .zip file containing a copy of the `seaice_ecdr` git
repository and the seaice_ecdr.constants.CDR_ANCILLARY_DIR ancillary dir.
"""

import datetime as dt
import shutil
import subprocess
import tempfile
from pathlib import Path

import click
from loguru import logger

from seaice_ecdr.constants import CDR_ANCILLARY_DIR


def clone_repo(
    *,
    clone_address: str,
    clone_dir: Path,
    ref: str,
    project_name="seaice_ecdr",
):
    result = subprocess.run(
        [
            # Clone the repo from the source. This avoids copying any
            # non-tracked files from the user's local copy.
            "git",
            "clone",
            clone_address,
        ],
        cwd=clone_dir,
    )

    result.check_returncode()

    result = subprocess.run(
        [
            "git",
            "checkout",
            ref,
        ],
        cwd=clone_dir / project_name,
    )

    result.check_returncode()


def make_archive_for_noaa(
    *, output_dir: Path, seaice_ecdr_ref: str, pm_icecon_ref: str, pm_tb_data_ref
):
    with tempfile.TemporaryDirectory() as tempdir:
        archive_name = f"seaice_ecdr_{dt.date.today():%Y%m%d}"
        archive_dir = Path(tempdir) / archive_name
        archive_dir.mkdir()
        clone_repo(
            clone_dir=archive_dir,
            clone_address="git@github.com:nsidc/seaice_ecdr.git",
            ref=seaice_ecdr_ref,
            project_name="seaice_ecdr",
        )
        clone_repo(
            clone_dir=archive_dir,
            clone_address="git@github.com:nsidc/pm_icecon.git",
            ref=pm_icecon_ref,
            project_name="pm_icecon",
        )
        clone_repo(
            clone_dir=archive_dir,
            clone_address="git@github.com:nsidc/pm_tb_data.git",
            ref=pm_tb_data_ref,
            project_name="pm_tb_data",
        )

        # Copy the ancillary data over
        shutil.copytree(CDR_ANCILLARY_DIR, Path(archive_dir) / CDR_ANCILLARY_DIR.name)

        # Now create a zip archive
        base_name_with_dir = str(output_dir / archive_name)
        output_fp = shutil.make_archive(
            base_name=base_name_with_dir, format="zip", root_dir=tempdir
        )
        logger.info(f"Wrote {output_fp}")


@click.command(
    name="create-noaa-archive",
    help="Create a .zip archive of the repository and ancillary data directory for NOAA.",
)
@click.option(
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
    help=(
        "Base output directory for standard ECDR outputs."
        " Subdirectories are created for outputs of"
        " different stages of processing."
    ),
    show_default=True,
)
@click.option(
    "--seaice-ecdr-ref",
    required=True,
    type=str,
    help="Ref of this code (`seaice_ecdr`) to include in the archive.",
)
@click.option(
    "--pm-icecon-ref",
    required=True,
    type=str,
    help="Ref of `pm_icecon` to include in the archive.",
)
@click.option(
    "--pm-tb-data-ref",
    required=True,
    type=str,
    help="Ref of `pm_tb_data` to include in the archive.",
)
def cli(
    output_dir: Path, seaice_ecdr_ref: str, pm_icecon_ref: str, pm_tb_data_ref: str
):
    make_archive_for_noaa(
        output_dir=output_dir,
        seaice_ecdr_ref=seaice_ecdr_ref,
        pm_icecon_ref=pm_icecon_ref,
        pm_tb_data_ref=pm_tb_data_ref,
    )


if __name__ == "__main__":
    cli()
