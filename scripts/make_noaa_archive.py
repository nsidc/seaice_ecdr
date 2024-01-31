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


def clone_repo(*, clone_dir: Path, ref: str):
    result = subprocess.run(
        [
            # Clone the repo from the source. This avoids copying any
            # non-tracked files from the user's local copy.
            "git",
            "clone",
            "git@github.com:nsidc/seaice_ecdr.git",
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
        cwd=clone_dir / "seaice_ecdr",
    )

    result.check_returncode()


def make_archive_for_noaa(*, output_dir: Path, ref: str = "main"):
    with tempfile.TemporaryDirectory() as tempdir:
        archive_name = f"archive_{dt.date.today():%Y%m%d}"
        archive_dir = Path(tempdir) / archive_name
        archive_dir.mkdir()
        clone_repo(clone_dir=archive_dir, ref=ref)

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
def cli(output_dir: Path, seaice_ecdr_ref: str):
    make_archive_for_noaa(output_dir=output_dir, ref=seaice_ecdr_ref)


if __name__ == "__main__":
    cli()
