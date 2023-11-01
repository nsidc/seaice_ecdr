"""Tests for initial daily ECDR generation."""

# TODO: The tests should probably not require "real" data, but
#  should work with mock data.  Or else, they should be moved to
#  tests/integration/ directory.

import datetime as dt
import sys
from pathlib import Path
from typing import Final

from loguru import logger
from pm_tb_data._types import NORTH

from seaice_ecdr.initial_daily_ecdr import (
    initial_daily_ecdr_dataset_for_au_si_tbs,
    write_ide_netcdf,
)
from seaice_ecdr.temporal_composite_daily import (
    get_standard_initial_daily_ecdr_filename,
    make_tiecdr_netcdf,
    read_with_create_initial_daily_ecdr,
)

# Set the default minimum log notification to Warning
try:
    logger.remove(0)  # Removes previous logger info
    logger.add(sys.stderr, level="WARNING")
except ValueError:
    logger.debug(f"Started logging in {__name__}")
    logger.add(sys.stderr, level="WARNING")


date = dt.date(2021, 2, 19)
hemisphere = NORTH
resolution: Final = "12"


def test_create_ide_file(tmpdir):
    """Verify that initial daily ecdr file can be created."""
    sample_ide_filepath = get_standard_initial_daily_ecdr_filename(
        date, hemisphere, resolution, output_directory=Path(tmpdir)
    )

    test_ide_ds = initial_daily_ecdr_dataset_for_au_si_tbs(
        date=date, hemisphere=hemisphere, resolution=resolution
    )
    written_path = write_ide_netcdf(
        ide_ds=test_ide_ds,
        output_filepath=sample_ide_filepath,
    )
    assert sample_ide_filepath == written_path
    assert sample_ide_filepath.exists()


def test_read_with_create_ide_file(tmpdir):
    """Verify that initial daily ecdr file can be created."""
    sample_ide_filepath = get_standard_initial_daily_ecdr_filename(
        date, hemisphere, resolution, output_directory=Path(tmpdir)
    )

    test_ide_ds_with_creation = read_with_create_initial_daily_ecdr(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        ide_dir=Path(tmpdir),
    )

    assert sample_ide_filepath.exists()
    test_ide_ds_with_reading = read_with_create_initial_daily_ecdr(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        ide_dir=Path(tmpdir),
    )

    assert test_ide_ds_with_creation == test_ide_ds_with_reading


def test_create_tiecdr_file(tmpdir):
    """Verify creation of a "pass 2" tiecdr file.

    tiecdr is short for temporally integrated eCDR file

    Only a few dates between July 2012 and Oct 2023 have enough
    missing NH data in AU_SI12 to make testing with actual
    data possible.
    """
    test_date = dt.date(2022, 3, 2)
    tmpdir_path = Path(tmpdir)
    initial_daily_dir = tmpdir_path / "initial"
    initial_daily_dir.mkdir()
    temporal_daily_dir = tmpdir_path / "temporal"
    temporal_daily_dir.mkdir()

    # For the test, only interpolate up to two days forward/back in time
    make_tiecdr_netcdf(
        date=test_date,
        hemisphere=hemisphere,
        resolution=resolution,
        output_dir=temporal_daily_dir,
        interp_range=2,
        ide_dir=initial_daily_dir,
    )
