"""Tests for initial daily ECDR generation."""

# TODO: The tests should probably not require "real" data, but
#  should work with mock data.  Or else, they should be moved to
#  tests/integration/ directory.

import datetime as dt
from pathlib import Path
from typing import Final

from pm_tb_data._types import NORTH

from seaice_ecdr.initial_daily_ecdr import get_idecdr_filepath
from seaice_ecdr.platforms import SUPPORTED_PLATFORM_ID
from seaice_ecdr.temporal_composite_daily import (
    make_tiecdr_netcdf,
    read_or_create_and_read_idecdr_ds,
)

date = dt.date(2021, 2, 19)
hemisphere = NORTH
resolution: Final = "25"
platform_id: SUPPORTED_PLATFORM_ID = "F17"
land_spillover_alg: Final = "NT2"
ancillary_source: Final = "CDRv5"


def test_read_or_create_and_read_idecdr_ds(tmpdir):
    """Verify that initial daily ecdr file can be created."""

    sample_ide_filepath = get_idecdr_filepath(
        date=date,
        platform_id=platform_id,
        hemisphere=hemisphere,
        resolution=resolution,
        intermediate_output_dir=Path(tmpdir),
    )

    test_ide_ds_with_creation = read_or_create_and_read_idecdr_ds(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        intermediate_output_dir=Path(tmpdir),
        land_spillover_alg=land_spillover_alg,
        ancillary_source=ancillary_source,
    )

    assert sample_ide_filepath.exists()
    test_ide_ds_with_reading = read_or_create_and_read_idecdr_ds(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        intermediate_output_dir=Path(tmpdir),
        land_spillover_alg=land_spillover_alg,
        ancillary_source=ancillary_source,
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

    # For the test, only interpolate up to two days forward/back in time
    fp = make_tiecdr_netcdf(
        date=test_date,
        hemisphere=hemisphere,
        resolution=resolution,
        intermediate_output_dir=Path(tmpdir),
        interp_range=2,
        land_spillover_alg=land_spillover_alg,
        ancillary_source=ancillary_source,
    )

    assert fp.is_file()
