"""Verify no change in initial amsr2_cdr data sets.

test_initial_daily_ecdr_generation.py

Verify that the initial daily ecdr files are the same as
pm_icecon's initial amsr2_cdr results.

"""

import datetime as dt
import sys
from typing import Final

import numpy as np
import pytest
import xarray as xr
from loguru import logger
from numpy.testing import assert_equal

from seaice_ecdr.cdr_alg import amsr2_cdr as pmi_amsr2_cdr
from seaice_ecdr.initial_daily_ecdr import (
    compute_initial_daily_ecdr_dataset as compute_idecdr_ds,
)

# Set the default minimum log notification to Warning
logger.remove(0)  # Removes previous logger info
logger.add(sys.stderr, level="WARNING")


@pytest.fixture(scope="session")
def sample_idecdr_dataset_nh():
    """Set up the sample NH initial daily ecdr data set."""
    logger.info("testing: Creating sample idecdr dataset")

    test_date = dt.datetime(2021, 4, 5).date()
    test_hemisphere: Final = "north"
    test_resolution: Final = "12"

    ide_conc_ds = compute_idecdr_ds(
        date=test_date,
        hemisphere=test_hemisphere,
        resolution=test_resolution,
    )
    return ide_conc_ds


@pytest.fixture(scope="session")
def sample_idecdr_dataset_sh():
    """Set up the sample SH initial daily ecdr data set."""
    logger.info("testing: Creating sample idecdr dataset")

    test_date = dt.datetime(2021, 4, 5).date()
    test_hemisphere: Final = "north"
    test_resolution: Final = "12"

    ide_conc_ds = compute_idecdr_ds(
        date=test_date,
        hemisphere=test_hemisphere,
        resolution=test_resolution,
    )
    return ide_conc_ds


@pytest.fixture(scope="session")
def sample_pmicecon_dataset():
    """Set up sample data set using pm_icecon."""
    logger.info("testing: Creating sample pmicecon dataset")

    test_date = dt.datetime(2021, 4, 5).date()
    test_hemisphere: Final = "north"
    test_resolution: Final = "12"

    pmicecon_conc_ds = pmi_amsr2_cdr(
        date=test_date,
        hemisphere=test_hemisphere,
        resolution=test_resolution,
    )
    return pmicecon_conc_ds


@pytest.fixture(scope="session")
def sample_idecdr_dataset():
    """Set up sample data set using idecdr."""
    logger.info("testing: Creating sample idecdr dataset")

    test_date = dt.datetime(2021, 4, 5).date()
    test_hemisphere: Final = "north"
    test_resolution: Final = "12"

    ide_conc_ds = compute_idecdr_ds(
        date=test_date,
        hemisphere=test_hemisphere,
        resolution=test_resolution,
    )
    return ide_conc_ds


def test_seaice_idecdr_and_pmicecon_conc_identical(
    sample_pmicecon_dataset, sample_idecdr_dataset
):
    """Test that pm_icecon yields a 'conc' field."""
    pmicecon_conc_ds = sample_pmicecon_dataset
    pmi_conc_field = np.array(pmicecon_conc_ds.variables["conc"])

    ide_conc_ds = sample_idecdr_dataset
    ide_conc_field = np.squeeze(np.array(ide_conc_ds.variables["conc"]))

    # We know that the original conc field has zeros where TBs were not
    # available, so only check where idecdr is not nan
    indexes_to_check = ~np.isnan(ide_conc_field)
    assert_equal(
        pmi_conc_field[indexes_to_check],
        ide_conc_field[indexes_to_check],
    )


def test_seaice_idecdr_can_output_to_netcdf(
    sample_idecdr_dataset_nh,
    sample_idecdr_dataset_sh,
):
    """Test that xarray dataset can be saved to a netCDF file."""
    import os

    # NH
    sample_output_filename_nh = "./sample_ecdr_nh.nc"
    sample_idecdr_dataset_nh.to_netcdf(sample_output_filename_nh)
    assert os.path.isfile(sample_output_filename_nh)

    # SH
    sample_output_filename_sh = "./sample_ecdr_sh.nc"
    sample_idecdr_dataset_sh.to_netcdf(sample_output_filename_sh)
    assert os.path.isfile(sample_output_filename_sh)


def test_seaice_idecdr_has_crs(
    sample_idecdr_dataset_nh,
    sample_idecdr_dataset_sh,
):
    """Test that pm_icecon yields a 'conc' field."""
    assert "crs" in sample_idecdr_dataset_nh.variables
    assert "crs" in sample_idecdr_dataset_sh.variables


def test_seaice_idecdr_is_Dataset(
    sample_idecdr_dataset_nh,
    sample_idecdr_dataset_sh,
):
    """Test that pm_icecon yields a 'conc' field."""
    # assert type(sample_idecdr_dataset_nh) == type(xr.Dataset())
    # assert type(sample_idecdr_dataset_sh) == type(xr.Dataset())
    assert isinstance(sample_idecdr_dataset_nh, type(xr.Dataset()))
    assert isinstance(sample_idecdr_dataset_sh, type(xr.Dataset()))
