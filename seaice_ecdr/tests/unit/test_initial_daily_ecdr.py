"""Tests of the initial daily ECDR product.

test_initial_daily_ecdr.py
"""

import datetime as dt

import pytest
import xarray as xr
from loguru import logger

from seaice_ecdr.initial_daily_ecdr import (
    compute_initial_daily_ecdr_dataset as compute_idecdr_ds,
)


@pytest.fixture(scope='session')
def sample_idecdr_dataset_nh():
    """Set up the sample NH initial daily ecdr data set."""
    logger.info('testing: Creating sample idecdr dataset')

    test_date = dt.datetime(2021, 4, 5).date()
    test_hemisphere = 'north'
    test_resolution = '12'

    ide_conc_ds = compute_idecdr_ds(
        date=test_date,
        hemisphere=test_hemisphere,
        resolution=test_resolution,
    )
    return ide_conc_ds


@pytest.fixture(scope='session')
def sample_idecdr_dataset_sh():
    """Set up the sample SH initial daily ecdr data set."""
    logger.info('testing: Creating sample idecdr dataset')

    test_date = dt.datetime(2021, 4, 5).date()
    test_hemisphere = 'south'
    test_resolution = '12'

    ide_conc_ds = compute_idecdr_ds(
        date=test_date,
        hemisphere=test_hemisphere,
        resolution=test_resolution,
    )
    return ide_conc_ds


def test_seaice_idecdr_is_Dataset(
    sample_idecdr_dataset_nh,
    sample_idecdr_dataset_sh,
):
    """Test that pm_icecon yields a 'conc' field."""
    # assert type(sample_idecdr_dataset_nh) == type(xr.Dataset())
    # assert type(sample_idecdr_dataset_sh) == type(xr.Dataset())
    assert isinstance(sample_idecdr_dataset_nh, xr.Dataset())
    assert isinstance(sample_idecdr_dataset_sh, xr.Dataset())


def test_seaice_idecdr_has_crs(
    sample_idecdr_dataset_nh,
    sample_idecdr_dataset_sh,
):
    """Test that pm_icecon yields a 'conc' field."""
    assert 'crs' in sample_idecdr_dataset_nh.variables
    assert 'crs' in sample_idecdr_dataset_sh.variables


def test_seaice_idecdr_can_output_to_netcdf(
    sample_idecdr_dataset_nh,
    sample_idecdr_dataset_sh,
):
    """Test that xarray dataset can be saved to a netCDF file."""
    import os

    # NH
    sample_output_filename_nh = './sample_ecdr_nh.nc'
    sample_idecdr_dataset_nh.to_netcdf(sample_output_filename_nh)
    assert os.path.isfile(sample_output_filename_nh)

    # SH
    sample_output_filename_sh = './sample_ecdr_sh.nc'
    sample_idecdr_dataset_sh.to_netcdf(sample_output_filename_sh)
    assert os.path.isfile(sample_output_filename_sh)
