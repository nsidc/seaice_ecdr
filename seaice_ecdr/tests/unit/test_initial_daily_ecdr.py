"""
test_initial_daily_ecdr.py

Tests of the initial daily ECDR product
"""

import datetime as dt
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from loguru import logger
from numpy.testing import assert_almost_equal, assert_equal
from numpy.typing import NDArray
from pm_icecon.cdr import amsr2_cdr as pmi_amsr2_cdr

from seaice_ecdr.initial_daily_ecdr import (
    compute_initial_daily_ecdr_dataset as compute_idecdr_ds,
)


@pytest.fixture(scope='session')
def sample_idecdr_dataset_nh():
    logger.info(f'testing: Creating sample idecdr dataset')

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
    logger.info(f'testing: Creating sample idecdr dataset')

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
    """ Test that pm_icecon yields a 'conc' field """
    assert type(sample_idecdr_dataset_nh) == type(xr.Dataset())
    assert type(sample_idecdr_dataset_sh) == type(xr.Dataset())


def test_seaice_idecdr_has_crs(
    sample_idecdr_dataset_nh,
    sample_idecdr_dataset_sh,
):
    """ Test that pm_icecon yields a 'conc' field """
    assert 'crs' in sample_idecdr_dataset_nh.variables
    assert 'crs' in sample_idecdr_dataset_sh.variables


def test_seaice_idecdr_can_output_to_netcdf(
    sample_idecdr_dataset_nh,
    sample_idecdr_dataset_sh,
):
    """ Test that xarray dataset can be saved to a netCDF file """
    import os

    # NH
    sample_output_filename_nh = './sample_ecdr_nh.nc'
    sample_idecdr_dataset_nh.to_netcdf(sample_output_filename_nh)
    assert os.path.isfile(sample_output_filename_nh)

    # SH
    sample_output_filename_sh = './sample_ecdr_sh.nc'
    sample_idecdr_dataset_sh.to_netcdf(sample_output_filename_sh)
    assert os.path.isfile(sample_output_filename_sh)
