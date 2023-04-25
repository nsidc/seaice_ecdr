"""
test_initial_daily_ecdr.py

Tests of the initial daily ECDR product
"""

import datetime as dt
from pathlib import Path
import pytest
from loguru import logger

import numpy as np
import xarray as xr
from numpy.testing import assert_almost_equal, assert_equal
from numpy.typing import NDArray

from pm_icecon.cdr import amsr2_cdr as pmi_amsr2_cdr
from seaice_ecdr.initial_daily_ecdr import compute_initial_daily_ecdr_dataset as compute_idecdr_ds


@pytest.fixture(scope='session')
def sample_idecdr_dataset():
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


def test_seaice_idecdr_is_Dataset(sample_idecdr_dataset):
    """ Test that pm_icecon yields a 'conc' field """
    assert type(sample_idecdr_dataset) == type(xr.Dataset())
