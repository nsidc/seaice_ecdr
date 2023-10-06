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
from pm_icecon.cdr import amsr2_cdr as pmi_amsr2_cdr

from seaice_ecdr.initial_daily_ecdr import (
    compute_initial_daily_ecdr_dataset as compute_idecdr_ds,
)

# Set the default minimum log notification to Warning
logger.remove(0)  # Removes previous logger info
logger.add(sys.stderr, level='WARNING')


@pytest.fixture(scope='session')
def sample_pmicecon_dataset():
    """Set up sample data set using pm_icecon."""
    logger.info('testing: Creating sample pmicecon dataset')

    test_date = dt.datetime(2021, 4, 5).date()
    test_hemisphere: Final = 'north'
    test_resolution: Final = '12'

    pmicecon_conc_ds = pmi_amsr2_cdr(
        date=test_date,
        hemisphere=test_hemisphere,
        resolution=test_resolution,
    )
    return pmicecon_conc_ds


@pytest.fixture(scope='session')
def sample_idecdr_dataset():
    """Set up sample data set using idecdr."""
    logger.info('testing: Creating sample idecdr dataset')

    test_date = dt.datetime(2021, 4, 5).date()
    test_hemisphere: Final = 'north'
    test_resolution: Final = '12'

    ide_conc_ds = compute_idecdr_ds(
        date=test_date,
        hemisphere=test_hemisphere,  # type: ignore
        resolution=test_resolution,
    )
    return ide_conc_ds


def test_testing_initial_daily_ecdr_generation():
    """Test that this integration test is being performed."""
    assert True


def test_pmicecon_fixture(sample_pmicecon_dataset):
    """Test that pm_icecon yields a 'conc' field."""
    # assert type(sample_pmicecon_dataset) == type(xr.Dataset())
    assert isinstance(sample_pmicecon_dataset, type(xr.Dataset()))


def test_pmicecon_conc_generation(sample_pmicecon_dataset):
    """Test that pm_icecon yields a 'conc' field."""
    # pmicecon_conc_ds = sample_pmicecon_dataset

    pmicecon_conc_varname = 'conc'
    assert (
        type(sample_pmicecon_dataset.variables[pmicecon_conc_varname])
        == xr.core.variable.Variable
    )


def test_seaice_idecdr_and_pmicecon_conc_identical(
    sample_pmicecon_dataset, sample_idecdr_dataset
):
    """Test that pm_icecon yields a 'conc' field."""
    pmicecon_conc_ds = sample_pmicecon_dataset
    pmi_conc_field = np.array(pmicecon_conc_ds.variables['conc'])

    ide_conc_ds = sample_idecdr_dataset
    ide_conc_field = np.squeeze(np.array(ide_conc_ds.variables['conc']))

    # We know that the original conc field has zeros where TBs were not
    # available, so only check where idecdr is not nan
    indexes_to_check = ~np.isnan(ide_conc_field)
    assert_equal(
        pmi_conc_field[indexes_to_check],
        ide_conc_field[indexes_to_check],
    )
