"""
test_initial_daily_ecdr_generation.py

Verify that the initial daily ecdr files are the same as pm_icecon's initial
amsr2_cdr results
"""

import datetime as dt
from pathlib import Path

import numpy as np
import xarray as xr
from numpy.testing import assert_almost_equal, assert_equal
from numpy.typing import NDArray

from pm_icecon.cdr import amsr2_cdr as pmi_amsr2_cdr
#from seaice_ecdr.initial_daily_ecdr import amsr2_cdr as ide_amsr2_cdr
from seaice_ecdr.initial_daily_ecdr import compute_initial_daily_ecdr_dataset as compute_idecdr_ds


""" None of this seems to work
# Ignore warnings about numpy header size changes
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
warnings.filterwarnings("ignore", message="numpy.ndarray size changed")
warnings.filterwarnings("ignore", message="numpy.ndarray size changed, may indicate binary incompatibility. Expected 16 from C header, got 96 from PyObject")
"""

def test_testing_initial_daily_ecdr_generation():
    """ Test that this integration test is being performed """
    assert True


def test_pmicecon_conc_generation():
    """ Test that pm_icecon yields a 'conc' field """
    test_date = dt.datetime(2021, 4, 5).date()
    test_hemisphere = 'north'
    test_resolution = '12'

    pmicecon_conc_ds = pmi_amsr2_cdr(
        date=test_date,
        hemisphere=test_hemisphere,
        resolution=test_resolution,
    )
    assert type(pmicecon_conc_ds) == type(xr.Dataset())

    pmicecon_conc_varname = 'conc'
    assert type(pmicecon_conc_ds.variables[pmicecon_conc_varname]) == xr.core.variable.Variable


def test_seaice_idecdr_and_pmicecon_conc_identical():
    """ Test that pm_icecon yields a 'conc' field """
    test_date = dt.datetime(2021, 4, 5).date()
    test_hemisphere = 'north'
    test_resolution = '12'

    pmicecon_conc_ds = pmi_amsr2_cdr(
        date=test_date,
        hemisphere=test_hemisphere,
        resolution=test_resolution,
    )
    pmi_conc_field = np.array(pmicecon_conc_ds.variables['conc'])

    ide_conc_ds = compute_idecdr_ds(
        date=test_date,
        hemisphere=test_hemisphere,
        resolution=test_resolution,
    )
    ide_conc_field = np.array(ide_conc_ds.variables['conc'])

    assert_equal(pmi_conc_field, ide_conc_field)
