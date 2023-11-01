"""Tests for initial daily ECDR generation."""

import datetime as dt
import sys
from typing import Final
from pathlib import Path

import pytest
import xarray as xr
from loguru import logger

from seaice_ecdr.initial_daily_ecdr import (
    initial_daily_ecdr_dataset_for_au_si_tbs as compute_idecdr_ds,
    make_idecdr_netcdf,
    get_idecdr_filepath,
)

from pm_tb_data._types import NORTH

# Set the default minimum log notification to Warning
logger.remove(0)  # Removes previous logger info
logger.add(sys.stderr, level="WARNING")

cdr_conc_fieldname = "conc"


@pytest.fixture(scope="session")
def sample_idecdr_dataset_nh():
    """Set up the sample NH initial daily ecdr data set."""
    logger.info("testing: Creating sample idecdr dataset")

    test_date = dt.datetime(2021, 4, 5).date()
    test_hemisphere = NORTH
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
    test_hemisphere = NORTH
    test_resolution: Final = "12"

    ide_conc_ds = compute_idecdr_ds(
        date=test_date,
        hemisphere=test_hemisphere,
        resolution=test_resolution,
    )
    return ide_conc_ds


def test_seaice_idecdr_can_output_to_netcdf(
    sample_idecdr_dataset_nh,
    sample_idecdr_dataset_sh,
    tmp_path,
):
    """Test that xarray dataset can be saved to a netCDF file."""

    # NH
    sample_output_filepath_nh = tmp_path / "sample_idecdr_nh.nc"
    sample_idecdr_dataset_nh.to_netcdf(sample_output_filepath_nh)
    assert sample_output_filepath_nh.is_file()

    # SH
    sample_output_filepath_sh = tmp_path / "sample_idecdr_sh.nc"
    sample_idecdr_dataset_sh.to_netcdf(sample_output_filepath_sh)
    assert sample_output_filepath_sh.is_file()


def test_seaice_idecdr_is_Dataset(
    sample_idecdr_dataset_nh,
    sample_idecdr_dataset_sh,
):
    """Test that idecdr is an xarray Dataset."""
    assert isinstance(sample_idecdr_dataset_nh, type(xr.Dataset()))
    assert isinstance(sample_idecdr_dataset_sh, type(xr.Dataset()))


def test_seaice_idecdr_has_crs(
    sample_idecdr_dataset_nh,
    sample_idecdr_dataset_sh,
):
    """Test that idecdr contains a 'conc' field."""
    assert "crs" in sample_idecdr_dataset_nh.variables
    assert "crs" in sample_idecdr_dataset_sh.variables


def test_seaice_idecdr_has_necessary_fields(
    sample_idecdr_dataset_nh,
    sample_idecdr_dataset_sh,
):
    """Test that idecdr netcdf has necessary data fields."""
    # TODO: the 'conc' var should become cdr_conc or cdr_conc_raw
    expected_fields = (
        "crs",
        "x",
        "y",
        "y",
        "conc",
        "qa_of_cdr_seaice_conc",
        "bt_conc_raw",
        "nt_conc_raw",
        "bt_weather_mask",
        "nt_weather_mask",
        "invalid_ice_mask",
        "spatint_bitmask",
    )
    for field_name in expected_fields:
        assert field_name in sample_idecdr_dataset_nh.variables.keys()
        assert field_name in sample_idecdr_dataset_sh.variables.keys()


def test_cli_idecdr_ncfile_creation(tmpdir):
    """Verify that code used in cli.sh creates netCDF file."""
    tmpdir_path = Path(tmpdir)
    test_date = dt.datetime(2021, 4, 5).date()
    test_hemisphere = NORTH
    test_resolution: Final = "12"
    make_idecdr_netcdf(
        date=test_date,
        hemisphere=test_hemisphere,
        resolution=test_resolution,
        cdr_data_dir=tmpdir_path,
    )
    output_path = get_idecdr_filepath(
        hemisphere=test_hemisphere,
        date=test_date,
        resolution=test_resolution,
        cdr_data_dir=tmpdir_path,
    )

    assert output_path.is_file()

    ds = xr.open_dataset(output_path)
    assert cdr_conc_fieldname in ds.variables.keys()


def test_can_drop_fields_from_idecdr_netcdf(
    sample_idecdr_dataset_nh,
    tmpdir,
):
    """Verify that specified fields can be excluded in idecdr nc files"""
    tmpdir_path = Path(tmpdir)
    test_date = dt.datetime(2021, 4, 5).date()
    test_hemisphere = NORTH
    test_resolution: Final = "12"
    make_idecdr_netcdf(
        date=test_date,
        hemisphere=test_hemisphere,
        resolution=test_resolution,
        cdr_data_dir=tmpdir_path,
        excluded_fields=(cdr_conc_fieldname,),
    )
    output_path = get_idecdr_filepath(
        hemisphere=test_hemisphere,
        date=test_date,
        resolution=test_resolution,
        cdr_data_dir=tmpdir_path,
    )

    assert output_path.is_file()

    ds = xr.open_dataset(output_path)
    assert cdr_conc_fieldname not in ds.variables.keys()
