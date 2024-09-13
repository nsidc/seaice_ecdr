"""Tests for initial daily ECDR generation."""

import datetime as dt
from pathlib import Path
from typing import Final

import pytest
import xarray as xr
from loguru import logger
from pm_tb_data._types import NORTH

from seaice_ecdr.initial_daily_ecdr import (
    get_idecdr_filepath,
    initial_daily_ecdr_dataset,
    make_idecdr_netcdf,
    write_ide_netcdf,
)
from seaice_ecdr.platforms import SUPPORTED_PLATFORM_ID

cdr_conc_fieldname = "conc"


@pytest.fixture(scope="session")
def sample_idecdr_dataset_nh():
    """Set up the sample NH initial daily ecdr data set."""
    logger.info("testing: Creating sample idecdr dataset")

    test_date = dt.datetime(2021, 4, 5).date()
    test_hemisphere = NORTH
    test_resolution: Final = "25"
    ancillary_source: Final = "CDRv5"

    ide_conc_ds = initial_daily_ecdr_dataset(
        date=test_date,
        hemisphere=test_hemisphere,
        resolution=test_resolution,
        land_spillover_alg="NT2",
        ancillary_source=ancillary_source,
    )
    return ide_conc_ds


@pytest.fixture(scope="session")
def sample_idecdr_dataset_sh():
    """Set up the sample SH initial daily ecdr data set."""
    logger.info("testing: Creating sample idecdr dataset")

    test_date = dt.datetime(2021, 4, 5).date()
    test_hemisphere = NORTH
    test_resolution: Final = "25"
    ancillary_source: Final = "CDRv5"

    ide_conc_ds = initial_daily_ecdr_dataset(
        date=test_date,
        hemisphere=test_hemisphere,
        resolution=test_resolution,
        land_spillover_alg="NT2",
        ancillary_source=ancillary_source,
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
    written_path = write_ide_netcdf(
        ide_ds=sample_idecdr_dataset_nh,
        output_filepath=sample_output_filepath_nh,
    )
    assert sample_output_filepath_nh == written_path
    assert sample_output_filepath_nh.exists()

    # SH
    sample_output_filepath_sh = tmp_path / "sample_idecdr_sh.nc"
    written_path = write_ide_netcdf(
        ide_ds=sample_idecdr_dataset_sh,
        output_filepath=sample_output_filepath_sh,
    )
    assert sample_output_filepath_sh == written_path
    assert sample_output_filepath_sh.exists()


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
    """Test that idecdr contains a 'crs' field."""
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
        "raw_bt_seaice_conc",
        "raw_nt_seaice_conc",
        "bt_weather_mask",
        "nt_weather_mask",
        "invalid_ice_mask",
        "spatial_interpolation_flag",
    )
    for field_name in expected_fields:
        assert field_name in sample_idecdr_dataset_nh.variables.keys()
        assert field_name in sample_idecdr_dataset_sh.variables.keys()


def test_cli_idecdr_ncfile_creation(tmpdir):
    """Verify that code used in cli.sh creates netCDF file."""
    tmpdir_path = Path(tmpdir)
    test_date = dt.datetime(2021, 4, 5).date()
    test_hemisphere = NORTH
    test_resolution: Final = "25"
    test_platform_id: SUPPORTED_PLATFORM_ID = "am2"
    ancillary_source: Final = "CDRv5"

    make_idecdr_netcdf(
        date=test_date,
        hemisphere=test_hemisphere,
        resolution=test_resolution,
        intermediate_output_dir=tmpdir_path,
        excluded_fields=[],
        land_spillover_alg="NT2",
        ancillary_source=ancillary_source,
    )
    output_path = get_idecdr_filepath(
        hemisphere=test_hemisphere,
        date=test_date,
        platform_id=test_platform_id,
        resolution=test_resolution,
        intermediate_output_dir=tmpdir_path,
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
    test_resolution: Final = "25"
    test_platform_id: SUPPORTED_PLATFORM_ID = "am2"
    ancillary_source: Final = "CDRv5"

    make_idecdr_netcdf(
        date=test_date,
        hemisphere=test_hemisphere,
        resolution=test_resolution,
        intermediate_output_dir=tmpdir_path,
        excluded_fields=(cdr_conc_fieldname,),
        land_spillover_alg="NT2",
        ancillary_source=ancillary_source,
    )
    output_path = get_idecdr_filepath(
        hemisphere=test_hemisphere,
        date=test_date,
        platform_id=test_platform_id,
        resolution=test_resolution,
        intermediate_output_dir=tmpdir_path,
    )

    assert output_path.is_file()

    ds = xr.open_dataset(output_path)
    assert cdr_conc_fieldname not in ds.variables.keys()


def test_seaice_idecdr_has_tyx_data_vars(
    sample_idecdr_dataset_nh,
    sample_idecdr_dataset_sh,
):
    """Test that idecdr netcdf has (time, y, x) dims for data fields."""
    expected_tyx_fields = (
        "conc",
        "qa_of_cdr_seaice_conc",
        "raw_bt_seaice_conc",
        "raw_nt_seaice_conc",
        "bt_weather_mask",
        "nt_weather_mask",
        "invalid_ice_mask",
        "spatial_interpolation_flag",
    )
    for field_name in expected_tyx_fields:
        nh_data_shape = sample_idecdr_dataset_nh[field_name].shape
        assert len(nh_data_shape) == 3

        sh_data_shape = sample_idecdr_dataset_sh[field_name].shape
        assert len(sh_data_shape) == 3
