"""Tests for initial daily ECDR generation."""

import datetime as dt
import sys
from pathlib import Path

from loguru import logger


from seaice_ecdr.temporal_composite_daily import (
    get_sample_idecdr_filename,
    iter_dates_near_date,
    get_standard_initial_daily_ecdr_filename,
)


# Set the default minimum log notification to Warning
try:
    logger.remove(0)  # Removes previous logger info
    logger.add(sys.stderr, level="WARNING")
except ValueError:
    logger.debug(sys.stderr, f"Started logging in {__name__}")
    logger.add(sys.stderr, level="WARNING")


date = dt.date(2021, 2, 19)
hemisphere = "north"
resolution = "12"


def test_sample_filename_generation():
    """Verify creation of sample filename."""

    expected_filename = "sample_idecdr_north_12_20210219.nc"
    sample_filename = get_sample_idecdr_filename(date, hemisphere, resolution)

    assert sample_filename == expected_filename


def test_date_iterator():
    """Verify operation of date temporal composite's date iterator."""
    # Single argument to iterator should yield that date
    expected_date = date
    for iter_date in iter_dates_near_date(date):
        assert iter_date == expected_date

    # range of dates should start from day before and end with day after
    expected_dates = [date - dt.timedelta(days=1), date, date + dt.timedelta(days=1)]
    for idx, iter_date in enumerate(iter_dates_near_date(date, day_range=1)):
        assert iter_date == expected_dates[idx]

    # Skipping future means no dates after the seed_date are provided
    expected_dates = [date - dt.timedelta(days=2), date - dt.timedelta(days=1), date]
    for idx, iter_date in enumerate(
        iter_dates_near_date(date, day_range=2, skip_future=True)
    ):
        assert iter_date == expected_dates[idx]


def test_access_to_standard_output_filename():
    """Verify that standard output file names can be generated."""
    sample_ide_filepath = get_standard_initial_daily_ecdr_filename(
        date, hemisphere, resolution, output_directory=""
    )
    expected_filepath = Path("idecdr_NH_20210219_ausi_12km.nc")

    assert sample_ide_filepath == expected_filepath


'''
def test_create_and_read_initial_daily_ecdr():
    """Verify that if ide file does not exist, it is created and can be read."""
    from pm_icecon.util import standard_output_filename
'''


'''
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


def test_seaice_idecdr_can_output_to_netcdf(
    sample_idecdr_dataset_nh,
    sample_idecdr_dataset_sh,
    tmp_path,
):
    """Test that xarray dataset can be saved to a netCDF file."""

    # NH
    sample_output_filepath_nh = tmp_path / "sample_ecdr_nh.nc"
    sample_idecdr_dataset_nh.to_netcdf(sample_output_filepath_nh)
    assert sample_output_filepath_nh.is_file()

    # SH
    sample_output_filepath_sh = tmp_path / "sample_ecdr_sh.nc"
    sample_idecdr_dataset_sh.to_netcdf(sample_output_filepath_sh)
    assert sample_output_filepath_sh.is_file()


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
    assert isinstance(sample_idecdr_dataset_nh, type(xr.Dataset()))
    assert isinstance(sample_idecdr_dataset_sh, type(xr.Dataset()))
'''
