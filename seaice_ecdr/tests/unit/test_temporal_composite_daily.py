"""Tests for initial daily ECDR generation."""

# TODO: The tests should probably not require "real" data, but
#  should work with mock data.  Or else, they should be moved to
#  tests/integration/ directory.

import datetime as dt
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import pandas as pd
import pytest

from loguru import logger


from seaice_ecdr.temporal_composite_daily import (
    get_sample_idecdr_filename,
    iter_dates_near_date,
    get_standard_initial_daily_ecdr_filename,
    temporally_composite_dataarray,
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


def test_temporal_composite_max_interp_range_9():
    """interp_range > 9 should yield runtime error."""

    with pytest.raises(RuntimeError, match=r'interp_range'):
        tempcomp_array, tempcomp_flags = temporally_composite_dataarray(
            target_date=dt.date(2020, 1, 1),
            da=xr.DataArray(coords=(range(2), range(3), range(4))),
            interp_range=10,
        )


def test_temporal_composite_da_oneday():
    """Verify that temporal composite of array with single day is no change."""
    no_siconc_val = 0.0
    not_siext_val = 8.9
    some_siconc_val = 78.2
    max_siconc_val = 100.0
    land_siconc_val = 254.0
    missing_siconc_val = np.nan

    mock_data = np.array([[
        [no_siconc_val, not_siext_val, some_siconc_val],
        [max_siconc_val, land_siconc_val, missing_siconc_val],
        [missing_siconc_val, missing_siconc_val, missing_siconc_val],
        [missing_siconc_val, missing_siconc_val, missing_siconc_val],
    ]])
    tdim, ydim, xdim = mock_data.shape
    mock_xs = np.arange(xdim)
    mock_ys = np.arange(ydim)

    mock_date = dt.date(2023, 1, 10)
    mock_dates=pd.date_range(mock_date, periods=tdim)

    ref_time=pd.Timestamp("2023-01-01")
    mock_dims=['time', 'y', 'x']
    mock_coords = dict(
        time=('time', mock_dates),
        reference_time=ref_time,
        y=(["y",], mock_ys),
        x=(["x",], mock_xs),
    )
    da = xr.DataArray(
        data=mock_data,
        dims=mock_dims,
        coords=mock_coords,
    )

    tcarray, tcflags = temporally_composite_dataarray(
        target_date=mock_date,
        da=da,
        interp_range=0,
    )

    assert np.array_equal(da, tcarray, equal_nan=True)


def test_temporal_composite_da_multiday():
    """Verify that temporal composite over multiple days.

    Setting up an array that tests up to two days in each direction
    """
    no_siconc_val = 0.0
    not_siext_val = 8.9
    some_siconc_val = 78.2
    max_siconc_val = 100.0
    land_siconc_val = 254.0
    left_siconc_val = 20.0
    right_siconc_val = 80.0
    mid_siconc_val = 50.0
    one_third_siconc_val = 40.0
    two_thirds_siconc_val = 60.0
    missing_siconc_val = np.nan

    still_missing_flag = 255

    # Set up mock array so that left (=prior) day test values are 20 if exist
    # Temporal flag values start at zero; add 10*prior_dist, add 1*next_dist,

    expected_temporal_bitmask = np.array(
        [
            # All valid values in orig field
            [0, 0, 0],
            # valid, then land (not interp'ed) and never-filled missing
            [0, 0, still_missing_flag],
            [10, 11, 12],   # have prior 1 and next not,1,2
            [20, 21, 22],   # have prior 2 and next not,1,2
            [still_missing_flag, 1, 2],    # no prior and next not,1,2
        ],
    )
    expected_temporal_composite_data = np.array([
        [
            [no_siconc_val, not_siext_val, some_siconc_val],
            [max_siconc_val, land_siconc_val, missing_siconc_val],
            [left_siconc_val, mid_siconc_val, one_third_siconc_val],
            [left_siconc_val, two_thirds_siconc_val, mid_siconc_val],
            [missing_siconc_val, right_siconc_val, right_siconc_val],
        ],
    ])
    # --- begin mock data ----------------------------------------------
    mock_data = np.array([
        [
            # this is target date minus 2 days
            [no_siconc_val, not_siext_val, some_siconc_val],
            [max_siconc_val, land_siconc_val, missing_siconc_val],
            # this middle should be ignored bc exists in day-1
            [missing_siconc_val, max_siconc_val, right_siconc_val],
            [left_siconc_val, left_siconc_val, left_siconc_val],
            [missing_siconc_val, missing_siconc_val, missing_siconc_val],
        ],
        [
            # this is target date minus 1 day
            [no_siconc_val, not_siext_val, some_siconc_val],
            [max_siconc_val, land_siconc_val, missing_siconc_val],
            [left_siconc_val, left_siconc_val, left_siconc_val],
            [missing_siconc_val, missing_siconc_val, missing_siconc_val],
            [missing_siconc_val, missing_siconc_val, missing_siconc_val],
        ],
        [
            # this is the target-date
            [no_siconc_val, not_siext_val, some_siconc_val],
            [max_siconc_val, land_siconc_val, missing_siconc_val],
            [missing_siconc_val, missing_siconc_val, missing_siconc_val],
            [missing_siconc_val, missing_siconc_val, missing_siconc_val],
            [missing_siconc_val, missing_siconc_val, missing_siconc_val],
        ],
        [
            # this is target date plus 1 day
            [no_siconc_val, not_siext_val, some_siconc_val],
            [max_siconc_val, land_siconc_val, missing_siconc_val],
            [missing_siconc_val, right_siconc_val, missing_siconc_val],
            [missing_siconc_val, right_siconc_val, missing_siconc_val],
            [missing_siconc_val, right_siconc_val, missing_siconc_val],
        ],
        [
            # this is target date plus 2 days
            [no_siconc_val, not_siext_val, some_siconc_val],
            [max_siconc_val, land_siconc_val, missing_siconc_val],
            # the next two middle values should be ignored because exist day+1
            [missing_siconc_val, max_siconc_val, right_siconc_val],
            [missing_siconc_val, missing_siconc_val, right_siconc_val],
            [missing_siconc_val, right_siconc_val, right_siconc_val],
        ],
    ])
    # --- end mock data ------------------------------------------------
    tdim, ydim, xdim = mock_data.shape
    mock_xs = np.arange(xdim)
    mock_ys = np.arange(ydim)

    mock_date = dt.date(2023, 1, 10)
    mock_start_date = mock_date - dt.timedelta(days=tdim//2)
    mock_dates=pd.date_range(mock_start_date, periods=tdim)

    ref_time=pd.Timestamp("2023-01-01")
    mock_dims=['time', 'y', 'x']
    mock_coords = dict(
        time=('time', mock_dates),
        reference_time=ref_time,
        y=(["y",], mock_ys),
        x=(["x",], mock_xs),
    )
    da = xr.DataArray(
        data=mock_data,
        dims=mock_dims,
        coords=mock_coords,
    )

    target_date = mock_date
    time_spread = (tdim - 1) // 2

    temporal_composite, temporal_bitmask = temporally_composite_dataarray(
        target_date=mock_date,
        da=da,
        interp_range=time_spread,
    )

    assert np.array_equal(temporal_composite.data, expected_temporal_composite_data, equal_nan=True)
    assert np.array_equal(temporal_bitmask, expected_temporal_bitmask, equal_nan=True)
