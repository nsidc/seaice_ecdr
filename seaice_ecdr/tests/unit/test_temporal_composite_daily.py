"""Unit tests for initial daily ECDR generation."""

import datetime as dt
import sys
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from loguru import logger
from pm_tb_data._types import NORTH

from seaice_ecdr.constants import ECDR_PRODUCT_VERSION
from seaice_ecdr.initial_daily_ecdr import get_idecdr_dir, get_idecdr_filepath
from seaice_ecdr.temporal_composite_daily import (
    iter_dates_near_date,
    temporally_composite_dataarray,
)

# Set the default minimum log notification to Warning
# TODO: Think about logging holistically...
try:
    logger.remove(0)  # Removes previous logger info
    logger.add(sys.stderr, level="WARNING")
except ValueError:
    logger.debug(f"Started logging in {__name__}")
    logger.add(sys.stderr, level="WARNING")


def compose_tyx_dataarray(
    data,
    xvals,
    yvals,
    datevals,
) -> xr.DataArray:
    """Create a simple data array with coords (time, y, x)."""
    dims = ["time", "y", "x"]
    coords = dict(
        time=("time", datevals),
        y=(
            [
                "y",
            ],
            yvals,
        ),
        x=(
            [
                "x",
            ],
            xvals,
        ),
    )
    dataarray = xr.DataArray(
        data=data,
        dims=dims,
        coords=coords,
    )

    return dataarray


def test_date_iterator():
    """Verify operation of date temporal composite's date iterator.

    A data with only data at the target date should yield the values
    on that date.
    """
    date = dt.date(2021, 2, 19)

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


def test_access_to_standard_output_filename(tmpdir):
    """Verify that standard output file names can be generated."""
    date = dt.date(2021, 2, 19)
    resolution: Final = "12.5"

    ecdr_data_dir = Path(tmpdir)
    sample_ide_filepath = get_idecdr_filepath(
        date=date,
        hemisphere=NORTH,
        resolution=resolution,
        ecdr_data_dir=ecdr_data_dir,
    )
    expected_filepath = (
        get_idecdr_dir(ecdr_data_dir=ecdr_data_dir)
        / f"idecdr_sic_psn12.5_20210219_ausi_{ECDR_PRODUCT_VERSION}.nc"
    )

    assert sample_ide_filepath == expected_filepath


def test_temporal_composite_max_interp_range_9():
    """interp_range > 9 should yield runtime error."""

    with pytest.raises(RuntimeError, match=r"interp_range"):
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

    mock_data = np.array(
        [
            [
                [no_siconc_val, not_siext_val, some_siconc_val],
                [max_siconc_val, land_siconc_val, missing_siconc_val],
                [missing_siconc_val, missing_siconc_val, missing_siconc_val],
                [missing_siconc_val, missing_siconc_val, missing_siconc_val],
            ]
        ]
    )
    tdim, ydim, xdim = mock_data.shape

    mock_xvals = np.arange(xdim)
    mock_yvals = np.arange(ydim)

    mock_date = dt.date(2023, 1, 10)
    mock_dates = pd.date_range(mock_date, periods=tdim)

    initial_data_array = compose_tyx_dataarray(
        mock_data, mock_xvals, mock_yvals, mock_dates
    )

    # Note: the key is to test that if interp_range is zero,
    #       then the output will equal the input
    temporal_composite, temporal_flags = temporally_composite_dataarray(
        target_date=mock_date,
        da=initial_data_array,
        interp_range=0,
    )

    assert np.array_equal(temporal_composite, initial_data_array, equal_nan=True)


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

    expected_temporal_flags = np.array(
        [
            # All valid values in orig field
            [0, 0, 0],
            # valid, then land (not interp'ed) and never-filled missing
            [0, 0, still_missing_flag],
            [still_missing_flag, 1, 2],  # no prior and next not,1,2
            [10, 11, 12],  # have prior 1 and next not,1,2
            [20, 21, 22],  # have prior 2 and next not,1,2
        ],
    )
    expected_temporal_composite_data = np.array(
        [
            [
                [no_siconc_val, not_siext_val, some_siconc_val],
                [max_siconc_val, land_siconc_val, missing_siconc_val],
                [missing_siconc_val, right_siconc_val, right_siconc_val],
                [left_siconc_val, mid_siconc_val, one_third_siconc_val],
                [left_siconc_val, two_thirds_siconc_val, mid_siconc_val],
            ],
        ]
    )
    # --- begin mock data ----------------------------------------------
    mock_data = np.array(
        [
            [
                # this is target date minus 2 days
                [no_siconc_val, not_siext_val, some_siconc_val],
                [max_siconc_val, land_siconc_val, missing_siconc_val],
                [missing_siconc_val, missing_siconc_val, missing_siconc_val],
                # the next row should be ignored bc exists in day-1
                [missing_siconc_val, max_siconc_val, right_siconc_val],
                [left_siconc_val, left_siconc_val, left_siconc_val],
            ],
            [
                # this is target date minus 1 day
                [no_siconc_val, not_siext_val, some_siconc_val],
                [max_siconc_val, land_siconc_val, missing_siconc_val],
                [missing_siconc_val, missing_siconc_val, missing_siconc_val],
                [left_siconc_val, left_siconc_val, left_siconc_val],
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
                [missing_siconc_val, right_siconc_val, right_siconc_val],
                # the next two middle values should be ignored because exist day+1
                [missing_siconc_val, max_siconc_val, right_siconc_val],
                [missing_siconc_val, missing_siconc_val, right_siconc_val],
            ],
        ]
    )
    # --- end mock data ------------------------------------------------
    tdim, ydim, xdim = mock_data.shape
    mock_xvals = np.arange(xdim)
    mock_yvals = np.arange(ydim)

    mock_date = dt.date(2023, 1, 10)
    mock_start_date = mock_date - dt.timedelta(days=tdim // 2)
    mock_dates = pd.date_range(mock_start_date, periods=tdim)

    input_data_array = compose_tyx_dataarray(
        mock_data, mock_xvals, mock_yvals, mock_dates
    )

    time_spread = (tdim - 1) // 2

    temporal_composite, temporal_flags = temporally_composite_dataarray(
        target_date=mock_date,
        da=input_data_array,
        interp_range=time_spread,
    )

    assert np.array_equal(
        temporal_composite.data, expected_temporal_composite_data, equal_nan=True
    )
    assert np.array_equal(temporal_flags, expected_temporal_flags, equal_nan=True)
