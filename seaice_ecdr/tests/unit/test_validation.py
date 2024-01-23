import datetime as dt
from pathlib import Path

import numpy as np
import xarray as xr

from seaice_ecdr.validation import ERROR_FILE_BITMASK, get_error_code


def test_get_error_code_no_errors():
    mock_seaice_conc = xr.DataArray([0, 0.25, 1, np.nan])
    actual_error_code = get_error_code(
        seaice_conc_var=mock_seaice_conc,
        num_bad_pixels=0,
        num_missing_pixels=0,
        num_melt_pixels=0,
        date=dt.date(2022, 3, 1),
        data_fp=Path("foo"),
    )

    expected_error_code = ERROR_FILE_BITMASK["no_problems"]

    assert actual_error_code == expected_error_code


def test_get_error_code_file_empty():
    mock_seaice_conc = xr.DataArray([np.nan, np.nan, np.nan, np.nan])
    actual_error_code = get_error_code(
        seaice_conc_var=mock_seaice_conc,
        num_bad_pixels=0,
        num_missing_pixels=0,
        num_melt_pixels=0,
        date=dt.date(2022, 3, 1),
        data_fp=Path("foo"),
    )

    expected_error_code = ERROR_FILE_BITMASK["file_exists_but_is_empty"]

    assert actual_error_code == expected_error_code


def test_get_error_code_bad_pixels_and_bad_melt():
    mock_seaice_conc = xr.DataArray([0, 0.25, 1, np.nan])
    actual_error_code = get_error_code(
        seaice_conc_var=mock_seaice_conc,
        num_bad_pixels=3,
        num_missing_pixels=0,
        num_melt_pixels=123,
        date=dt.date(2022, 2, 20),
        data_fp=Path("foo"),
    )

    expected_error_code = (
        ERROR_FILE_BITMASK["file_exists_but_conc_values_are_bad"]
        + ERROR_FILE_BITMASK["melt_flagged_on_wrong_day"]
    )

    assert actual_error_code == expected_error_code
