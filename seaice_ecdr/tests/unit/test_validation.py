import datetime as dt
from pathlib import Path

from seaice_ecdr.validation import ERROR_FILE_BITMASK, get_error_code


def test_get_error_code_no_errors():
    actual_error_code = get_error_code(
        num_total_pixels=10000,
        num_bad_pixels=0,
        num_missing_pixels=0,
        num_melt_pixels=0,
        date=dt.date(2022, 3, 1),
        data_fp=Path("foo"),
    )

    expected_error_code = ERROR_FILE_BITMASK["no_problems"]

    assert actual_error_code == expected_error_code


def test_get_error_code_file_empty():
    # The file is considered empty if the total number of pixels equals the
    # number of missing pixels.
    actual_error_code = get_error_code(
        num_total_pixels=100,
        num_bad_pixels=0,
        num_missing_pixels=100,
        num_melt_pixels=0,
        date=dt.date(2022, 3, 1),
        data_fp=Path("foo"),
    )

    expected_error_code = ERROR_FILE_BITMASK["file_exists_but_is_empty"]

    assert actual_error_code == expected_error_code


def test_get_error_code_bad_pixels_and_bad_melt():
    actual_error_code = get_error_code(
        num_total_pixels=10000,
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
