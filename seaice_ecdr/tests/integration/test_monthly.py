import pytest
import xarray as xr
from pm_tb_data._types import NORTH

from seaice_ecdr import monthly
from seaice_ecdr.tests.integration import base_output_dir_test_path  # noqa


@pytest.mark.order(after="test_complete_daily.py::test_make_cdecdr_netcdf")
def test_make_monthly_nc(base_output_dir_test_path, monkeypatch):  # noqa
    # usually we require at least 20 days of data for a valid month. This mock
    # data is just 3 days in size, so we need to mock the
    # "check_min_days_for_valid_month" function.
    monkeypatch.setattr(
        monthly, "check_min_days_for_valid_month", lambda *_args, **_kwargs: True
    )

    output_path = monthly.make_monthly_nc(
        year=2022,
        month=3,
        hemisphere=NORTH,
        resolution="12.5",
        base_output_dir=base_output_dir_test_path,
    )

    assert output_path.is_file()

    # Confirm that the netcdf file is readable and matches the dataset provided
    # by `make_monthly_ds`
    ds = xr.open_dataset(output_path)
    assert ds is not None

    # Assert that the checksums exist where we expect them to be.
    checksum_filepath = (
        base_output_dir_test_path
        / "complete"
        / NORTH
        / "checksums"
        / "monthly"
        / (output_path.name + ".mnf")
    )
    assert checksum_filepath.is_file()
