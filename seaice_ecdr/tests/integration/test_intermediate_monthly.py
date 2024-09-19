from typing import Final

import pytest
import xarray as xr
from pm_tb_data._types import NORTH

from seaice_ecdr import intermediate_monthly
from seaice_ecdr.tests.integration import base_output_dir_test_path  # noqa
from seaice_ecdr.util import get_intermediate_output_dir

ancillary_source: Final = "CDRv5"


@pytest.mark.order(after="test_intermediate_daily.py::test_make_cdecdr_netcdf")
def test_make_intermediate_monthly_nc(base_output_dir_test_path, monkeypatch):  # noqa
    # usually we require at least 20 days of data for a valid month. This mock
    # data is just 3 days in size, so we need to mock the
    # "check_min_days_for_valid_month" function.
    monkeypatch.setattr(
        intermediate_monthly,
        "check_min_days_for_valid_month",
        lambda *_args, **_kwargs: True,
    )

    intermediate_output_dir = get_intermediate_output_dir(
        base_output_dir=base_output_dir_test_path,
        hemisphere=NORTH,
        is_nrt=False,
    )

    output_path = intermediate_monthly.make_intermediate_monthly_nc(
        year=2022,
        month=3,
        hemisphere=NORTH,
        resolution="25",
        intermediate_output_dir=intermediate_output_dir,
        ancillary_source=ancillary_source,
    )

    assert output_path.is_file()

    # Confirm that the netcdf file is readable and matches the dataset provided
    # by `make_intermediate_monthly_ds`
    ds = xr.open_dataset(output_path)
    assert ds is not None
