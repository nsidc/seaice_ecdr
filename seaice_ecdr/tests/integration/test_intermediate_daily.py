import datetime as dt

from pm_tb_data._types import NORTH

from seaice_ecdr.intermediate_daily import make_standard_cdecdr_netcdf
from seaice_ecdr.tests.integration import base_output_dir_test_path  # noqa


def test_make_standard_cdecdr_netcdf(base_output_dir_test_path):  # noqa
    for day in range(1, 5):
        output_path = make_standard_cdecdr_netcdf(
            date=dt.date(2022, 3, day),
            hemisphere=NORTH,
            resolution="25",
            base_output_dir=base_output_dir_test_path,
            land_spillover_alg="NT2",
        )

        assert output_path.is_file()
