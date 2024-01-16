import datetime as dt

from pm_tb_data._types import NORTH

from seaice_ecdr.complete_daily_ecdr import make_cdecdr_netcdf
from seaice_ecdr.tests.integration import ecdr_data_dir_test_path  # noqa


def test_make_cdecdr_netcdf(ecdr_data_dir_test_path):  # noqa
    for day in range(1, 5):
        output_path = make_cdecdr_netcdf(
            date=dt.date(2022, 3, day),
            hemisphere=NORTH,
            resolution="12.5",
            ecdr_data_dir=ecdr_data_dir_test_path,
        )

        assert output_path.is_file()
