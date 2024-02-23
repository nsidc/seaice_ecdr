import datetime as dt

from pm_tb_data._types import NORTH

from seaice_ecdr.checksum import get_checksum_filepath
from seaice_ecdr.complete_daily_ecdr import make_standard_cdecdr_netcdf
from seaice_ecdr.tests.integration import ecdr_data_dir_test_path  # noqa


def test_make_standard_cdecdr_netcdf(ecdr_data_dir_test_path):  # noqa
    for day in range(1, 5):
        output_path = make_standard_cdecdr_netcdf(
            date=dt.date(2022, 3, day),
            hemisphere=NORTH,
            resolution="12.5",
            ecdr_data_dir=ecdr_data_dir_test_path,
        )

        assert output_path.is_file()

        # Assert that the checksums exist where we expect them to be.
        checksum_filepath = get_checksum_filepath(
            input_filepath=output_path,
            ecdr_data_dir=ecdr_data_dir_test_path,
            product_type="daily",
        )
        assert checksum_filepath.is_file()
