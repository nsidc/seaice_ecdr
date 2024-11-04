import datetime as dt

import datatree
import pytest
from pm_tb_data._types import NORTH

from seaice_ecdr.publish_daily import publish_daily_nc
from seaice_ecdr.tests.integration import base_output_dir_test_path  # noqa


@pytest.mark.order(after="test_intermediate_daily.py::test_make_standard_cdecdr_netcdf")
def test_publish_daily_nc(base_output_dir_test_path):  # noqa
    for day in range(1, 5):
        output_path = publish_daily_nc(
            base_output_dir=base_output_dir_test_path,
            hemisphere="north",
            resolution="25",
            date=dt.date(2022, 3, day),
        )

        assert output_path.is_file()

        ds = datatree.open_datatree(output_path)

        # TODO: we would expect this date to contain the amsr2 prototype group,
        # but the integration tests do not currently run with the AMSR2 platform
        # config, so the `prototype_am2` group is not present.
        # assert "prototype_am2" in ds.groups

        assert "/cdr_supplementary" in ds.groups

        # Assert that the checksums exist where we expect them to be.
        checksum_filepath = (
            base_output_dir_test_path
            / "complete"
            / NORTH
            / "checksums"
            / "daily"
            / "2022"
            / (output_path.name + ".mnf")
        )
        assert checksum_filepath.is_file()
