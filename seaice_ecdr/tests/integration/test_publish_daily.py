import datetime as dt

import datatree

from seaice_ecdr.publish_daily import publish_daily_nc
from seaice_ecdr.tests.integration import base_output_dir_test_path  # noqa


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
        # config, so the `prototype_amsr2` group is not present.
        # assert "prototype_amsr2" in ds.groups
        assert "/cdr_supplementary" in ds.groups

        assert "valid_range" not in ds.time.attrs.keys()
        assert "valid_range" not in ds.x.attrs.keys()
        assert "valid_range" not in ds.y.attrs.keys()