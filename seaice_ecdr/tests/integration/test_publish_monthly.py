import datatree
import pytest
from pm_tb_data._types import NORTH

from seaice_ecdr.publish_monthly import prepare_monthly_nc_for_publication
from seaice_ecdr.tests.integration import base_output_dir_test_path  # noqa


@pytest.mark.order(
    after="test_intermediate_monthly.py::test_make_intermediate_monthly_nc"
)
def test_publish_monthly_nc(base_output_dir_test_path):  # noqa
    output_path = prepare_monthly_nc_for_publication(
        base_output_dir=base_output_dir_test_path,
        hemisphere="north",
        resolution="25",
        year=2022,
        month=3,
        is_nrt=False,
    )

    assert output_path.is_file()

    ds = datatree.open_datatree(output_path)

    # TODO: we would expect this date to contain the amsr2 prototype group,
    # but the integration tests do not currently run with the AMSR2 platform
    # config, so the `prototype_am2` group is not present.
    # assert "prototype_am2" in ds.groups
    assert "/cdr_supplementary" in ds.groups

    assert "valid_range" not in ds.time.attrs.keys()
    # assert "valid_range" not in ds.x.attrs.keys()
    # assert "valid_range" not in ds.y.attrs.keys()

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
