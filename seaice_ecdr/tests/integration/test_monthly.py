import datetime as dt
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from pm_tb_data._types import NORTH

from seaice_ecdr.complete_daily_ecdr import make_cdecdr_netcdf
from seaice_ecdr.monthly import get_daily_ds_for_month


@pytest.fixture(scope="session")
def tmpdir_path():
    tmpdir = TemporaryDirectory()
    tmpdir_path = Path(tmpdir.name)

    yield tmpdir_path

    tmpdir.cleanup()


@pytest.fixture(scope="session")
def daily_complete_data(tmpdir_path):
    for day in range(1, 5):
        make_cdecdr_netcdf(
            date=dt.date(2022, 3, day),
            hemisphere=NORTH,
            resolution="12.5",
            ecdr_data_dir=tmpdir_path,
        )


def test_get_daily_ds_for_month(daily_complete_data, tmpdir_path):  # noqa
    ds = get_daily_ds_for_month(
        year=2022,
        month=3,
        ecdr_data_dir=tmpdir_path,
        hemisphere=NORTH,
        resolution="12.5",
    )

    assert ds is not None
