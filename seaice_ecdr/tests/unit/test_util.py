import datetime as dt

from pm_tb_data._types import NORTH, SOUTH

from seaice_ecdr.util import standard_daily_filename


def test_daily_filename_north():
    expected = "sic_psn12.5_20210101_amsr2_v05r00.nc"

    actual = standard_daily_filename(
        hemisphere=NORTH, resolution="12.5", sat="amsr2", date=dt.date(2021, 1, 1)
    )

    assert actual == expected


def test_daily_filename_south():
    expected = "sic_pss12.5_20210101_amsr2_v05r00.nc"

    actual = standard_daily_filename(
        hemisphere=SOUTH, resolution="12.5", sat="amsr2", date=dt.date(2021, 1, 1)
    )

    assert actual == expected
