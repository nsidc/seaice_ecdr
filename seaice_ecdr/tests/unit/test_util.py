import datetime as dt

from pm_tb_data._types import NORTH, SOUTH

from seaice_ecdr.util import (
    standard_daily_aggregate_filename,
    standard_daily_filename,
    standard_monthly_aggregate_filename,
    standard_monthly_filename,
)


def test_daily_filename_north():
    expected = "sic_psn12.5_20210101_am2_v05r00.nc"

    actual = standard_daily_filename(
        hemisphere=NORTH, resolution="12.5", sat="am2", date=dt.date(2021, 1, 1)
    )

    assert actual == expected


def test_daily_filename_south():
    expected = "sic_pss12.5_20210101_am2_v05r00.nc"

    actual = standard_daily_filename(
        hemisphere=SOUTH, resolution="12.5", sat="am2", date=dt.date(2021, 1, 1)
    )

    assert actual == expected


def test_daily_aggregate_filename():
    expected = "sic_psn12.5_20210101-20211231_v05r00.nc"

    actual = standard_daily_aggregate_filename(
        hemisphere=NORTH,
        resolution="12.5",
        start_date=dt.date(2021, 1, 1),
        end_date=dt.date(2021, 12, 31),
    )

    assert actual == expected


def test_monthly_filename_north():
    expected = "sic_psn12.5_202101_am2_v05r00.nc"

    actual = standard_monthly_filename(
        hemisphere=NORTH,
        resolution="12.5",
        sat="am2",
        year=2021,
        month=1,
    )

    assert actual == expected


def test_monthly_filename_south():
    expected = "sic_pss12.5_202101_am2_v05r00.nc"

    actual = standard_monthly_filename(
        hemisphere=SOUTH,
        resolution="12.5",
        sat="am2",
        year=2021,
        month=1,
    )

    assert actual == expected


def test_monthly_aggregate_filename():
    expected = "sic_pss12.5_202101-202112_v05r00.nc"

    actual = standard_monthly_aggregate_filename(
        hemisphere=SOUTH,
        resolution="12.5",
        start_year=2021,
        start_month=1,
        end_year=2021,
        end_month=12,
    )

    assert actual == expected
