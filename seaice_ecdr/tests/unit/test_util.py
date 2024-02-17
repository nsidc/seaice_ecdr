import datetime as dt
from typing import Final

from pm_tb_data._types import NORTH, SOUTH

from seaice_ecdr.constants import ECDR_PRODUCT_VERSION
from seaice_ecdr.multiprocess_daily import get_dates_by_year
from seaice_ecdr.util import (
    date_range,
    sat_from_filename,
    standard_daily_aggregate_filename,
    standard_daily_filename,
    standard_monthly_aggregate_filename,
    standard_monthly_filename,
)


def test_daily_filename_north():
    expected = f"sic_psn12.5_20210101_am2_{ECDR_PRODUCT_VERSION}.nc"

    actual = standard_daily_filename(
        hemisphere=NORTH, resolution="12.5", sat="am2", date=dt.date(2021, 1, 1)
    )

    assert actual == expected


def test_daily_filename_south():
    expected = f"sic_pss12.5_20210101_am2_{ECDR_PRODUCT_VERSION}.nc"

    actual = standard_daily_filename(
        hemisphere=SOUTH, resolution="12.5", sat="am2", date=dt.date(2021, 1, 1)
    )

    assert actual == expected


def test_daily_aggregate_filename():
    expected = f"sic_psn12.5_20210101-20211231_{ECDR_PRODUCT_VERSION}.nc"

    actual = standard_daily_aggregate_filename(
        hemisphere=NORTH,
        resolution="12.5",
        start_date=dt.date(2021, 1, 1),
        end_date=dt.date(2021, 12, 31),
    )

    assert actual == expected


def test_monthly_filename_north():
    expected = f"sic_psn12.5_202101_am2_{ECDR_PRODUCT_VERSION}.nc"

    actual = standard_monthly_filename(
        hemisphere=NORTH,
        resolution="12.5",
        sat="am2",
        year=2021,
        month=1,
    )

    assert actual == expected


def test_monthly_filename_south():
    expected = f"sic_pss12.5_202101_am2_{ECDR_PRODUCT_VERSION}.nc"

    actual = standard_monthly_filename(
        hemisphere=SOUTH,
        resolution="12.5",
        sat="am2",
        year=2021,
        month=1,
    )

    assert actual == expected


def test_monthly_aggregate_filename():
    expected = f"sic_pss12.5_202101-202112_{ECDR_PRODUCT_VERSION}.nc"

    actual = standard_monthly_aggregate_filename(
        hemisphere=SOUTH,
        resolution="12.5",
        start_year=2021,
        start_month=1,
        end_year=2021,
        end_month=12,
    )

    assert actual == expected


def test_daily_sat_from_filename():
    expected_sat: Final = "am2"
    fn = standard_daily_filename(
        hemisphere=NORTH, resolution="12.5", sat=expected_sat, date=dt.date(2021, 1, 1)
    )

    actual_sat = sat_from_filename(fn)

    assert expected_sat == actual_sat


def test_monthly_sat_from_filename():
    expected_sat: Final = "F17"
    fn = standard_monthly_filename(
        hemisphere=SOUTH,
        resolution="12.5",
        sat=expected_sat,
        year=2021,
        month=1,
    )

    actual_sat = sat_from_filename(fn)

    assert expected_sat == actual_sat


def test_date_range():
    start_date = dt.date(2021, 1, 2)
    end_date = dt.date(2021, 1, 5)
    expected = [
        start_date,
        dt.date(2021, 1, 3),
        dt.date(2021, 1, 4),
        end_date,
    ]
    actual = list(date_range(start_date=start_date, end_date=end_date))

    assert expected == actual


def test_get_dates_by_year():
    actual = get_dates_by_year(
        [
            dt.date(2021, 1, 3),
            dt.date(2021, 1, 2),
            dt.date(2022, 1, 1),
            dt.date(1997, 3, 2),
            dt.date(1997, 4, 15),
            dt.date(2022, 1, 2),
        ]
    )

    expected = [
        [
            dt.date(1997, 3, 2),
            dt.date(1997, 4, 15),
        ],
        [
            dt.date(2021, 1, 2),
            dt.date(2021, 1, 3),
        ],
        [
            dt.date(2022, 1, 1),
            dt.date(2022, 1, 2),
        ],
    ]

    assert actual == expected
