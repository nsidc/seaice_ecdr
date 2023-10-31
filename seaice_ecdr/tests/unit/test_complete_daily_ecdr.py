"""Tests of the routines in test_complete_daily_ecdr.py.  """
import datetime as dt

from seaice_ecdr import complete_daily_ecdr as cdecdr


def test_cdecdr_date_iter_starts_with_jan1():
    """Verify that the date iterator starts with the first day of the year."""
    year = 2020
    day_of_year = 100
    target_date = dt.date(year, 1, 1) + dt.timedelta(days=day_of_year - 1)
    for iter_date in cdecdr.iter_cdecdr_dates(target_date):
        first_date = iter_date
        assert first_date == dt.date(year, 1, 1)

        return


def test_cdecdr_date_iter_ends_with_target_date():
    """Verify that the date iterator finishs with the passed date."""
    year = 2020
    day_of_year = 100
    target_date = dt.date(year, 1, 1) + dt.timedelta(days=day_of_year - 1)
    for iter_date in cdecdr.iter_cdecdr_dates(target_date):
        latest_date = iter_date

    assert latest_date == target_date
