"""Tests of the routines in test_complete_daily_ecdr.py.  """
import datetime as dt
from pathlib import Path

import numpy as np
from pm_tb_data._types import NORTH, SOUTH

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


def test_no_melt_onset_for_southern_hemisphere():
    """Verify that melt onset is all fill value when not in melt season."""
    for date in (dt.date(2020, 2, 1), dt.date(2021, 6, 2), dt.date(2020, 10, 3)):
        melt_onset_field = cdecdr.create_melt_onset_field(
            date=date,
            hemisphere=SOUTH,
            resolution="12",
            tie_dir=Path("./"),
            cde_dir=Path("./"),
        )
        assert melt_onset_field is None


def test_melt_onset_field_outside_melt_season():
    """Verify that melt onset is all fill value when not in melt season."""
    hemisphere = NORTH
    tie_dir = Path("./")
    cde_dir = Path("./")
    no_melt_flag = 255

    for date in (dt.date(2020, 2, 1), dt.date(2020, 10, 3)):
        melt_onset_field = cdecdr.create_melt_onset_field(
            date=date,
            hemisphere=hemisphere,
            resolution="12",
            tie_dir=tie_dir,
            cde_dir=cde_dir,
            no_melt_flag=no_melt_flag,
        )
        assert np.all(melt_onset_field == no_melt_flag)
