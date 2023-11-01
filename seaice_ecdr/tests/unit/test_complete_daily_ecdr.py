"""Tests of the routines in test_complete_daily_ecdr.py.  """
import datetime as dt
from pathlib import Path

import numpy as np
from pm_tb_data._types import NORTH, SOUTH

from seaice_ecdr import complete_daily_ecdr as cdecdr


def test_no_melt_onset_for_southern_hemisphere():
    """Verify that melt onset is all fill value when not in melt season."""
    for date in (dt.date(2020, 2, 1), dt.date(2021, 6, 2), dt.date(2020, 10, 3)):
        melt_onset_field = cdecdr.create_melt_onset_field(
            date=date,
            hemisphere=SOUTH,
            resolution="12",
            ide_dir=Path("./"),
            tie_dir=Path("./"),
            output_dir=Path("./"),
        )
        assert melt_onset_field is None


def test_melt_onset_field_outside_melt_season():
    """Verify that melt onset is all fill value when not in melt season."""
    hemisphere = NORTH
    ide_dir = Path("./")
    tie_dir = Path("./")
    output_dir = Path("./")
    no_melt_flag = 255

    for date in (dt.date(2020, 2, 1), dt.date(2020, 10, 3)):
        melt_onset_field = cdecdr.create_melt_onset_field(
            date=date,
            hemisphere=hemisphere,
            resolution="12",
            ide_dir=ide_dir,
            tie_dir=tie_dir,
            output_dir=output_dir,
            no_melt_flag=no_melt_flag,
        )
        assert np.all(melt_onset_field == no_melt_flag)
