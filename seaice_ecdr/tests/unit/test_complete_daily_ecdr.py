"""Tests of the routines in test_complete_daily_ecdr.py.  """

import datetime as dt
from pathlib import Path

import numpy as np
import pytest
from pm_tb_data._types import NORTH, SOUTH

from seaice_ecdr import complete_daily_ecdr as cdecdr
from seaice_ecdr.melt import MELT_ONSET_FILL_VALUE
from seaice_ecdr.util import get_complete_output_dir, get_intermediate_output_dir


def test_no_melt_onset_for_southern_hemisphere(tmpdir):
    """Verify that attempting to create a melt onset field for the SH raises an error"""
    complete_output_dir = get_complete_output_dir(
        base_output_dir=Path(tmpdir),
    )
    intermediate_output_dir = get_intermediate_output_dir(
        base_output_dir=Path(tmpdir),
        is_nrt=False,
    )
    for date in (dt.date(2020, 2, 1), dt.date(2021, 6, 2), dt.date(2020, 10, 3)):
        with pytest.raises(RuntimeError):
            cdecdr.create_melt_onset_field(
                date=date,
                hemisphere=SOUTH,
                resolution="12.5",
                complete_output_dir=complete_output_dir,
                intermediate_output_dir=intermediate_output_dir,
                is_nrt=False,
            )


def test_melt_onset_field_outside_melt_season(tmpdir):
    """Verify that melt onset is all fill value when not in melt season."""
    hemisphere = NORTH

    complete_output_dir = get_complete_output_dir(
        base_output_dir=Path(tmpdir),
    )
    intermediate_output_dir = get_intermediate_output_dir(
        base_output_dir=Path(tmpdir),
        is_nrt=False,
    )
    for date in (dt.date(2020, 2, 1), dt.date(2020, 10, 3)):
        melt_onset_field = cdecdr.create_melt_onset_field(
            date=date,
            hemisphere=hemisphere,
            resolution="12.5",
            complete_output_dir=complete_output_dir,
            intermediate_output_dir=intermediate_output_dir,
            is_nrt=False,
        )
        assert np.all(melt_onset_field == MELT_ONSET_FILL_VALUE)
