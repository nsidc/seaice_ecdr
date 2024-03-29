import datetime as dt
from typing import get_args

from pm_tb_data._types import Hemisphere

from seaice_ecdr.ancillary import get_smmr_invalid_ice_mask


def test_get_smmr_invalid_ice_mask():
    for hemisphere in get_args(Hemisphere):
        icemask = get_smmr_invalid_ice_mask(
            hemisphere=hemisphere, date=dt.date(2023, 1, 29)
        )

        assert icemask.dtype == bool
        assert icemask.any()
