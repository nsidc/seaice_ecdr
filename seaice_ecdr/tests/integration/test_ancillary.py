import datetime as dt
from typing import get_args

from pm_tb_data._types import Hemisphere

from seaice_ecdr import ancillary
from seaice_ecdr.ancillary import get_smmr_invalid_ice_mask


def test_get_smmr_invalid_ice_mask():
    for hemisphere in get_args(Hemisphere):
        icemask = get_smmr_invalid_ice_mask(
            hemisphere=hemisphere, date=dt.date(2023, 1, 29)
        )

        assert icemask.dtype == bool
        assert icemask.any()


def test_verify_invalid_ice_mask_is_boolean():  # noqa
    """If the dtype attribute of invalid_ice_mask isn't set to bool,
    then using it to mask data in xarray will fail."""
    invalid_ice_mask = ancillary.get_invalid_ice_mask(
        hemisphere="north",
        date=dt.date(2013, 11, 1),
        resolution="12.5",
        platform="am2",
    )
    assert invalid_ice_mask.dtype == "bool"
