import datetime as dt
from typing import Final, get_args

import numpy as np
from pm_tb_data._types import Hemisphere

from seaice_ecdr.ancillary import (
    ANCILLARY_SOURCES,
    get_adj123_field,
    get_non_ocean_mask,
    get_smmr_invalid_ice_mask,
)

test_ancillary_source: Final = "CDRv5"


def test_ancillary_source_is_valid():
    assert test_ancillary_source in get_args(ANCILLARY_SOURCES)


def test_get_smmr_invalid_ice_masks():
    for ancillary_source in get_args(ANCILLARY_SOURCES):
        for hemisphere in get_args(Hemisphere):
            icemask = get_smmr_invalid_ice_mask(
                date=dt.date(2023, 1, 29),
                hemisphere=hemisphere,
                resolution="25",
                ancillary_source=ancillary_source,
            )

            assert icemask.dtype == bool
            assert icemask.any()


def test_adj123_does_not_overlap_land():
    test_resolution: Final = "25"

    for hemisphere in get_args(Hemisphere):
        non_ocean_mask = get_non_ocean_mask(
            hemisphere=hemisphere,
            resolution=test_resolution,
            ancillary_source=test_ancillary_source,
        )

        adj123_mask = get_adj123_field(
            hemisphere=hemisphere,
            resolution=test_resolution,
            ancillary_source=test_ancillary_source,
        )

        is_land = non_ocean_mask.data
        is_adj1 = adj123_mask.data == 1
        is_adj2 = adj123_mask.data == 2
        is_adj3 = adj123_mask.data == 3

        """ Sample debugging segment
        try:
            assert not np.any(is_land & is_adj1)
        except AssertionError as e:
            print('adj123 value of 1 overlaps with land:')
            print(f'   ancillary_source: {test_ancillary_source}')
            print(f'         hemisphere: {hemisphere}')
            print(f'         resolution: {test_resolution}')
            breakpoint()
            raise e
        """
        assert not np.any(is_land & is_adj1)
        assert not np.any(is_land & is_adj2)
        assert not np.any(is_land & is_adj3)
