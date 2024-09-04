import datetime as dt

import numpy as np
from pm_tb_data._types import NORTH, SOUTH

from seaice_ecdr.platforms import PLATFORM_START_DATES
from seaice_ecdr.tb_data import get_ecdr_tb_data


def test_get_ecdr_tb_data():
    for date, platform in PLATFORM_START_DATES.items():
        ecdr_tb_data = get_ecdr_tb_data(date=date, hemisphere=NORTH)
        assert ecdr_tb_data.platform == platform.short_name

        assert not np.all(np.isnan(ecdr_tb_data.tbs.v19))
        assert not np.all(np.isnan(ecdr_tb_data.tbs.h19))
        assert not np.all(np.isnan(ecdr_tb_data.tbs.v22))
        assert not np.all(np.isnan(ecdr_tb_data.tbs.v37))
        assert not np.all(np.isnan(ecdr_tb_data.tbs.h37))


def test_get_ecdr_tb_data_missing_channel():
    """Test that missing channels from the source get mapped to an all-null grid.

    We know this happens at least once: on 10/10/1995 SH for 22v (from F13).
    """
    ecdr_tb_data = get_ecdr_tb_data(date=dt.date(1995, 10, 10), hemisphere=SOUTH)

    # v22 is known to be missing for this day and hemisphere.
    assert np.all(np.isnan(ecdr_tb_data.tbs.v22))

    # The other channels should continue to exist.
    assert not np.all(np.isnan(ecdr_tb_data.tbs.v19))
    assert not np.all(np.isnan(ecdr_tb_data.tbs.h19))
    assert not np.all(np.isnan(ecdr_tb_data.tbs.v37))
    assert not np.all(np.isnan(ecdr_tb_data.tbs.h37))
