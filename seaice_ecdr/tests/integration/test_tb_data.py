import numpy as np
from pm_tb_data._types import NORTH

from seaice_ecdr.platforms import get_platform_start_dates
from seaice_ecdr.tb_data import get_ecdr_tb_data


def test_get_ecdr_tb_data():
    platform_start_dates = get_platform_start_dates()
    for date, platform in platform_start_dates.items():
        ecdr_tb_data = get_ecdr_tb_data(date=date, hemisphere=NORTH)
        assert ecdr_tb_data.platform == platform

        assert not np.all(np.isnan(ecdr_tb_data.tbs.v19))
        assert not np.all(np.isnan(ecdr_tb_data.tbs.h19))
        assert not np.all(np.isnan(ecdr_tb_data.tbs.v22))
        assert not np.all(np.isnan(ecdr_tb_data.tbs.v37))
        assert not np.all(np.isnan(ecdr_tb_data.tbs.h37))
