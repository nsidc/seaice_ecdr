import datetime as dt
from unittest.mock import patch

import numpy as np

from seaice_ecdr.initial_daily_ecdr import calc_cdr_conc
from seaice_ecdr.platforms.config import F08_PLATFORM


def test_calc_cdr_conc():
    date = dt.date(1991, 9, 15)

    conc_threshold = 33.0

    bt_conc = np.array(
        [
            [6.663049, 7.6880083, 11.933167, 10.900373, 21.032698],
            [7.706575, 8.171715, 12.4997015, 12.725052, 22.634706],
            [10.355176, 9.139128, 11.273984, 17.03636, 13.835804],
            [11.290873, 15.144656, 17.58346, 17.247152, 25.801811],
            [np.nan, 31.862719, 38.679565, 37.66979, 47.36086],
            [conc_threshold, 31.0, 38.0, 37.0, 124.0],
        ],
    )

    nt_conc = np.array(
        [
            [0.0, 0.0, 1.1881592, 1.4489053, 9.787772],
            [0.0, 0.0, 0.43176675, 1.8004451, 10.667047],
            [0.49504876, 0.8246981, 1.7427928, 5.8003826, 4.3895097],
            [1.3648397, 2.6308284, 7.9928584, 6.5740256, 12.076052],
            [np.nan, 16.607742, 25.372278, 24.640556, 28.790665],
            [conc_threshold, 50.0, 25.0, 44.0, 68.0],
        ],
    )

    expected_cdr_conc = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [np.nan, 0, 38.679565, 37.66979, 47.36086],
            [conc_threshold, 0, 38.0, 44.0, 100],
        ],
    )

    with patch(
        "seaice_ecdr.initial_daily_ecdr.get_cdr_conc_threshold",
        return_value=conc_threshold,
    ) as _patched_get_cdr_conc_threshold:
        actual_cdr_conc = calc_cdr_conc(
            bt_conc=bt_conc,
            nt_conc=nt_conc,
            date=date,
            platform=F08_PLATFORM,
            hemisphere="north",
        )

    np.testing.assert_array_equal(expected_cdr_conc, actual_cdr_conc)
