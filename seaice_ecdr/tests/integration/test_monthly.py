import datetime as dt
from pathlib import Path

import numpy as np
import xarray as xr
from pm_tb_data._types import NORTH

from seaice_ecdr import intermediate_monthly, util
from seaice_ecdr.intermediate_monthly import (
    CDR_SEAICE_CONC_QA_FLAG_DAILY_BITMASKS,
    _platform_id_for_month,
    make_intermediate_monthly_ds,
)


def _mock_daily_ds_for_month():
    # Each row is one pixel through time.
    # time ->
    _mock_data = [
        # average_concentration_exceeds_0.15 and at_least_half_the_days_have_sea_ice_conc_exceeds_0.15
        [0.20, 0.20, 0.20],
        # average_concentration_exceeds_0.30 and at_least_half_the_days_have_sea_ice_conc_exceeds_0.30
        [0.33, 0.33, 0.33],
        # at_least_half_the_days_have_sea_ice_conc_exceeds_0.15
        [0, 0.16, 0.16],
        # at_least_half_the_days_have_sea_ice_conc_exceeds_0.30 and 15 and average_concentration_exceeds_0.15
        [0, 0.31, 0.31],
        # invalid_ice_mask_applied
        [0, 0, 0],
        # at_least_one_day_during_month_has_spatial_interpolation
        [0.03, 0.02, 0.01],
        # at_least_one_day_during_month_has_temporal_interpolation
        [0.01, 0.02, 0.03],
        # at_least_one_day_during_month_has_melt_detected, average_concentration_exceeds_0.15 and 30
        [np.nan, 0.82, np.nan],
        # Land flag. All nan.
        [np.nan, np.nan, np.nan],
    ]

    _mock_surface_type_mask = [
        # average_concentration_exceeds_0.15 and at_least_half_the_days_have_sea_ice_conc_exceeds_0.15
        [50, 50, 50],
        # average_concentration_exceeds_0.30 and at_least_half_the_days_have_sea_ice_conc_exceeds_0.30
        [50, 50, 50],
        # at_least_half_the_days_have_sea_ice_conc_exceeds_0.15
        [50, 50, 50],
        # at_least_half_the_days_have_sea_ice_conc_exceeds_0.30 and 15 and average_concentration_exceeds_0.15
        [50, 50, 50],
        # invalid_ice_mask_applied
        [50, 50, 50],
        # at_least_one_day_during_month_has_spatial_interpolation
        [50, 50, 50],
        # at_least_one_day_during_month_has_temporal_interpolation
        [50, 50, 50],
        # at_least_one_day_during_month_has_melt_detected, average_concentration_exceeds_0.15 and 30
        [50, 50, 50],
        # Land flag. All nan.
        [250, 250, 250],
    ]

    _mock_daily_qa_fields = [
        # average_concentration_exceeds_0.15
        [np.nan, np.nan, np.nan],
        # average_concentration_exceeds_0.30
        [np.nan, np.nan, np.nan],
        # at_least_half_the_days_have_sea_ice_conc_exceeds_0.15
        [np.nan, np.nan, np.nan],
        # at_least_half_the_days_have_sea_ice_conc_exceeds_0.30
        [np.nan, np.nan, np.nan],
        # invalid_ice_mask_applied
        [
            CDR_SEAICE_CONC_QA_FLAG_DAILY_BITMASKS["invalid_ice_mask_applied"],
            CDR_SEAICE_CONC_QA_FLAG_DAILY_BITMASKS["invalid_ice_mask_applied"],
            CDR_SEAICE_CONC_QA_FLAG_DAILY_BITMASKS["invalid_ice_mask_applied"],
        ],
        # at_least_one_day_during_month_has_spatial_interpolation
        [
            CDR_SEAICE_CONC_QA_FLAG_DAILY_BITMASKS["spatial_interpolation_applied"],
            np.nan,
            np.nan,
        ],
        # at_least_one_day_during_month_has_temporal_interpolation
        [
            np.nan,
            np.nan,
            CDR_SEAICE_CONC_QA_FLAG_DAILY_BITMASKS["temporal_interpolation_applied"],
        ],
        # at_least_one_day_during_month_has_melt_detected
        [
            np.nan,
            CDR_SEAICE_CONC_QA_FLAG_DAILY_BITMASKS["start_of_melt_detected"],
            np.nan,
        ],
        # land flag.
        [np.nan, np.nan, np.nan],
    ]

    _mock_daily_melt_onset = [
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        [255, 255, 255],
        [
            255,
            255,
            255,
        ],
        [
            255,
            255,
            255,
        ],
        [
            255,
            255,
            255,
        ],
        [255, 61, 61],
        [255, 255, 255],
    ]

    _mock_daily_ds = xr.Dataset(
        data_vars=dict(
            cdr_seaice_conc=(("x", "time"), _mock_data),
            raw_nt_seaice_conc=(("x", "time"), _mock_data),
            raw_bt_seaice_conc=(("x", "time"), _mock_data),
            cdr_seaice_conc_qa_flag=(("x", "time"), _mock_daily_qa_fields),
            cdr_melt_onset_day=(("x", "time"), _mock_daily_melt_onset),
            surface_type_mask=(("x", "time"), _mock_surface_type_mask),
            filepaths=(
                ("time",),
                [Path("/tmp/foo.nc"), Path("/tmp/bar.nc"), Path("/tmp/baz.nc")],
            ),
            crs=(
                ("time"),
                ["a"] * 3,
                {
                    "long_name": "fake_NH_crs",
                },
            ),
        ),
        coords=dict(
            x=list(range(9)),
            # doy 60, 61, 62; need to be datetime64 to mimic xarray time
            time=np.array(
                [dt.date(2022, 3, 1), dt.date(2022, 3, 2), dt.date(2022, 3, 3)],
                dtype=np.datetime64,
            ),
        ),
    )

    _mock_daily_ds.attrs["year"] = 2022
    _mock_daily_ds.attrs["month"] = 3
    _mock_daily_ds.surface_type_mask.attrs = dict(
        flag_values=np.array([50, 75, 100, 200, 250], dtype=np.byte),
        flag_meanings="ocean lake polehole_mask coast land",
    )
    _mock_daily_ds.attrs["platform_id"] = "F17"

    return _mock_daily_ds


def test_monthly_ds(monkeypatch, tmpdir):
    _mock_daily_ds = _mock_daily_ds_for_month()

    # usually we require at least 20 days of data for a valid month. This mock
    # data is just 3 days in size, so we need to mock the
    # "check_min_days_for_valid_month" function.
    monkeypatch.setattr(
        intermediate_monthly,
        "check_min_days_for_valid_month",
        lambda *_args, **_kwargs: True,
    )

    _mock_oceanmask = xr.DataArray(
        [
            True,
            False,
            True,
            True,
        ],
        dims=("y",),
        coords=dict(y=list(range(4))),
    )
    monkeypatch.setattr(
        util, "get_ocean_mask", lambda *_args, **_kwargs: _mock_oceanmask
    )

    actual = make_intermediate_monthly_ds(
        daily_ds_for_month=_mock_daily_ds,
        platform_id="am2",
        hemisphere=NORTH,
        resolution="25",
    )

    # Test that the dataset only contains the variables we expect.
    expected_vars = sorted(
        [
            "cdr_seaice_conc_monthly",
            "cdr_seaice_conc_monthly_stdev",
            "cdr_melt_onset_day_monthly",
            "cdr_seaice_conc_monthly_qa_flag",
            "crs",
            "surface_type_mask",
        ]
    )
    actual_vars = sorted([str(var) for var in actual.keys()])

    assert expected_vars == actual_vars

    # Test that we can write out the data and read it back without changing it
    output_fp = tmpdir / "test.nc"
    actual.to_netcdf(output_fp)

    after_write = xr.open_dataset(output_fp)

    # Assert that all results are close to 0.01 (1% SIC).
    # TODO: should this be even closer? The max diff in the conc fields is:
    #   0.0033333327372868787
    xr.testing.assert_allclose(actual, after_write, atol=0.009)


def test__platform_id_for_month():
    assert "am2" == _platform_id_for_month(platform_ids=["am2", "am2", "am2", "am2"])

    assert "am2" == _platform_id_for_month(platform_ids=["F17", "F17", "am2", "am2"])

    assert "F17" == _platform_id_for_month(platform_ids=["F13", "F13", "F13", "F17"])

    assert "am2" == _platform_id_for_month(platform_ids=["F13", "F17", "am2"])
