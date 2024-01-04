import datetime as dt
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr
from pm_tb_data._types import NORTH

from seaice_ecdr import monthly
from seaice_ecdr.complete_daily_ecdr import get_ecdr_dir
from seaice_ecdr.monthly import (
    QA_OF_CDR_SEAICE_CONC_DAILY_BITMASKS,
    QA_OF_CDR_SEAICE_CONC_MONTHLY_BITMASKS,
    _calc_conc_monthly,
    _get_daily_complete_filepaths_for_month,
    _qa_field_has_flag,
    _sat_for_month,
    calc_melt_onset_day_cdr_seaice_conc_monthly,
    calc_qa_of_cdr_seaice_conc_monthly,
    calc_stdv_of_cdr_seaice_conc_monthly,
    calc_surface_type_mask_monthly,
    check_min_days_for_valid_month,
    make_monthly_ds,
)


def test__get_daily_complete_filepaths_for_month(fs):
    ecdr_data_dir = Path("/path/to/data/dir")
    fs.create_dir(ecdr_data_dir)
    complete_dir = get_ecdr_dir(ecdr_data_dir=ecdr_data_dir)
    year = 2022
    month = 3
    _fake_files_for_test_year_month_and_hemisphere = [
        complete_dir / "sic_psn12.5_20220301_am2_v05r00.nc",
        complete_dir / "sic_psn12.5_20220302_am2_v05r00.nc",
        complete_dir / "sic_psn12.5_20220303_am2_v05r00.nc",
    ]
    _fake_files = [
        complete_dir / "sic_psn12.5_20220201_am2_v05r00.nc",
        complete_dir / "sic_pss12.5_20220201_am2_v05r00.nc",
        complete_dir / "sic_psn12.5_20220202_am2_v05r00.nc",
        complete_dir / "sic_pss12.5_20220202_am2_v05r00.nc",
        complete_dir / "sic_psn12.5_20220203_am2_v05r00.nc",
        complete_dir / "sic_pss12.5_20220203_am2_v05r00.nc",
        complete_dir / "sic_pss12.5_20220301_am2_v05r00.nc",
        complete_dir / "sic_pss12.5_20220302_am2_v05r00.nc",
        complete_dir / "sic_pss12.5_20220303_am2_v05r00.nc",
        *_fake_files_for_test_year_month_and_hemisphere,
        complete_dir / "sic_psn12.5_20220401_am2_v05r00.nc",
        complete_dir / "sic_pss12.5_20220401_am2_v05r00.nc",
        complete_dir / "sic_psn12.5_20220402_am2_v05r00.nc",
        complete_dir / "sic_pss12.5_20220402_am2_v05r00.nc",
        complete_dir / "sic_psn12.5_20220403_am2_v05r00.nc",
        complete_dir / "sic_pss12.5_20220403_am2_v05r00.nc",
    ]
    for _file in _fake_files:
        fs.create_file(_file)

    actual = _get_daily_complete_filepaths_for_month(
        year=year,
        month=month,
        ecdr_data_dir=ecdr_data_dir,
        resolution="12.5",
        hemisphere=NORTH,
    )

    assert sorted(_fake_files_for_test_year_month_and_hemisphere) == sorted(actual)


def test_check_min_day_for_valid_month():
    def _mock_daily_ds_for_month(num_days: int) -> xr.Dataset:
        return xr.Dataset(
            data_vars=dict(time=[dt.date(2022, 3, x) for x in range(1, num_days + 1)])
        )

    # Check that no error is raised for AMSR2, full month's worth of data
    check_min_days_for_valid_month(
        daily_ds_for_month=_mock_daily_ds_for_month(31),
        sat="am2",
    )

    # Check that an error is raised for AMSR2, not a full month's worth of data
    with pytest.raises(RuntimeError):
        check_min_days_for_valid_month(
            daily_ds_for_month=_mock_daily_ds_for_month(19),
            sat="am2",
        )

    # Check that an error is not raised for n07, with modified min worth of data
    check_min_days_for_valid_month(
        daily_ds_for_month=_mock_daily_ds_for_month(10),
        sat="n07",
    )

    # Check that an error is raised for n07, not a full month's worth of data
    with pytest.raises(RuntimeError):
        check_min_days_for_valid_month(
            daily_ds_for_month=_mock_daily_ds_for_month(9),
            sat="n07",
        )


def test__qa_field_has_flag():
    flag_1 = 1
    flag_2 = 2
    flag_3 = 4
    mock_qa_field = xr.DataArray(
        [flag_1, (flag_1 | flag_2), (flag_1 | flag_2 | flag_3)]
    )

    # flag 1
    expected = xr.DataArray([True, True, True])
    actual = _qa_field_has_flag(
        qa_field=mock_qa_field,
        flag_value=flag_1,
    )
    npt.assert_array_equal(expected, actual)

    # flag 2
    expected = xr.DataArray([False, True, True])
    actual = _qa_field_has_flag(
        qa_field=mock_qa_field,
        flag_value=flag_2,
    )
    npt.assert_array_equal(expected, actual)

    # flag 3
    expected = xr.DataArray([False, False, True])
    actual = _qa_field_has_flag(
        qa_field=mock_qa_field,
        flag_value=flag_3,
    )
    npt.assert_array_equal(expected, actual)


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
        # region_masked_by_ocean_climatology
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
        # region_masked_by_ocean_climatology
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
        # region_masked_by_ocean_climatology
        [
            QA_OF_CDR_SEAICE_CONC_DAILY_BITMASKS["valid_ice_mask_applied"],
            QA_OF_CDR_SEAICE_CONC_DAILY_BITMASKS["valid_ice_mask_applied"],
            QA_OF_CDR_SEAICE_CONC_DAILY_BITMASKS["valid_ice_mask_applied"],
        ],
        # at_least_one_day_during_month_has_spatial_interpolation
        [
            QA_OF_CDR_SEAICE_CONC_DAILY_BITMASKS["spatial_interpolation_applied"],
            np.nan,
            np.nan,
        ],
        # at_least_one_day_during_month_has_temporal_interpolation
        [
            np.nan,
            np.nan,
            QA_OF_CDR_SEAICE_CONC_DAILY_BITMASKS["temporal_interpolation_applied"],
        ],
        # at_least_one_day_during_month_has_melt_detected
        [
            np.nan,
            QA_OF_CDR_SEAICE_CONC_DAILY_BITMASKS["start_of_melt_detected"],
            np.nan,
        ],
        # land flag.
        [np.nan, np.nan, np.nan],
    ]

    _mock_daily_melt_onset = [
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
        [
            np.nan,
            np.nan,
            np.nan,
        ],
        [
            np.nan,
            np.nan,
            np.nan,
        ],
        [
            np.nan,
            np.nan,
            np.nan,
        ],
        [np.nan, 61, 61],
        [np.nan, np.nan, np.nan],
    ]

    _mock_daily_ds = xr.Dataset(
        data_vars=dict(
            cdr_seaice_conc=(("x", "time"), _mock_data),
            raw_nt_seaice_conc=(("x", "time"), _mock_data),
            raw_bt_seaice_conc=(("x", "time"), _mock_data),
            qa_of_cdr_seaice_conc=(("x", "time"), _mock_daily_qa_fields),
            melt_onset_day_cdr_seaice_conc=(("x", "time"), _mock_daily_melt_onset),
            surface_type_mask=(("x", "time"), _mock_surface_type_mask),
            filepaths=(
                ("time",),
                [Path("/tmp/foo.nc"), Path("/tmp/bar.nc"), Path("/tmp/baz.nc")],
            ),
            crs=(("time"), ["a"] * 3),
        ),
        coords=dict(
            x=list(range(9)),
            # doy 60, 61, 62
            time=[dt.date(2022, 3, 1), dt.date(2022, 3, 2), dt.date(2022, 3, 3)],
        ),
    )

    _mock_daily_ds.attrs["year"] = 2022
    _mock_daily_ds.attrs["month"] = 3
    _mock_daily_ds.surface_type_mask.attrs = dict(
        flag_values=np.array([50, 75, 100, 200, 250], dtype=np.byte),
        flag_meanings="ocean lake polehole_mask coast land",
    )

    return _mock_daily_ds


def test_calc_qa_of_cdr_seaice_conc_monthly():
    _mock_daily_ds = _mock_daily_ds_for_month()
    expected_flags = np.array(
        [
            QA_OF_CDR_SEAICE_CONC_MONTHLY_BITMASKS["average_concentration_exceeds_0.15"]
            + QA_OF_CDR_SEAICE_CONC_MONTHLY_BITMASKS[
                "at_least_half_the_days_have_sea_ice_conc_exceeds_0.15"
            ],
            QA_OF_CDR_SEAICE_CONC_MONTHLY_BITMASKS["average_concentration_exceeds_0.15"]
            + QA_OF_CDR_SEAICE_CONC_MONTHLY_BITMASKS[
                "at_least_half_the_days_have_sea_ice_conc_exceeds_0.15"
            ]
            + QA_OF_CDR_SEAICE_CONC_MONTHLY_BITMASKS[
                "average_concentration_exceeds_0.30"
            ]
            + QA_OF_CDR_SEAICE_CONC_MONTHLY_BITMASKS[
                "at_least_half_the_days_have_sea_ice_conc_exceeds_0.30"
            ],
            QA_OF_CDR_SEAICE_CONC_MONTHLY_BITMASKS[
                "at_least_half_the_days_have_sea_ice_conc_exceeds_0.15"
            ],
            QA_OF_CDR_SEAICE_CONC_MONTHLY_BITMASKS[
                "at_least_half_the_days_have_sea_ice_conc_exceeds_0.30"
            ]
            + QA_OF_CDR_SEAICE_CONC_MONTHLY_BITMASKS[
                "average_concentration_exceeds_0.15"
            ]
            + QA_OF_CDR_SEAICE_CONC_MONTHLY_BITMASKS[
                "at_least_half_the_days_have_sea_ice_conc_exceeds_0.15"
            ],
            QA_OF_CDR_SEAICE_CONC_MONTHLY_BITMASKS[
                "region_masked_by_ocean_climatology"
            ],
            QA_OF_CDR_SEAICE_CONC_MONTHLY_BITMASKS[
                "at_least_one_day_during_month_has_spatial_interpolation"
            ],
            QA_OF_CDR_SEAICE_CONC_MONTHLY_BITMASKS[
                "at_least_one_day_during_month_has_temporal_interpolation"
            ],
            QA_OF_CDR_SEAICE_CONC_MONTHLY_BITMASKS[
                "at_least_one_day_during_month_has_melt_detected"
            ]
            + QA_OF_CDR_SEAICE_CONC_MONTHLY_BITMASKS[
                "average_concentration_exceeds_0.15"
            ]
            + QA_OF_CDR_SEAICE_CONC_MONTHLY_BITMASKS[
                "average_concentration_exceeds_0.30"
            ],
            0,
        ]
    )

    _mean_daily_conc = _mock_daily_ds.cdr_seaice_conc.mean(dim="time")
    actual = calc_qa_of_cdr_seaice_conc_monthly(
        daily_ds_for_month=_mock_daily_ds,
        cdr_seaice_conc_monthly=_mean_daily_conc,
    )

    npt.assert_array_equal(expected_flags, actual.values)


def test__calc_conc_monthly():
    # time ->
    pixel_one = np.array([0.12, 0.15, 0.23])
    pixel_two = np.array([0.46, 0.55, 0.54])
    pixel_three = np.array([0.89, 0.99, 0.89])
    pixel_four = np.array([0.89, np.nan, 0.89])
    _mock_data = [
        pixel_one,
        pixel_two,
        pixel_three,
        pixel_four,
    ]

    mock_daily_conc = xr.DataArray(
        data=_mock_data,
        dims=["y", "time"],
        coords=dict(
            time=[dt.date(2022, 3, 1), dt.date(2022, 3, 2), dt.date(2022, 3, 3)],
            y=list(range(4)),
        ),
    )

    actual = _calc_conc_monthly(
        daily_conc_for_month=mock_daily_conc,
        long_name="mock_long_name",
        name="mock_name",
    )

    assert actual.attrs["long_name"] == "mock_long_name"
    npt.assert_array_equal(
        actual.values,
        np.array(
            [
                np.nanmean(pixel_one),
                np.nanmean(pixel_two),
                np.nanmean(pixel_three),
                np.nanmean(pixel_four),
            ]
        ),
    )


def test_calc_stdv_of_cdr_seaice_conc_monthly():
    # time ->
    pixel_one = np.array([0.12, 0.15, 0.23])
    pixel_two = np.array([0.46, 0.55, 0.54])
    pixel_three = np.array([0.89, 0.99, 0.89])
    pixel_four = np.array([0.89, np.nan, 0.89])
    _mock_data = [
        pixel_one,
        pixel_two,
        pixel_three,
        pixel_four,
    ]

    mock_daily_conc = xr.DataArray(
        data=_mock_data,
        dims=["y", "time"],
        coords=dict(
            time=[dt.date(2022, 3, 1), dt.date(2022, 3, 2), dt.date(2022, 3, 3)],
            y=list(range(4)),
        ),
    )

    actual = calc_stdv_of_cdr_seaice_conc_monthly(
        daily_cdr_seaice_conc=mock_daily_conc,
    )

    assert (
        actual.attrs["long_name"]
        == "Passive Microwave Monthly Northern Hemisphere Sea Ice Concentration Source Estimated Standard Deviation"
    )

    npt.assert_array_equal(
        actual.values,
        np.array(
            [
                np.nanstd(pixel_one, ddof=1),
                np.nanstd(pixel_two, ddof=1),
                np.nanstd(pixel_three, ddof=1),
                np.nanstd(pixel_four, ddof=1),
            ]
        ),
    )


def test_calc_melt_onset_day_cdr_seaice_conc_monthly():
    # time ->
    pixel_one = np.array([60, 60, 60])
    pixel_two = np.array([np.nan, 200, 200])
    pixel_three = np.array([np.nan, np.nan, 244])
    pixel_four = np.array([np.nan, np.nan, np.nan])
    pixel_five = np.array([0, 0, 0])
    _mock_data = [
        pixel_one,
        pixel_two,
        pixel_three,
        pixel_four,
        pixel_five,
    ]

    mock_daily_melt = xr.DataArray(
        data=_mock_data,
        dims=["y", "time"],
        coords=dict(
            time=[dt.date(2022, 3, 1), dt.date(2022, 3, 2), dt.date(2022, 3, 3)],
            y=list(range(5)),
        ),
    )

    actual = calc_melt_onset_day_cdr_seaice_conc_monthly(
        daily_melt_onset_for_month=mock_daily_melt,
    )

    assert actual.long_name == "Monthly Day of Snow Melt Onset Over Sea Ice"
    npt.assert_array_equal(
        actual.values,
        np.array(
            [
                60,
                200,
                244,
                np.nan,
                0,
            ]
        ),
    )


def test_monthly_ds(monkeypatch, tmpdir):
    _mock_daily_ds = _mock_daily_ds_for_month()

    # usually we require at least 20 days of data for a valid month. This mock
    # data is just 3 days in size, so we need to mock the
    # "check_min_days_for_valid_month" function.
    monkeypatch.setattr(
        monthly, "check_min_days_for_valid_month", lambda *_args, **_kwargs: True
    )

    actual = make_monthly_ds(
        daily_ds_for_month=_mock_daily_ds,
        sat="am2",
        hemisphere=NORTH,
    )

    # Test that the dataset only contains the variables we expect.
    expected_vars = sorted(
        [
            "cdr_seaice_conc_monthly",
            "stdv_of_cdr_seaice_conc_monthly",
            "melt_onset_day_cdr_seaice_conc_monthly",
            "qa_of_cdr_seaice_conc_monthly",
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


def test__sat_for_month():
    assert "am2" == _sat_for_month(sats=["am2", "am2", "am2", "am2"])

    assert "am2" == _sat_for_month(sats=["F17", "F17", "am2", "am2"])

    assert "F17" == _sat_for_month(sats=["F13", "F13", "F13", "F17"])

    assert "am2" == _sat_for_month(sats=["F13", "F17", "am2"])


def test_calc_surface_mask_monthly():
    # Each row is one pixel through time.
    # time ->
    _mock_surface_type_data = [
        # all ocean
        [50, 50, 50],
        # Second element in time marked pole hole
        [50, 100, 50],
        # All pole-hole thru time
        [100, 100, 100],
        # Coast is consistent thru time:
        [200, 200, 200],
        # Land is consistent thru time:
        [250, 250, 250],
        # Lake is consistent thru time:
        [75, 75, 75],
    ]

    mock_daily_ds_for_month = xr.Dataset(
        data_vars=dict(
            surface_type_mask=(("x", "time"), _mock_surface_type_data),
        ),
        coords=dict(
            x=list(range(6)),
            # doy 60, 61, 62
            time=[dt.date(2022, 3, 1), dt.date(2022, 3, 2), dt.date(2022, 3, 3)],
        ),
    )

    mock_daily_ds_for_month.surface_type_mask.attrs = dict(
        flag_values=np.array([50, 75, 100, 200, 250], dtype=np.byte),
        flag_meanings="ocean lake polehole_mask coast land",
    )

    actual = calc_surface_type_mask_monthly(
        daily_ds_for_month=mock_daily_ds_for_month,
    )

    expected_surface_type_data = [
        # all ocean
        50,
        # Second element in time marked pole hole
        100,
        # All pole-hole thru time
        100,
        # Coast is consistent thru time:
        200,
        # Land is consistent thru time:
        250,
        # Lake is consistent thru time:
        75,
    ]

    assert (actual.data == expected_surface_type_data).all()
