import datetime as dt
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pytest
import xarray as xr

from seaice_ecdr.complete_daily_ecdr import get_ecdr_dir
from seaice_ecdr.monthly import (
    QA_OF_CDR_SEAICE_CONC_DAILY_FLAGS,
    QA_OF_CDR_SEAICE_CONC_MONTHLY_FLAGS,
    _calc_conc_monthly,
    _get_daily_complete_filepaths_for_month,
    _qa_field_has_flag,
    calc_qa_of_cdr_seaice_conc_monthly,
    check_min_days_for_valid_month,
)


def test__get_daily_complete_filepaths_for_month(fs):
    ecdr_data_dir = Path("/path/to/data/dir")
    fs.create_dir(ecdr_data_dir)
    complete_dir = get_ecdr_dir(ecdr_data_dir=ecdr_data_dir)
    year = 2022
    month = 3
    _fake_files_for_test_year_month = [
        complete_dir / "cdecdr_sic_psn12.5km_20220301_am2_v05r00.nc",
        complete_dir / "cdecdr_sic_psn12.5km_20220302_am2_v05r00.nc",
        complete_dir / "cdecdr_sic_psn12.5km_20220303_am2_v05r00.nc",
    ]
    _fake_files = [
        complete_dir / "cdecdr_sic_psn12.5km_20220201_am2_v05r00.nc",
        complete_dir / "cdecdr_sic_psn12.5km_20220202_am2_v05r00.nc",
        complete_dir / "cdecdr_sic_psn12.5km_20220203_am2_v05r00.nc",
        *_fake_files_for_test_year_month,
        complete_dir / "cdecdr_sic_psn12.5km_20220401_am2_v05r00.nc",
        complete_dir / "cdecdr_sic_psn12.5km_20220402_am2_v05r00.nc",
        complete_dir / "cdecdr_sic_psn12.5km_20220403_am2_v05r00.nc",
    ]
    for _file in _fake_files:
        fs.create_file(_file)

    actual = _get_daily_complete_filepaths_for_month(
        year=year,
        month=month,
        ecdr_data_dir=ecdr_data_dir,
        sat="am2",
    )

    assert sorted(_fake_files_for_test_year_month) == sorted(actual)


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


def test_calc_qa_of_cdr_seaice_conc_monthly():
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
            QA_OF_CDR_SEAICE_CONC_DAILY_FLAGS["valid_ice_mask_applied"],
            QA_OF_CDR_SEAICE_CONC_DAILY_FLAGS["valid_ice_mask_applied"],
            QA_OF_CDR_SEAICE_CONC_DAILY_FLAGS["valid_ice_mask_applied"],
        ],
        # at_least_one_day_during_month_has_spatial_interpolation
        [
            QA_OF_CDR_SEAICE_CONC_DAILY_FLAGS["spatial_interpolation_applied"],
            np.nan,
            np.nan,
        ],
        # at_least_one_day_during_month_has_temporal_interpolation
        [
            np.nan,
            np.nan,
            QA_OF_CDR_SEAICE_CONC_DAILY_FLAGS["temporal_interpolation_applied"],
        ],
        # at_least_one_day_during_month_has_melt_detected
        [np.nan, QA_OF_CDR_SEAICE_CONC_DAILY_FLAGS["start_of_melt_detected"], np.nan],
        # land flag.
        [np.nan, np.nan, np.nan],
    ]

    expected_flags = np.array(
        [
            QA_OF_CDR_SEAICE_CONC_MONTHLY_FLAGS["average_concentration_exceeds_0.15"]
            + QA_OF_CDR_SEAICE_CONC_MONTHLY_FLAGS[
                "at_least_half_the_days_have_sea_ice_conc_exceeds_0.15"
            ],
            QA_OF_CDR_SEAICE_CONC_MONTHLY_FLAGS["average_concentration_exceeds_0.15"]
            + QA_OF_CDR_SEAICE_CONC_MONTHLY_FLAGS[
                "at_least_half_the_days_have_sea_ice_conc_exceeds_0.15"
            ]
            + QA_OF_CDR_SEAICE_CONC_MONTHLY_FLAGS["average_concentration_exceeds_0.30"]
            + QA_OF_CDR_SEAICE_CONC_MONTHLY_FLAGS[
                "at_least_half_the_days_have_sea_ice_conc_exceeds_0.30"
            ],
            QA_OF_CDR_SEAICE_CONC_MONTHLY_FLAGS[
                "at_least_half_the_days_have_sea_ice_conc_exceeds_0.15"
            ],
            QA_OF_CDR_SEAICE_CONC_MONTHLY_FLAGS[
                "at_least_half_the_days_have_sea_ice_conc_exceeds_0.30"
            ]
            + QA_OF_CDR_SEAICE_CONC_MONTHLY_FLAGS["average_concentration_exceeds_0.15"]
            + QA_OF_CDR_SEAICE_CONC_MONTHLY_FLAGS[
                "at_least_half_the_days_have_sea_ice_conc_exceeds_0.15"
            ],
            QA_OF_CDR_SEAICE_CONC_MONTHLY_FLAGS["region_masked_by_ocean_climatology"],
            QA_OF_CDR_SEAICE_CONC_MONTHLY_FLAGS[
                "at_least_one_day_during_month_has_spatial_interpolation"
            ],
            QA_OF_CDR_SEAICE_CONC_MONTHLY_FLAGS[
                "at_least_one_day_during_month_has_temporal_interpolation"
            ],
            QA_OF_CDR_SEAICE_CONC_MONTHLY_FLAGS[
                "at_least_one_day_during_month_has_melt_detected"
            ]
            + QA_OF_CDR_SEAICE_CONC_MONTHLY_FLAGS["average_concentration_exceeds_0.15"]
            + QA_OF_CDR_SEAICE_CONC_MONTHLY_FLAGS["average_concentration_exceeds_0.30"],
            QA_OF_CDR_SEAICE_CONC_MONTHLY_FLAGS["fill_value"],
        ]
    )

    _mock_daily_ds = xr.Dataset(
        data_vars=dict(
            cdr_seaice_conc=(("x", "time"), _mock_data),
            qa_of_cdr_seaice_conc=(("x", "time"), _mock_daily_qa_fields),
        ),
        coords=dict(
            x=list(range(9)),
            time=[dt.date(2022, 3, 1), dt.date(2022, 3, 2), dt.date(2022, 3, 3)],
        ),
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
