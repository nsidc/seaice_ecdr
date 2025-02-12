import datetime as dt
from pathlib import Path

import numpy as np
import numpy.testing as nptesting  # Note: npt is numpy.typing
import pytest
import xarray as xr
from loguru import logger
from pm_tb_data._types import NORTH

from seaice_ecdr import util
from seaice_ecdr.constants import ECDR_PRODUCT_VERSION
from seaice_ecdr.intermediate_daily import get_ecdr_dir
from seaice_ecdr.intermediate_monthly import (
    CDR_SEAICE_CONC_MONTHLY_QA_FLAG_BITMASKS_NORTH,
    CDR_SEAICE_CONC_QA_FLAG_DAILY_BITMASKS,
    _get_daily_complete_filepaths_for_month,
    _platform_id_for_month,
    _qa_field_has_flag,
    calc_cdr_melt_onset_day_monthly,
    calc_cdr_seaice_conc_monthly,
    calc_cdr_seaice_conc_monthly_qa_flag,
    calc_cdr_seaice_conc_monthly_stdev,
    calc_surface_type_mask_monthly,
    check_min_days_for_valid_month,
)


def test__get_daily_complete_filepaths_for_month(fs):
    intermediate_output_dir = Path("/path/to/data/dir/intermediate")
    fs.create_dir(intermediate_output_dir)
    nh_intermediate_dir = get_ecdr_dir(
        intermediate_output_dir=intermediate_output_dir / "north",
        year=2022,
        is_nrt=False,
    )
    sh_intermediate_dir = get_ecdr_dir(
        intermediate_output_dir=intermediate_output_dir / "south",
        year=2022,
        is_nrt=False,
    )
    year = 2022
    month = 3
    _fake_files_for_test_year_month_and_hemisphere = [
        nh_intermediate_dir / f"sic_psn25_20220301_F17_{ECDR_PRODUCT_VERSION}.nc",
        nh_intermediate_dir / f"sic_psn25_20220302_F17_{ECDR_PRODUCT_VERSION}.nc",
        nh_intermediate_dir / f"sic_psn25_20220303_F17_{ECDR_PRODUCT_VERSION}.nc",
    ]
    _fake_files = [
        nh_intermediate_dir / f"sic_psn25_20220201_F17_{ECDR_PRODUCT_VERSION}.nc",
        nh_intermediate_dir / f"sic_pss25_20220201_F17_{ECDR_PRODUCT_VERSION}.nc",
        nh_intermediate_dir / f"sic_psn25_20220202_F17_{ECDR_PRODUCT_VERSION}.nc",
        nh_intermediate_dir / f"sic_pss25_20220202_F17_{ECDR_PRODUCT_VERSION}.nc",
        nh_intermediate_dir / f"sic_psn25_20220203_F17_{ECDR_PRODUCT_VERSION}.nc",
        nh_intermediate_dir / f"sic_pss25_20220203_F17_{ECDR_PRODUCT_VERSION}.nc",
        nh_intermediate_dir / f"sic_pss25_20220301_F17_{ECDR_PRODUCT_VERSION}.nc",
        nh_intermediate_dir / f"sic_pss25_20220302_F17_{ECDR_PRODUCT_VERSION}.nc",
        nh_intermediate_dir / f"sic_pss25_20220303_F17_{ECDR_PRODUCT_VERSION}.nc",
        *_fake_files_for_test_year_month_and_hemisphere,
        nh_intermediate_dir / f"sic_psn25_20220401_F17_{ECDR_PRODUCT_VERSION}.nc",
        sh_intermediate_dir / f"sic_pss25_20220401_F17_{ECDR_PRODUCT_VERSION}.nc",
        nh_intermediate_dir / f"sic_psn25_20220402_F17_{ECDR_PRODUCT_VERSION}.nc",
        sh_intermediate_dir / f"sic_pss25_20220402_F17_{ECDR_PRODUCT_VERSION}.nc",
        nh_intermediate_dir / f"sic_psn25_20220403_F17_{ECDR_PRODUCT_VERSION}.nc",
        sh_intermediate_dir / f"sic_pss25_20220403_F17_{ECDR_PRODUCT_VERSION}.nc",
    ]
    for _file in _fake_files:
        logger.info(f"creating {_file=}")
        fs.create_file(_file)

    actual = _get_daily_complete_filepaths_for_month(
        year=year,
        month=month,
        intermediate_output_dir=intermediate_output_dir / NORTH,
        resolution="25",
        hemisphere=NORTH,
        is_nrt=False,
    )

    assert sorted(_fake_files_for_test_year_month_and_hemisphere) == sorted(actual)


def test_check_min_day_for_valid_month():
    def _mock_daily_ds_for_month(num_days: int) -> xr.Dataset:
        return xr.Dataset(
            data_vars=dict(
                time=np.array(
                    [dt.date(2022, 3, x) for x in range(1, num_days + 1)],
                    dtype=np.datetime64,
                )
            ),
            attrs={"platform_id": "F17"},
        )

    # Check that no error is raised for AMSR2, full month's worth of data
    check_min_days_for_valid_month(
        daily_ds_for_month=_mock_daily_ds_for_month(31),
        platform_id="am2",
    )

    # Check that an error is raised for AMSR2, not a full month's worth of data
    with pytest.raises(RuntimeError):
        check_min_days_for_valid_month(
            daily_ds_for_month=_mock_daily_ds_for_month(19),
            platform_id="am2",
        )

    # Check that an error is not raised for n07, with modified min worth of data
    check_min_days_for_valid_month(
        daily_ds_for_month=_mock_daily_ds_for_month(10),
        platform_id="n07",
    )

    # Check that an error is raised for n07, not a full month's worth of data
    with pytest.raises(RuntimeError):
        check_min_days_for_valid_month(
            daily_ds_for_month=_mock_daily_ds_for_month(9),
            platform_id="n07",
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
    nptesting.assert_array_equal(expected, actual)

    # flag 2
    expected = xr.DataArray([False, True, True])
    actual = _qa_field_has_flag(
        qa_field=mock_qa_field,
        flag_value=flag_2,
    )
    nptesting.assert_array_equal(expected, actual)

    # flag 3
    expected = xr.DataArray([False, False, True])
    actual = _qa_field_has_flag(
        qa_field=mock_qa_field,
        flag_value=flag_3,
    )
    nptesting.assert_array_equal(expected, actual)


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


def test_calc_cdr_seaice_conc_monthly_qa_flag():
    _mock_daily_ds = _mock_daily_ds_for_month()
    CDR_SEAICE_CONC_MONTHLY_QA_FLAG_BITMASKS = (
        CDR_SEAICE_CONC_MONTHLY_QA_FLAG_BITMASKS_NORTH
    )
    expected_flags = np.array(
        [
            CDR_SEAICE_CONC_MONTHLY_QA_FLAG_BITMASKS[
                "average_concentration_exceeds_0.15"
            ]
            + CDR_SEAICE_CONC_MONTHLY_QA_FLAG_BITMASKS[
                "at_least_half_the_days_have_sea_ice_conc_exceeds_0.15"
            ],
            CDR_SEAICE_CONC_MONTHLY_QA_FLAG_BITMASKS[
                "average_concentration_exceeds_0.15"
            ]
            + CDR_SEAICE_CONC_MONTHLY_QA_FLAG_BITMASKS[
                "at_least_half_the_days_have_sea_ice_conc_exceeds_0.15"
            ]
            + CDR_SEAICE_CONC_MONTHLY_QA_FLAG_BITMASKS[
                "average_concentration_exceeds_0.30"
            ]
            + CDR_SEAICE_CONC_MONTHLY_QA_FLAG_BITMASKS[
                "at_least_half_the_days_have_sea_ice_conc_exceeds_0.30"
            ],
            CDR_SEAICE_CONC_MONTHLY_QA_FLAG_BITMASKS[
                "at_least_half_the_days_have_sea_ice_conc_exceeds_0.15"
            ],
            CDR_SEAICE_CONC_MONTHLY_QA_FLAG_BITMASKS[
                "at_least_half_the_days_have_sea_ice_conc_exceeds_0.30"
            ]
            + CDR_SEAICE_CONC_MONTHLY_QA_FLAG_BITMASKS[
                "average_concentration_exceeds_0.15"
            ]
            + CDR_SEAICE_CONC_MONTHLY_QA_FLAG_BITMASKS[
                "at_least_half_the_days_have_sea_ice_conc_exceeds_0.15"
            ],
            CDR_SEAICE_CONC_MONTHLY_QA_FLAG_BITMASKS["invalid_ice_mask_applied"],
            CDR_SEAICE_CONC_MONTHLY_QA_FLAG_BITMASKS[
                "at_least_one_day_during_month_has_spatial_interpolation"
            ],
            CDR_SEAICE_CONC_MONTHLY_QA_FLAG_BITMASKS[
                "at_least_one_day_during_month_has_temporal_interpolation"
            ],
            CDR_SEAICE_CONC_MONTHLY_QA_FLAG_BITMASKS[
                "at_least_one_day_during_month_has_melt_detected"
            ]
            + CDR_SEAICE_CONC_MONTHLY_QA_FLAG_BITMASKS[
                "average_concentration_exceeds_0.15"
            ]
            + CDR_SEAICE_CONC_MONTHLY_QA_FLAG_BITMASKS[
                "average_concentration_exceeds_0.30"
            ],
            0,
        ]
    )

    _mean_daily_conc = _mock_daily_ds.cdr_seaice_conc.mean(dim="time")
    actual = calc_cdr_seaice_conc_monthly_qa_flag(
        daily_ds_for_month=_mock_daily_ds,
        cdr_seaice_conc_monthly=_mean_daily_conc,
        hemisphere="north",
    )

    nptesting.assert_array_equal(expected_flags, actual.values)


def test__calc_conc_monthly(monkeypatch):
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
            time=np.array(
                [dt.date(2022, 3, 1), dt.date(2022, 3, 2), dt.date(2022, 3, 3)],
                dtype=np.datetime64,
            ),
            y=list(range(4)),
        ),
    )

    mock_daily_ds = xr.Dataset(
        data_vars=dict(
            cdr_seaice_conc=mock_daily_conc,
            crs=(
                ("time"),
                ["a"] * 3,
                {
                    "long_name": "fake_NH_crs",
                },
            ),
        ),
        attrs=dict(
            platform_id="F17",
        ),
    )

    # Mock the ocean mask used to determine if there are any missing pixels.
    _mock_oceanmask = xr.DataArray(
        [
            True,
            True,
            True,
            True,
        ],
        dims=("y",),
        coords=dict(y=list(range(4))),
    )
    monkeypatch.setattr(
        util, "get_ocean_mask", lambda *_args, **_kwargs: _mock_oceanmask
    )

    actual = calc_cdr_seaice_conc_monthly(
        daily_ds_for_month=mock_daily_ds,
        hemisphere="north",
        resolution="25",
        ancillary_source="CDRv5",
    )

    nptesting.assert_array_equal(
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


def test_calc_cdr_seaice_conc_monthly_stdev():
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

    actual = calc_cdr_seaice_conc_monthly_stdev(
        daily_cdr_seaice_conc=mock_daily_conc,
    )

    assert (
        actual.attrs["long_name"]
        == "NOAA/NSIDC CDR of Passive Microwave Monthly Northern Hemisphere Sea Ice Concentration Source Estimated Standard Deviation"
    )

    nptesting.assert_array_equal(
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


def test_calc_cdr_melt_onset_day_monthly():
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
            time=[
                np.datetime64(date)
                for date in [
                    dt.date(2022, 3, 1),
                    dt.date(2022, 3, 2),
                    dt.date(2022, 3, 3),
                ]
            ],
            y=list(range(5)),
        ),
    )

    actual = calc_cdr_melt_onset_day_monthly(
        daily_melt_onset_for_month=mock_daily_melt,
    )

    assert (
        actual.long_name == "NOAA/NSIDC CDR Monthly Day of Snow Melt Onset Over Sea Ice"
    )
    nptesting.assert_array_equal(
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


def test__platform_id_for_month():
    assert "am2" == _platform_id_for_month(platform_ids=["am2", "am2", "am2", "am2"])

    assert "am2" == _platform_id_for_month(platform_ids=["F17", "F17", "am2", "am2"])

    assert "F17" == _platform_id_for_month(platform_ids=["F13", "F13", "F13", "F17"])

    assert "am2" == _platform_id_for_month(platform_ids=["F13", "F17", "am2"])


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
        hemisphere=NORTH,
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
