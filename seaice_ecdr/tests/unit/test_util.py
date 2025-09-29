import datetime as dt
from pathlib import Path
from typing import Final

import numpy as np
import pytest
import xarray as xr
from pm_tb_data._types import NORTH, SOUTH

from seaice_ecdr import util
from seaice_ecdr.constants import ECDR_NRT_PRODUCT_VERSION, ECDR_PRODUCT_VERSION
from seaice_ecdr.multiprocess_intermediate_daily import get_dates_by_year
from seaice_ecdr.platforms.models import SUPPORTED_PLATFORM_ID
from seaice_ecdr.util import (
    clean_outputs_for_date_range,
    date_range,
    find_standard_monthly_netcdf_files,
    get_complete_output_dir,
    get_intermediate_output_dir,
    get_num_missing_pixels,
    nrt_daily_filename,
    nrt_monthly_filename,
    platform_id_from_filename,
    raise_error_for_dates,
    standard_daily_aggregate_filename,
    standard_daily_filename,
    standard_monthly_aggregate_filename,
    standard_monthly_filename,
)


def test_daily_filename_north():
    expected = f"sic_psn12.5_20210101_am2_{ECDR_PRODUCT_VERSION}.nc"

    actual = standard_daily_filename(
        hemisphere=NORTH, resolution="12.5", platform_id="am2", date=dt.date(2021, 1, 1)
    )

    assert actual == expected


def test_daily_filename_south():
    expected = f"sic_pss12.5_20210101_am2_{ECDR_PRODUCT_VERSION}.nc"

    actual = standard_daily_filename(
        hemisphere=SOUTH, resolution="12.5", platform_id="am2", date=dt.date(2021, 1, 1)
    )

    assert actual == expected


def test_nrt_daily_filename():
    expected = f"sic_psn12.5_20210101_am2_icdr_{ECDR_NRT_PRODUCT_VERSION}.nc"

    actual = nrt_daily_filename(
        hemisphere=NORTH, resolution="12.5", platform_id="am2", date=dt.date(2021, 1, 1)
    )

    assert actual == expected


def test_nrt_monthly_filename():
    expected = f"sic_psn25_202409_F17_icdr_{ECDR_NRT_PRODUCT_VERSION}.nc"

    actual = nrt_monthly_filename(
        hemisphere=NORTH,
        resolution="25",
        platform_id="F17",
        year=2024,
        month=9,
    )

    assert actual == expected


def test_daily_aggregate_filename():
    expected = f"sic_psn12.5_20210101-20211231_{ECDR_PRODUCT_VERSION}.nc"

    actual = standard_daily_aggregate_filename(
        hemisphere=NORTH,
        resolution="12.5",
        start_date=dt.date(2021, 1, 1),
        end_date=dt.date(2021, 12, 31),
    )

    assert actual == expected


def test_monthly_filename_north():
    expected = f"sic_psn12.5_202101_am2_{ECDR_PRODUCT_VERSION}.nc"

    actual = standard_monthly_filename(
        hemisphere=NORTH,
        resolution="12.5",
        platform_id="am2",
        year=2021,
        month=1,
    )

    assert actual == expected


def test_monthly_filename_south():
    expected = f"sic_pss12.5_202101_am2_{ECDR_PRODUCT_VERSION}.nc"

    actual = standard_monthly_filename(
        hemisphere=SOUTH,
        resolution="12.5",
        platform_id="am2",
        year=2021,
        month=1,
    )

    assert actual == expected


def test_monthly_aggregate_filename():
    expected = f"sic_pss12.5_202101-202112_{ECDR_PRODUCT_VERSION}.nc"

    actual = standard_monthly_aggregate_filename(
        hemisphere=SOUTH,
        resolution="12.5",
        start_year=2021,
        start_month=1,
        end_year=2021,
        end_month=12,
    )

    assert actual == expected


def test_daily_platform_id_from_filename():
    expected_platform_id: Final = "am2"
    fn = standard_daily_filename(
        hemisphere=NORTH,
        resolution="12.5",
        platform_id=expected_platform_id,
        date=dt.date(2021, 1, 1),
    )

    actual_platform_id = platform_id_from_filename(fn)

    assert expected_platform_id == actual_platform_id


def test_monthly_platform_id_from_filename():
    expected_platform_id: Final = "F17"
    fn = standard_monthly_filename(
        hemisphere=SOUTH,
        resolution="12.5",
        platform_id=expected_platform_id,
        year=2021,
        month=1,
    )

    actual_platform_id = platform_id_from_filename(fn)

    assert expected_platform_id == actual_platform_id


def test_daily_platform_id_from_daily_nrt_filename():
    expected_platform_id: Final = "F17"
    fn = nrt_daily_filename(
        hemisphere=SOUTH,
        resolution="25",
        platform_id=expected_platform_id,
        date=dt.date(2021, 1, 1),
    )

    actual_platform_id = platform_id_from_filename(fn)

    assert expected_platform_id == actual_platform_id


def test_daily_platform_id_from_monthly_nrt_filename():
    expected_platform_id: Final = "F17"
    fn = nrt_monthly_filename(
        hemisphere=SOUTH,
        resolution="25",
        platform_id=expected_platform_id,
        year=2024,
        month=9,
    )

    actual_platform_id = platform_id_from_filename(fn)

    assert expected_platform_id == actual_platform_id


def test_find_standard_monthly_netcdf_files_platform_wildcard(fs):
    monthly_output_dir = Path("/path/to/data/dir/monthly")
    fs.create_dir(monthly_output_dir)
    platform_ids: list[SUPPORTED_PLATFORM_ID] = ["am2", "F17"]
    for platform_id in platform_ids:
        fake_monthly_filename = standard_monthly_filename(
            hemisphere=NORTH,
            resolution="25",
            platform_id=platform_id,
            year=2021,
            month=1,
        )
        fake_monthly_filepath = monthly_output_dir / fake_monthly_filename
        fs.create_file(fake_monthly_filepath)

    found_files_wildcard_platform_id = find_standard_monthly_netcdf_files(
        search_dir=monthly_output_dir,
        hemisphere=NORTH,
        resolution="25",
        platform_id="*",
        year=2021,
        month=1,
    )

    assert len(found_files_wildcard_platform_id) == 2


def test_find_standard_monthly_netcdf_files_yearmonth_wildcard(fs):
    monthly_output_dir = Path("/path/to/data/dir/monthly")
    fs.create_dir(monthly_output_dir)
    for year, month in [(2022, 1), (2022, 2)]:
        fake_monthly_filename = standard_monthly_filename(
            hemisphere=NORTH,
            resolution="25",
            platform_id="F17",
            year=year,
            month=month,
        )
        fake_monthly_filepath = monthly_output_dir / fake_monthly_filename
        fs.create_file(fake_monthly_filepath)

    found_files_wildcard_platform_id = find_standard_monthly_netcdf_files(
        search_dir=monthly_output_dir,
        hemisphere=NORTH,
        resolution="25",
        platform_id="F17",
        year="*",
        month="*",
    )

    assert len(found_files_wildcard_platform_id) == 2


def test_date_range():
    start_date = dt.date(2021, 1, 2)
    end_date = dt.date(2021, 1, 5)
    expected = [
        start_date,
        dt.date(2021, 1, 3),
        dt.date(2021, 1, 4),
        end_date,
    ]
    actual = list(date_range(start_date=start_date, end_date=end_date))

    assert expected == actual


def test_get_dates_by_year():
    actual = get_dates_by_year(
        [
            dt.date(2021, 1, 3),
            dt.date(2021, 1, 2),
            dt.date(2022, 1, 1),
            dt.date(1997, 3, 2),
            dt.date(1997, 4, 15),
            dt.date(2022, 1, 2),
        ]
    )

    expected = [
        [
            dt.date(1997, 3, 2),
            dt.date(1997, 4, 15),
        ],
        [
            dt.date(2021, 1, 2),
            dt.date(2021, 1, 3),
        ],
        [
            dt.date(2022, 1, 1),
            dt.date(2022, 1, 2),
        ],
    ]

    assert actual == expected


def test_get_num_missing_pixels(monkeypatch):
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

    _mock_sic = xr.DataArray(
        [
            0.99,
            np.nan,
            0.25,
            np.nan,
        ],
        dims=("y",),
        coords=dict(y=list(range(4))),
    )

    detected_missing = get_num_missing_pixels(
        seaice_conc_var=_mock_sic,
        hemisphere="north",
        resolution="12.5",
    )

    assert detected_missing == 1


def test_raise_error_for_dates():
    # If no dates are passed, no error should be raised.
    raise_error_for_dates(error_dates=[])

    # If one or more dates are passed, an error should be raised.
    with pytest.raises(RuntimeError):
        raise_error_for_dates(error_dates=[dt.date(2011, 1, 1)])


def test_clean_outputs_for_date_range(fs):
    hemisphere = NORTH
    start_date = dt.date(2025, 1, 1)
    end_date = dt.date(2025, 1, 2)

    base_output_dir = Path("/output/")
    complete_output_dir = get_complete_output_dir(
        hemisphere=hemisphere,
        base_output_dir=base_output_dir,
    )

    # for date in start_date, end_date
    complete_files = [
        complete_output_dir / "sic_psn12.5_20241226_am2_v06r00.nc",
        complete_output_dir / "sic_psn12.5_20241227_am2_v06r00.nc",
        complete_output_dir / "sic_psn12.5_20241228_am2_v06r00.nc",
        complete_output_dir / "sic_psn12.5_20241229_am2_v06r00.nc",
        complete_output_dir / "sic_psn12.5_20241230_am2_v06r00.nc",
        complete_output_dir / "sic_psn12.5_20241231_am2_v06r00.nc",
        complete_output_dir / "sic_psn12.5_20250101_am2_v06r00.nc",  # target
        complete_output_dir / "sic_psn12.5_20250102_am2_v06r00.nc",  # target
        complete_output_dir / "sic_psn12.5_20250103_am2_v06r00.nc",
        complete_output_dir / "sic_psn12.5_20250104_am2_v06r00.nc",
        complete_output_dir / "sic_psn12.5_20250105_am2_v06r00.nc",
        complete_output_dir / "sic_psn12.5_20250106_am2_v06r00.nc",
        complete_output_dir / "sic_psn12.5_20250107_am2_v06r00.nc",
        complete_output_dir / "sic_psn12.5_20250108_am2_v06r00.nc",
    ]
    for complete_file in complete_files:
        fs.create_file(complete_file)

    intermediate_output_dir = get_intermediate_output_dir(
        hemisphere=hemisphere,
        base_output_dir=base_output_dir,
    )
    intermediate_files = [
        intermediate_output_dir / "sic_psn12.5_20241226_am2_v06r00.nc",
        intermediate_output_dir / "sic_psn12.5_20241227_am2_v06r00.nc",
        intermediate_output_dir / "sic_psn12.5_20241228_am2_v06r00.nc",
        intermediate_output_dir / "sic_psn12.5_20241229_am2_v06r00.nc",
        intermediate_output_dir / "sic_psn12.5_20241230_am2_v06r00.nc",
        intermediate_output_dir / "sic_psn12.5_20241231_am2_v06r00.nc",
        intermediate_output_dir / "sic_psn12.5_20250101_am2_v06r00.nc",  # target
        intermediate_output_dir / "sic_psn12.5_20250102_am2_v06r00.nc",  # target
        intermediate_output_dir / "sic_psn12.5_20250103_am2_v06r00.nc",
        intermediate_output_dir / "sic_psn12.5_20250104_am2_v06r00.nc",
        intermediate_output_dir / "sic_psn12.5_20250105_am2_v06r00.nc",
        intermediate_output_dir / "sic_psn12.5_20250106_am2_v06r00.nc",
        intermediate_output_dir / "sic_psn12.5_20250107_am2_v06r00.nc",
        intermediate_output_dir / "sic_psn12.5_20250108_am2_v06r00.nc",
    ]
    for intermediate_file in intermediate_files:
        fs.create_file(intermediate_file)

    clean_outputs_for_date_range(
        hemisphere=hemisphere,
        base_output_dir=base_output_dir,
        start_date=start_date,
        end_date=end_date,
    )

    expected_complete_files = [
        complete_output_dir / "sic_psn12.5_20241226_am2_v06r00.nc",
        complete_output_dir / "sic_psn12.5_20241227_am2_v06r00.nc",
        complete_output_dir / "sic_psn12.5_20241228_am2_v06r00.nc",
        complete_output_dir / "sic_psn12.5_20241229_am2_v06r00.nc",
        complete_output_dir / "sic_psn12.5_20241230_am2_v06r00.nc",
        complete_output_dir / "sic_psn12.5_20241231_am2_v06r00.nc",
        complete_output_dir / "sic_psn12.5_20250103_am2_v06r00.nc",
        complete_output_dir / "sic_psn12.5_20250104_am2_v06r00.nc",
        complete_output_dir / "sic_psn12.5_20250105_am2_v06r00.nc",
        complete_output_dir / "sic_psn12.5_20250106_am2_v06r00.nc",
        complete_output_dir / "sic_psn12.5_20250107_am2_v06r00.nc",
        complete_output_dir / "sic_psn12.5_20250108_am2_v06r00.nc",
    ]

    expected_intermediate_files = [
        intermediate_output_dir / "sic_psn12.5_20241226_am2_v06r00.nc",
        intermediate_output_dir / "sic_psn12.5_20250108_am2_v06r00.nc",
    ]

    actual_complete_files = list(complete_output_dir.rglob("*.nc"))
    actual_intermediate_files = list(intermediate_output_dir.rglob("*.nc"))

    assert set(actual_complete_files) == set(expected_complete_files)
    assert set(actual_intermediate_files) == set(expected_intermediate_files)
