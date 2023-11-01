import datetime as dt
from pathlib import Path

import pytest
import xarray as xr

from seaice_ecdr.complete_daily_ecdr import get_ecdr_dir
from seaice_ecdr.monthly import (
    _get_daily_complete_filepaths_for_month,
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
