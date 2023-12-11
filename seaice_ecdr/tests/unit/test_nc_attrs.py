import numpy as np
import pandas as pd
import xarray as xr

from seaice_ecdr.nc_attrs import (
    _get_software_version_id,
    _get_time_coverage_attrs,
    get_platforms_for_sats,
)


def test__get_time_coverage_attrs_daily():
    # Typical daily complete file
    expected_daily = {
        "time_coverage_start": "2022-03-01T00:00:00Z",
        "time_coverage_end": "2022-03-01T23:59:59Z",
        "time_coverage_duration": "P1D",
        "time_coverage_resolution": "P1D",
    }
    actual_daily = _get_time_coverage_attrs(
        time=xr.DataArray(
            pd.date_range(start="2022-03-01", end="2022-03-01", freq="D")
        ),
        temporality="daily",
        aggregate=False,
    )
    assert expected_daily == actual_daily


def test__get_time_coverage_attrs_daily_aggregate():
    # Daily aggregate for a full year
    expected_daily_aggregate = {
        "time_coverage_start": "2022-01-01T00:00:00Z",
        "time_coverage_end": "2022-12-31T23:59:59Z",
        "time_coverage_duration": "P1Y",
        "time_coverage_resolution": "P1D",
    }
    actual_daily_aggregate = _get_time_coverage_attrs(
        time=xr.DataArray(
            pd.date_range(start="2022-01-01", end="2022-12-31", freq="D")
        ),
        temporality="daily",
        aggregate=True,
    )
    assert expected_daily_aggregate == actual_daily_aggregate


def test__get_time_coverage_attrs_daily_aggregate_partial_year():
    # Daily aggregate for a partial year
    expected_daily_partial_aggregate = {
        "time_coverage_start": "2022-01-01T00:00:00Z",
        "time_coverage_end": "2022-02-01T23:59:59Z",
        "time_coverage_duration": "P32D",
        "time_coverage_resolution": "P1D",
    }
    actual_daily_partial_aggregate = _get_time_coverage_attrs(
        time=xr.DataArray(
            pd.date_range(start="2022-01-01", end="2022-02-01", freq="D")
        ),
        temporality="daily",
        aggregate=True,
    )
    assert expected_daily_partial_aggregate == actual_daily_partial_aggregate

    # Daily aggregate for a partial year at the beginning of the record
    expected_daily_partial_aggregate_eoy = {
        "time_coverage_start": "1978-10-01T00:00:00Z",
        "time_coverage_end": "1978-12-31T23:59:59Z",
        "time_coverage_duration": "P92D",
        "time_coverage_resolution": "P1D",
    }
    actual_daily_partial_aggregate_eoy = _get_time_coverage_attrs(
        time=xr.DataArray(
            pd.date_range(start="1978-10-01", end="1978-12-31", freq="D")
        ),
        temporality="daily",
        aggregate=True,
    )
    assert expected_daily_partial_aggregate_eoy == actual_daily_partial_aggregate_eoy


def test__get_time_coverage_attrs_monthly():
    # Typical monthly file
    expected_monthly = {
        "time_coverage_start": "2022-03-01T00:00:00Z",
        "time_coverage_end": "2022-03-31T23:59:59Z",
        "time_coverage_duration": "P1M",
        "time_coverage_resolution": "P1M",
    }
    actual_monthly = _get_time_coverage_attrs(
        time=xr.DataArray(
            pd.date_range(start="2022-03-01", end="2022-03-01", freq="D")
        ),
        temporality="monthly",
        aggregate=False,
    )
    assert expected_monthly == actual_monthly


def test__get_time_coverage_attrs_monthly_aggregate():
    # Monthly aggregate file
    expected_monthly_aggregate = {
        "time_coverage_start": "2022-01-01T00:00:00Z",
        "time_coverage_end": "2022-05-31T23:59:59Z",
        "time_coverage_duration": "P5M",
        "time_coverage_resolution": "P1M",
    }
    actual_monthly_aggregate = _get_time_coverage_attrs(
        time=xr.DataArray(
            [np.datetime64(f"2022-{month:02}-01") for month in range(1, 5 + 1)]
        ),
        temporality="monthly",
        aggregate=True,
    )
    assert expected_monthly_aggregate == actual_monthly_aggregate


def test__get_software_version_id():
    software_ver_id = _get_software_version_id()

    # The softawre version id should look something like:
    # git@github.com:nsidc/seaice_ecdr.git@10fdd316452d0d69fbcf4e7915b66c227298b0ec
    assert "github" in software_ver_id
    assert "@" in software_ver_id


def test_get_platforms_for_sat():
    expected = [
        "DMSP 5D-2/F13 > Defense Meteorological Satellite Program-F13",
        "DMSP 5D-3/F17 > Defense Meteorological Satellite Program-F17",
        "GCOM-W1 > Global Change Observation Mission 1st-Water",
    ]

    actual = get_platforms_for_sats(
        [
            "F13",
            "F17",
            "F17",
            "am2",
            "am2",
            "am2",
        ]
    )

    assert expected == actual
