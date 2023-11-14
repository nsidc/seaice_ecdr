from seaice_ecdr.nc_attrs import (
    _get_software_version_id,
    _get_time_coverage_duration_resolution,
)


def test__get_time_coverage_duration_resolution():
    expected_daily = {
        "time_coverage_duration": "P1D",
        "time_coverage_resolution": "P1D",
    }
    actual_daily = _get_time_coverage_duration_resolution(
        temporality="daily",
        aggregate=False,
    )
    assert expected_daily == actual_daily

    expected_daily_aggregate = {
        "time_coverage_duration": "P1Y",
        "time_coverage_resolution": "P1D",
    }
    actual_daily_aggregate = _get_time_coverage_duration_resolution(
        temporality="daily",
        aggregate=True,
    )
    assert expected_daily_aggregate == actual_daily_aggregate

    expected_monthly = {
        "time_coverage_duration": "P1M",
        "time_coverage_resolution": "P1M",
    }
    actual_monthly = _get_time_coverage_duration_resolution(
        temporality="monthly",
        aggregate=False,
    )
    assert expected_monthly == actual_monthly

    expected_monthly_aggregate = {
        "time_coverage_resolution": "P1M",
    }
    actual_monthly_aggregate = _get_time_coverage_duration_resolution(
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
