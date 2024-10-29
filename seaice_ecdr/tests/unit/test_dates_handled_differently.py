import datetime as dt

from pm_tb_data._types import NORTH, SOUTH

from seaice_ecdr.days_treated_differently import (
    day_has_all_bad_tbs,
    day_has_all_empty_fields,
)


def test_all_tbs_for_day_are_bad():
    """Test that days whose TBs are known to be bad are known
    Initially, these test values are from CDRv4 code:
       /vagrant/source/tests/test_cdr_configuration.py
    """
    # North
    assert day_has_all_bad_tbs("n07", NORTH, dt.date(1984, 7, 3))
    assert day_has_all_bad_tbs("n07", NORTH, dt.date(1986, 12, 5))
    assert day_has_all_bad_tbs("am2", NORTH, dt.date(2018, 12, 16))
    assert not day_has_all_bad_tbs("F08", NORTH, dt.date(1988, 12, 5))

    # South
    assert day_has_all_bad_tbs("n07", SOUTH, dt.date(1986, 12, 5))
    assert day_has_all_bad_tbs("F17", SOUTH, dt.date(2008, 3, 24))
    assert day_has_all_bad_tbs("F17", SOUTH, dt.date(2015, 8, 6))
    assert day_has_all_bad_tbs("am2", NORTH, dt.date(2018, 12, 16))
    assert not day_has_all_bad_tbs("F17", SOUTH, dt.date(2016, 8, 6))


def test_day_has_all_empty_fields():
    """Test for days that should have no data in it because it
    is in a long period of no available data such that not even
    temporal interpolation of the data should be used.
    Original test dates came from CDRv4 code:
       /vagrant/source/tests/test_cdr_configuration.py
    """
    # Date is before start of first possible platform's data (n07)
    assert day_has_all_empty_fields("n07", "north", dt.date(1978, 10, 24))
    assert day_has_all_empty_fields("n07", "south", dt.date(1978, 10, 24))

    # TODO: Combine tests of this with platform availability?

    # North
    assert day_has_all_empty_fields("n07", "north", dt.date(1984, 7, 3))
    # assert day_has_all_empty_fields("F08", "north", dt.date(1990, 12, 27))
    assert not day_has_all_empty_fields("F17", "north", dt.date(2010, 12, 10))

    # South
    assert day_has_all_empty_fields("n07", "south", dt.date(1984, 8, 23))
    assert day_has_all_empty_fields("F08", "south", dt.date(1987, 12, 10))
    assert not day_has_all_empty_fields("F17", "south", dt.date(2010, 12, 10))


"""
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
    expected = f"sic_psn12.5_20210101_am2_{ECDR_NRT_PRODUCT_VERSION}_P.nc"

    actual = nrt_daily_filename(
        hemisphere=NORTH, resolution="12.5", platform_id="am2", date=dt.date(2021, 1, 1)
    )

    assert actual == expected


def test_nrt_monthly_filename():
    expected = f"sic_psn25_202409_F17_{ECDR_NRT_PRODUCT_VERSION}_P.nc"

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
        ancillary_source="CDRv5",
    )

    assert detected_missing == 1


def test_raise_error_for_dates():
    # If no dates are passed, no error should be raised.
    raise_error_for_dates(error_dates=[])

    # If one or more dates are passed, an error should be raised.
    with pytest.raises(RuntimeError):
        raise_error_for_dates(error_dates=[dt.date(2011, 1, 1)])
"""
