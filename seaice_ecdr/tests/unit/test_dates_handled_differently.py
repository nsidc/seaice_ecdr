import datetime as dt

from pm_tb_data._types import NORTH, SOUTH

from seaice_ecdr.days_treated_differently import (
    day_has_all_bad_tbs,
    day_has_all_empty_fields,
)


def test_all_tbs_for_day_are_bad():
    """Test that days whose TBs are known to be bad are known"""
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
    """
    # Date is before start of first possible platform's data (n07)
    assert day_has_all_empty_fields("n07", "north", dt.date(1978, 10, 24))
    assert day_has_all_empty_fields("n07", "south", dt.date(1978, 10, 24))

    # TODO: Combine tests of this with platform availability?

    # North
    assert day_has_all_empty_fields("n07", "north", dt.date(1984, 7, 3))
    assert not day_has_all_empty_fields("F17", "north", dt.date(2010, 12, 10))

    # South
    assert day_has_all_empty_fields("n07", "south", dt.date(1984, 8, 23))
    assert day_has_all_empty_fields("F08", "south", dt.date(1987, 12, 10))
    assert not day_has_all_empty_fields("F17", "south", dt.date(2010, 12, 10))
