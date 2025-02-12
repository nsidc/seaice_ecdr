"""Contains routines for handling days that are treated with non-standard
processing."""

import datetime as dt
import os
from functools import cache
from pathlib import Path

import yaml
from pm_tb_data._types import Hemisphere

from seaice_ecdr.platforms.models import SUPPORTED_PLATFORM_ID

_this_dir = Path(__file__).parent
DEFAULT_MISSING_DATES_CONFIG_FILEPATH = Path(
    _this_dir / "./config/dates_handled_differently.yml"
).resolve()


@cache
def _load_dates_treated_differently():
    """The default configuration file can be overridden
    by using an environment variable:
        MISSING_DATES_CONFIG_FILEPATH

    Yaml file has structure like:
      treat_tbs_as_missing:
        n07:
          NORTH:
            - ['1984-07-03', '1984-08-04']
      treat_outputs_as_missing:
        n07:
          NORTH:
            - ['1984-07-03', '1984-08-04']

    """
    if missing_dates_config_fp_str := os.environ.get("MISSING_DATES_CONFIG_FILEPATH"):
        missing_dates_filepath = Path(missing_dates_config_fp_str)
    else:
        missing_dates_filepath = Path(DEFAULT_MISSING_DATES_CONFIG_FILEPATH)

    # TODO: Use pydantic to implement validator here
    #       eg follow example in seaice_ecdr/platforms/models.py

    with open(missing_dates_filepath, "r") as f:
        return yaml.safe_load(f)


def _missing_range_strs_to_dates(missing_ranges):
    date_format = "%Y-%m-%d"
    date_ranges = []
    for start_str, end_str in missing_ranges:
        start_date = dt.datetime.strptime(start_str, date_format).date()
        end_date = dt.datetime.strptime(end_str, date_format).date()
        date_ranges.append((start_date, end_date))

    return date_ranges


def periods_with_missing_tbs(
    platform_id: SUPPORTED_PLATFORM_ID,
    hemisphere: Hemisphere,
):
    """Returns a list of date ranges for which tbs should be considered
    missing.
    """
    dates_treated_differently = _load_dates_treated_differently()
    try:
        ranges = dates_treated_differently["treat_tbs_as_missing"][platform_id][
            hemisphere
        ]
    except KeyError:
        ranges = []

    return _missing_range_strs_to_dates(ranges)


def periods_of_cdr_missing_data(
    platform_id: SUPPORTED_PLATFORM_ID,
    hemisphere: Hemisphere,
):
    """Returns a list of date ranges for which CDR's output should be
    all-missing.
    """
    dates_treated_differently = _load_dates_treated_differently()
    try:
        ranges = dates_treated_differently["treat_outputs_as_missing"][platform_id][
            hemisphere
        ]
    except KeyError:
        ranges = []

    return _missing_range_strs_to_dates(ranges)


def months_of_cdr_missing_data(
    platform_id: SUPPORTED_PLATFORM_ID,
    hemisphere: Hemisphere,
):
    """Returns a list of year-month strings
    for which CDR's output should be all-missing.
    """
    dates_treated_differently = _load_dates_treated_differently()
    try:
        missing_year_months = dates_treated_differently["treat_months_as_missing"][
            platform_id
        ][hemisphere]
    except KeyError:
        missing_year_months = []

    return missing_year_months


def year_month_is_all_empty(year_month, empty_year_months):
    """
    Given a list of date ranges (inclusive start/end),
    return boolean if date intersects any range.
    """
    for empty_year_month in empty_year_months:
        if year_month == empty_year_month:
            return True

    return False


def date_in_ranges(date, date_ranges):
    """
    Given a list of date ranges (inclusive start/end),
    return boolean if date intersects any range.
    """
    for start_date, end_date in date_ranges:
        if start_date <= date and date <= end_date:
            return True

    return False


def day_has_all_bad_tbs(
    platform_id: SUPPORTED_PLATFORM_ID,
    hemisphere: Hemisphere,
    date: dt.date,
) -> bool:
    missing_tb_periods = periods_with_missing_tbs(platform_id, hemisphere)

    return date_in_ranges(date, missing_tb_periods)


def day_has_all_empty_fields(
    platform_id: SUPPORTED_PLATFORM_ID,
    hemisphere: Hemisphere,
    date: dt.date,
) -> bool:
    # TODO: Cause this to be related to defined platform_id/dates?
    if date < dt.date(1978, 10, 25):
        return True

    missing_tb_periods = periods_of_cdr_missing_data(platform_id, hemisphere)

    return date_in_ranges(date, missing_tb_periods)
