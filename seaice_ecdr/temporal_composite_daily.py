"""Routines for generating temporally composited file.

"""

import datetime as dt


""" No logger yet...
# Set the default minimum log notification to Warning
logger.remove(0)  # Removes previous logger info
logger.add(sys.stderr, level="INFO")
"""


def get_sample_idecdr_filename(
    date,
    hemisphere,
    resolution,
):
    """Return name of sample inidial daily ecdr file."""
    sample_idecdr_filename = (
        f"sample_idecdr_{hemisphere}_{resolution}_" + f'{date.strftime("%Y%m%d")}.nc'
    )

    return sample_idecdr_filename


def iter_dates_near_date(
    seed_date: dt.date,
    day_range: int = 0,
    skip_future: bool = False,
):
    """Return iterator of dates near a given date."""
    first_date = seed_date - dt.timedelta(days=day_range)
    first_date = seed_date - dt.timedelta(days=day_range)
    if skip_future:
        last_date = seed_date
    else:
        last_date = seed_date + dt.timedelta(days=day_range)

    date = first_date
    while date <= last_date:
        yield date
        date += dt.timedelta(days=1)


def gen_temporal_composite_daily(
    date,
    hemisphere,
    resolution,
):
    """Create a temporally composited daily data set."""
    print("gen_temporal_composite_daily() not yet implemented...")


if __name__ == "__main__":
    date = dt.datetime(2021, 2, 16).date()
    hemisphere = "north"
    resolution = "12"

    gen_temporal_composite_daily(date, hemisphere, resolution)
