import datetime as dt
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Final

from loguru import logger
from pm_tb_data._types import Hemisphere

from seaice_ecdr.complete_daily_ecdr import create_cdecdr_for_date
from seaice_ecdr.constants import STANDARD_BASE_OUTPUT_DIR
from seaice_ecdr.initial_daily_ecdr import create_idecdr_for_date
from seaice_ecdr.temporal_composite_daily import create_tiecdr_for_date
from seaice_ecdr.util import date_range


def cli(
    hemisphere: Hemisphere,
    start_date: dt.date,
    end_date: dt.date,
    ecdr_data_dir: Path,
    overwrite: bool,
):
    dates = list(date_range(start_date=start_date, end_date=end_date))

    # craete a range of dates that includes the interpolation range. This
    # ensures that data expected for temporal interpolation of the requested
    # dates has the data it needs.
    initial_start_date = start_date - dt.timedelta(days=5)
    initial_end_date = min(end_date + dt.timedelta(days=5), dt.date.today())
    initial_file_dates = list(
        date_range(start_date=initial_start_date, end_date=initial_end_date)
    )

    resolution: Final = "12.5"

    _create_idecdr_wrapper = partial(
        create_idecdr_for_date,
        hemisphere=hemisphere,
        resolution=resolution,
        ecdr_data_dir=ecdr_data_dir,
    )

    _create_tiecdr_wrapper = partial(
        create_tiecdr_for_date,
        hemisphere=hemisphere,
        resolution=resolution,
        ecdr_data_dir=ecdr_data_dir,
    )

    _complete_daily_wrapper = partial(
        create_cdecdr_for_date,
        hemisphere=hemisphere,
        resolution=resolution,
        ecdr_data_dir=ecdr_data_dir,
        overwrite_cde=overwrite,
    )

    logger.info("About to multiprocess. Removing logger for now to reduce clutter.")
    logger.remove()
    # leave one core free.
    # num_cores = os.cpu_count() - 1
    # logger.info(f"Multiprocessing daily data with {num_cores=}")
    with Pool(6) as multi_pool:
        multi_pool.map(_create_idecdr_wrapper, initial_file_dates)
        multi_pool.map(_create_tiecdr_wrapper, dates)
        multi_pool.map(_complete_daily_wrapper, dates)


if __name__ == "__main__":
    cli(
        hemisphere="north",
        start_date=dt.date(2022, 1, 1),
        end_date=dt.date(2022, 1, 30),
        ecdr_data_dir=STANDARD_BASE_OUTPUT_DIR,
        overwrite=True,
    )
