import datetime as dt
from functools import partial
from itertools import chain
from multiprocessing import Pool
from pathlib import Path
from typing import get_args

import click
from pm_tb_data._types import Hemisphere

from seaice_ecdr._types import ECDR_SUPPORTED_RESOLUTIONS
from seaice_ecdr.ancillary import ANCILLARY_SOURCES
from seaice_ecdr.cli.util import datetime_to_date
from seaice_ecdr.complete_daily_ecdr import create_standard_ecdr_for_dates
from seaice_ecdr.constants import DEFAULT_BASE_OUTPUT_DIR
from seaice_ecdr.initial_daily_ecdr import create_idecdr_for_date
from seaice_ecdr.platforms import PLATFORM_CONFIG
from seaice_ecdr.spillover import LAND_SPILL_ALGS
from seaice_ecdr.temporal_composite_daily import make_tiecdr_netcdf
from seaice_ecdr.util import (
    date_range,
    get_dates_by_year,
    get_intermediate_output_dir,
    raise_error_for_dates,
)

# TODO:
#  ancillary sources are given in seaice_ecdr.ancillary.ANCILLARY_SOURCES
#    but are manually spelled out in the click options here


@click.command(name="multiprocess-daily")
@click.option(
    "-d",
    "--start-date",
    required=True,
    type=click.DateTime(
        formats=(
            "%Y-%m-%d",
            "%Y%m%d",
            "%Y.%m.%d",
        )
    ),
    callback=datetime_to_date,
)
@click.option(
    "--end-date",
    required=True,
    type=click.DateTime(
        formats=(
            "%Y-%m-%d",
            "%Y%m%d",
            "%Y.%m.%d",
        )
    ),
    # Like `datetime_to_date` but allows `None`.
    callback=lambda _ctx, _param, value: value if value is None else value.date(),
    default=None,
    help="If given, run temporal composite for `--date` through this end date.",
)
@click.option(
    "-h",
    "--hemisphere",
    required=True,
    type=click.Choice(get_args(Hemisphere)),
)
@click.option(
    "--base-output-dir",
    required=True,
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
    default=DEFAULT_BASE_OUTPUT_DIR,
    help=(
        "Base output directory for standard ECDR outputs."
        " Subdirectories are created for outputs of"
        " different stages of processing."
    ),
    show_default=True,
)
@click.option(
    "--overwrite",
    is_flag=True,
)
@click.option(
    "--land-spillover-alg",
    required=False,
    # type=click.Choice(["BT_NT", "NT2", "ILS"]),
    type=click.Choice(get_args(LAND_SPILL_ALGS)),
    default="BT_NT",
)
@click.option(
    "--ancillary-source",
    required=True,
    # type=click.Choice(["CDRv4", "CDRv5"]),
    type=click.Choice(get_args(ANCILLARY_SOURCES)),
    default="CDRv5",
)
@click.option(
    "--resolution",
    required=True,
    type=click.Choice(get_args(ECDR_SUPPORTED_RESOLUTIONS)),
)
def cli(
    start_date: dt.date,
    end_date: dt.date,
    hemisphere: Hemisphere,
    base_output_dir: Path,
    overwrite: bool,
    land_spillover_alg: LAND_SPILL_ALGS,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ancillary_source: ANCILLARY_SOURCES,
):
    dates = list(date_range(start_date=start_date, end_date=end_date))
    dates_by_year = get_dates_by_year(dates)

    # craete a range of dates that includes the interpolation range. This
    # ensures that data expected for temporal interpolation of the requested
    # dates has the data it needs.
    earliest_date = PLATFORM_CONFIG.get_first_platform_start_date()
    initial_start_date = max(earliest_date, start_date - dt.timedelta(days=5))
    initial_end_date = min(end_date + dt.timedelta(days=5), dt.date.today())
    initial_file_dates = list(
        date_range(start_date=initial_start_date, end_date=initial_end_date)
    )

    intermediate_output_dir = get_intermediate_output_dir(
        base_output_dir=base_output_dir,
        hemisphere=hemisphere,
        is_nrt=False,
    )

    _create_idecdr_wrapper = partial(
        create_idecdr_for_date,
        hemisphere=hemisphere,
        resolution=resolution,
        intermediate_output_dir=intermediate_output_dir,
        overwrite_ide=overwrite,
        land_spillover_alg=land_spillover_alg,
        ancillary_source=ancillary_source,
    )

    _create_tiecdr_wrapper = partial(
        make_tiecdr_netcdf,
        hemisphere=hemisphere,
        resolution=resolution,
        intermediate_output_dir=intermediate_output_dir,
        overwrite_tie=overwrite,
        land_spillover_alg=land_spillover_alg,
        ancillary_source=ancillary_source,
    )

    _complete_daily_wrapper = partial(
        create_standard_ecdr_for_dates,
        hemisphere=hemisphere,
        resolution=resolution,
        base_output_dir=base_output_dir,
        overwrite_cde=overwrite,
        land_spillover_alg=land_spillover_alg,
        ancillary_source=ancillary_source,
    )

    # Use 6 cores. This seems to perform well. Using the max number available
    # can cause issues...
    with Pool(6) as multi_pool:
        multi_pool.map(_create_idecdr_wrapper, initial_file_dates)
        multi_pool.map(_create_tiecdr_wrapper, dates)

        # The complete daily data must be generated sequentially within
        # each year. This is because the melt onset calculation requires access to
        # previous days' worth of complete daily data. Since the melt season is
        # DOY 50-244, we don't need to worry about the beginning of the year
        # requesting data for previous years of melt data.
        error_dates_lists: list[list[dt.date]] = multi_pool.map(
            _complete_daily_wrapper, dates_by_year
        )

    # Flatten the list-of-lists.
    error_dates: list[dt.date] = list(chain.from_iterable(error_dates_lists))
    raise_error_for_dates(error_dates=error_dates)
