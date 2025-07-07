import copy
import datetime as dt
from pathlib import Path
from typing import Literal, get_args

import click
from pm_tb_data._types import Hemisphere

from seaice_ecdr._types import ECDR_SUPPORTED_RESOLUTIONS
from seaice_ecdr.ancillary import ANCILLARY_SOURCES
from seaice_ecdr.cli.util import CLI_EXE_PATH, datetime_to_date, run_cmd
from seaice_ecdr.constants import (
    DEFAULT_ANCILLARY_SOURCE,
    DEFAULT_BASE_OUTPUT_DIR,
    DEFAULT_CDR_RESOLUTION,
    DEFAULT_SPILLOVER_ALG,
)
from seaice_ecdr.platforms.config import PROTOTYPE_PLATFORM_START_DATES_CONFIG_FILEPATH
from seaice_ecdr.publish_daily import publish_daily_nc_for_dates
from seaice_ecdr.spillover import LAND_SPILL_ALGS

_THIS_DIR = Path(__file__).parent

# TODO: the prototype platform start date should ideally be read from the
# platform start
# date config.
# TODO: this needs to be kept consistent with similar variables in
# `publish_monthly.py` and `cli.monthly`! If this gets updated, those need to be
# too!
PROTOTYPE_START_DATE: dt.date | None = None


def make_25km_ecdr(
    start_date: dt.date,
    end_date: dt.date,
    hemisphere: Hemisphere,
    base_output_dir: Path,
    no_multiprocessing: bool,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    land_spillover_alg: LAND_SPILL_ALGS,
    ancillary_source: ANCILLARY_SOURCES,
):
    # Use the default platform dates, which excludes AMSR2
    if no_multiprocessing:
        daily_intermediate_cmd = "intermediate-daily"
    else:
        daily_intermediate_cmd = "multiprocess-intermediate-daily"
    run_cmd(
        f"{CLI_EXE_PATH} {daily_intermediate_cmd}"
        f" --start-date {start_date:%Y-%m-%d} --end-date {end_date:%Y-%m-%d}"
        f" --hemisphere {hemisphere}"
        f" --base-output-dir {base_output_dir}"
        f" --land-spillover-alg {land_spillover_alg}"
        f" --resolution {resolution}"
        f" --ancillary-source {ancillary_source}"
    )

    # If the given start & end date intersect with the prototype period, run that
    # separately:
    if PROTOTYPE_START_DATE and end_date >= PROTOTYPE_START_DATE:
        proto_start_date = start_date
        proto_start_date = max(start_date, PROTOTYPE_START_DATE)

        if proto_start_date >= PROTOTYPE_START_DATE:
            run_cmd(
                f"export PLATFORM_START_DATES_CONFIG_FILEPATH={PROTOTYPE_PLATFORM_START_DATES_CONFIG_FILEPATH} &&"
                f" {CLI_EXE_PATH} {daily_intermediate_cmd}"
                f" --start-date {proto_start_date:%Y-%m-%d} --end-date {end_date:%Y-%m-%d}"
                f" --hemisphere {hemisphere}"
                f" --base-output-dir {base_output_dir}"
                f" --land-spillover-alg {land_spillover_alg}"
                f" --resolution {resolution}"
                f" --ancillary-source {ancillary_source}"
            )

    # Prepare the daily data for publication
    publish_daily_nc_for_dates(
        base_output_dir=base_output_dir,
        start_date=start_date,
        end_date=end_date,
        hemisphere=hemisphere,
        resolution=resolution,
    )


@click.command(name="daily")
@click.option(
    "-d",
    "--date",
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
    required=False,
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
    type=click.Choice([*get_args(Hemisphere), "both"]),
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
    "--no-multiprocessing",
    help="Disable multiprocessing. Useful for debugging purposes.",
    is_flag=True,
    default=False,
)
@click.option(
    "--resolution",
    required=True,
    type=click.Choice(get_args(ECDR_SUPPORTED_RESOLUTIONS)),
    default=DEFAULT_CDR_RESOLUTION,
)
@click.option(
    "--land-spillover-alg",
    required=True,
    type=click.Choice(get_args(LAND_SPILL_ALGS)),
    default=DEFAULT_SPILLOVER_ALG,
)
@click.option(
    "--ancillary-source",
    required=True,
    type=click.Choice(get_args(ANCILLARY_SOURCES)),
    default=DEFAULT_ANCILLARY_SOURCE,
)
def cli(
    *,
    date: dt.date,
    end_date: dt.date | None,
    hemisphere: Hemisphere | Literal["both"],
    base_output_dir: Path,
    no_multiprocessing: bool,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    land_spillover_alg: LAND_SPILL_ALGS,
    ancillary_source: ANCILLARY_SOURCES,
):
    if end_date is None:
        end_date = copy.copy(date)

    if hemisphere == "both":
        hemispheres: list[Hemisphere] = ["north", "south"]
    else:
        hemispheres = [hemisphere]

    for hemisphere in hemispheres:
        make_25km_ecdr(
            start_date=date,
            end_date=end_date,
            hemisphere=hemisphere,
            base_output_dir=base_output_dir,
            no_multiprocessing=no_multiprocessing,
            resolution=resolution,
            land_spillover_alg=land_spillover_alg,
            ancillary_source=ancillary_source,
        )


if __name__ == "__main__":
    cli()
