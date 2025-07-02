"""Wrapper around the nrt CLI to override platform start dates config

This is a hack, and should be unnecessary once the code is refactored to make it
easier to configure the platform start dates.
"""

import copy
import datetime as dt
from pathlib import Path
from typing import Literal, get_args

import click
from loguru import logger
from pm_tb_data._types import Hemisphere

from seaice_ecdr.cli.util import CLI_EXE_PATH, run_cmd
from seaice_ecdr.constants import DEFAULT_BASE_NRT_OUTPUT_DIR
from seaice_ecdr.platforms.config import (
    NRT_AM2_PLATFORM_START_DATES_CONFIG_FILEPATH,
)


@click.command(name="daily-nrt")
@click.option(
    "-d",
    "--date",
    "--start-date",
    required=False,
    default=None,
    type=click.DateTime(formats=("%Y-%m-%d", "%Y%m%d", "%Y.%m.%d")),
    callback=lambda _ctx, _param, value: value if value is None else value.date(),
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
    "--last-n-days",
    required=False,
    type=click.INT,
    default=None,
    help="If given, run temporal composite for the last n dates.",
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
    default=DEFAULT_BASE_NRT_OUTPUT_DIR,
    help=(
        "Base output directory for NRT ECDR outputs."
        " Subdirectories are created for outputs of"
        " different stages of processing."
    ),
    show_default=True,
)
@click.option(
    "--overwrite",
    is_flag=True,
    help=("Overwrite intermediate and final outputs."),
)
def cli(
    *,
    date: dt.date | None,
    end_date: dt.date | None,
    last_n_days: int | None,
    hemisphere: Hemisphere | Literal["both"],
    base_output_dir: Path,
    overwrite: bool,
):
    base_output_dir = base_output_dir / "CDR"

    base_output_dir.mkdir(exist_ok=True)

    if last_n_days and (date or end_date):
        raise RuntimeError(
            "`--last-n-days` is incompatible with `--date` and `--end-date`"
        )

    if last_n_days:
        date = dt.date.today() - dt.timedelta(days=last_n_days)
        # The end date should be the day before today. We only expect today's
        # data to be available on the following day.
        end_date = dt.date.today() - dt.timedelta(days=1)

    if end_date is None:
        end_date = copy.copy(date)

    logger.info(f"Creating NRT data for {date} through {end_date}")

    overwrite_str = " --overwrite" if overwrite else ""

    if hemisphere == "both":
        hemispheres = ["north", "south"]
    else:
        hemispheres = [hemisphere]

    nrt_platform_start_dates_filepath = NRT_AM2_PLATFORM_START_DATES_CONFIG_FILEPATH

    for hemi in hemispheres:
        run_cmd(
            f"export PLATFORM_START_DATES_CONFIG_FILEPATH={nrt_platform_start_dates_filepath} &&"
            f"{CLI_EXE_PATH} nrt"
            f" --hemisphere {hemi}"
            f" --base-output-dir {base_output_dir}"
            f" --date {date:%Y-%m-%d}"
            f" --end-date {end_date:%Y-%m-%d}" + overwrite_str
        )
