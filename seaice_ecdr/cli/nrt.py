"""Wrapper around the nrt CLI to override platform start dates config

This is a hack, and should be unnecessary once the code is refactored to make it
easier to configure the platform start dates.
"""

import copy
import datetime as dt
from pathlib import Path
from typing import get_args

import click
from pm_tb_data._types import Hemisphere

from seaice_ecdr.cli.util import CLI_EXE_PATH, datetime_to_date, run_cmd
from seaice_ecdr.constants import DEFAULT_BASE_NRT_OUTPUT_DIR
from seaice_ecdr.platforms.config import NRT_PLATFORM_START_DATES_CONFIG_FILEPATH


@click.command(name="daily-nrt")
@click.option(
    "-d",
    "--date",
    required=True,
    type=click.DateTime(formats=("%Y-%m-%d", "%Y%m%d", "%Y.%m.%d")),
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
    date: dt.date,
    end_date: dt.date | None,
    hemisphere: Hemisphere,
    base_output_dir: Path,
    overwrite: bool,
):
    if end_date is None:
        end_date = copy.copy(date)

    overwrite_str = " --overwrite" if overwrite else ""

    run_cmd(
        f"export PLATFORM_START_DATES_CONFIG_FILEPATH={NRT_PLATFORM_START_DATES_CONFIG_FILEPATH} &&"
        f"{CLI_EXE_PATH} nrt"
        f" --hemisphere {hemisphere}"
        f" --base-output-dir {base_output_dir}"
        f" --date {date:%Y-%m-%d}"
        f" --end-date {end_date:%Y-%m-%d}" + overwrite_str
    )
