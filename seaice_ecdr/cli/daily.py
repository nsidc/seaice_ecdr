import copy
import datetime as dt
import subprocess
from pathlib import Path
from typing import Final, get_args

import click
from pm_tb_data._types import Hemisphere

from seaice_ecdr.cli.util import datetime_to_date
from seaice_ecdr.constants import DEFAULT_BASE_OUTPUT_DIR
from seaice_ecdr.platforms.config import PROTOTYPE_PLATFORM_START_DATES_CONFIG_FILEPATH
from seaice_ecdr.publish_daily import publish_daily_nc_for_dates

_THIS_DIR = Path(__file__).parent

CLI_EXE_PATH = _THIS_DIR / "../../scripts/cli.sh"


def _run_cmd(cmd: str):
    subprocess.run(
        cmd,
        shell=True,
        check=True,
    )


def make_25km_ecdr(
    start_date: dt.date,
    end_date: dt.date,
    hemisphere: Hemisphere,
    base_output_dir: Path,
):
    RESOLUTION: Final = "25"
    LAND_SPILLOVER_ALG: Final = "BT_NT"
    ANCILLARY_SOURCE: Final = "CDRv4"
    # Use the default platform dates, which excldues AMSR2
    _run_cmd(
        f"{CLI_EXE_PATH} multiprocess-intermediate-daily"
        f" --start-date {start_date:%Y-%m-%d} --end-date {end_date:%Y-%m-%d}"
        f" --hemisphere {hemisphere}"
        f" --base-output-dir {base_output_dir}"
        f" --land-spillover-alg {LAND_SPILLOVER_ALG}"
        f" --resolution {RESOLUTION}"
        f" --ancillary-source {ANCILLARY_SOURCE}"
    )

    # If the given start & end date intersect with the AMSR2 period, run that
    # separately:
    # TODO: the amsr2 start date should be read from the platform start date config.
    am2_start_date = dt.date(2013, 1, 1)
    if start_date >= am2_start_date:
        _run_cmd(
            f"export PLATFORM_START_DATES_CONFIG_FILEPATH={PROTOTYPE_PLATFORM_START_DATES_CONFIG_FILEPATH} &&"
            f" {CLI_EXE_PATH} multiprocess-intermediate-daily"
            f" --start-date {start_date:%Y-%m-%d} --end-date {end_date:%Y-%m-%d}"
            f" --hemisphere {hemisphere}"
            f" --base-output-dir {base_output_dir}"
            f" --land-spillover-alg {LAND_SPILLOVER_ALG}"
            f" --resolution {RESOLUTION}"
            f" --ancillary-source {ANCILLARY_SOURCE}"
        )

    # Prepare the daily data for publication
    publish_daily_nc_for_dates(
        base_output_dir=base_output_dir,
        start_date=start_date,
        end_date=end_date,
        hemisphere=hemisphere,
        resolution=RESOLUTION,
    )


@click.command(name="daily")
@click.option(
    "-d",
    "--date",
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
def cli(
    *,
    date: dt.date,
    end_date: dt.date | None,
    hemisphere: Hemisphere,
    base_output_dir: Path,
):
    if end_date is None:
        end_date = copy.copy(date)
    make_25km_ecdr(
        start_date=date,
        end_date=end_date,
        hemisphere=hemisphere,
        base_output_dir=base_output_dir,
    )


if __name__ == "__main__":
    cli()