import datetime as dt
from pathlib import Path
from typing import get_args

import click
import pandas as pd
from pm_tb_data._types import Hemisphere

from seaice_ecdr._types import ECDR_SUPPORTED_RESOLUTIONS
from seaice_ecdr.ancillary import ANCILLARY_SOURCES
from seaice_ecdr.cli.util import CLI_EXE_PATH, run_cmd
from seaice_ecdr.constants import (
    DEFAULT_ANCILLARY_SOURCE,
    DEFAULT_BASE_OUTPUT_DIR,
    DEFAULT_CDR_RESOLUTION,
    DEFAULT_SPILLOVER_ALG,
)
from seaice_ecdr.platforms.config import (
    DEFAULT_PLATFORM_START_DATES_CONFIG_FILEPATH,
    PROTOTYPE_PLATFORM_START_DATES_CONFIG_FILEPATH,
)
from seaice_ecdr.publish_monthly import prepare_monthly_nc_for_publication
from seaice_ecdr.spillover import LAND_SPILL_ALGS


def make_monthly_25km_ecdr(
    year: int,
    month: int,
    end_year: int,
    end_month: int,
    hemisphere: Hemisphere,
    base_output_dir: Path,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    land_spillover_alg: LAND_SPILL_ALGS,
    ancillary_source: ANCILLARY_SOURCES,
):
    # TODO: the amsr2 start date should ideally be read from the platform start
    # date config.
    PROTOTYPE_PLATFORM_START_YEAR = 2013
    PROTOTYPE_PLATFORM_START_MONTH = 1

    PROTOTYPE_PLATFORM_START_DATE = dt.date(
        PROTOTYPE_PLATFORM_START_YEAR, PROTOTYPE_PLATFORM_START_MONTH, 1
    )

    # Use the default platform dates, which excludes AMSR2
    run_cmd(
        f"export PLATFORM_START_DATES_CONFIG_FILEPATH={DEFAULT_PLATFORM_START_DATES_CONFIG_FILEPATH} &&"
        f" {CLI_EXE_PATH} intermediate-monthly"
        f" --year {year} --month {month}"
        f" --end-year {end_year} --end-month {end_month}"
        f" --hemisphere {hemisphere}"
        f" --base-output-dir {base_output_dir}"
        f" --resolution {resolution}"
        f" --ancillary-source {ancillary_source}"
    )

    # If the given start & end date intersect with the AMSR2 period,
    # run that separately:
    if dt.date(end_year, end_month, 1) >= PROTOTYPE_PLATFORM_START_DATE:
        proto_year = year
        proto_month = month
        if dt.date(year, month, 1) < PROTOTYPE_PLATFORM_START_DATE:
            proto_year = PROTOTYPE_PLATFORM_START_YEAR
            proto_month = PROTOTYPE_PLATFORM_START_MONTH

        run_cmd(
            f"export PLATFORM_START_DATES_CONFIG_FILEPATH={PROTOTYPE_PLATFORM_START_DATES_CONFIG_FILEPATH} &&"
            f"{CLI_EXE_PATH} intermediate-monthly"
            f" --year {proto_year} --month {proto_month}"
            f" --end-year {end_year} --end-month {end_month}"
            f" --hemisphere {hemisphere}"
            f" --base-output-dir {base_output_dir}"
            f" --resolution {resolution}"
            f" --ancillary-source {ancillary_source}"
        )

    # Prepare the monthly data for publication
    for period in pd.period_range(
        start=pd.Period(year=year, month=month, freq="M"),
        end=pd.Period(year=end_year, month=end_month, freq="M"),
        freq="M",
    ):
        prepare_monthly_nc_for_publication(
            year=period.year,
            month=period.month,
            base_output_dir=base_output_dir,
            hemisphere=hemisphere,
            resolution=resolution,
            is_nrt=False,
        )


@click.command(name="monthly")
@click.option(
    "--year",
    required=True,
    type=int,
    help="Year for which to create the monthly file.",
)
@click.option(
    "--month",
    required=True,
    type=int,
    help="Month for which to create the monthly file.",
)
@click.option(
    "--end-year",
    required=False,
    default=None,
    type=int,
    help="If given, the end year for which to create monthly files.",
)
@click.option(
    "--end-month",
    required=False,
    default=None,
    type=int,
    help="If given, the end year for which to create monthly files.",
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
    year: int,
    month: int,
    end_year: int | None,
    end_month: int | None,
    hemisphere: Hemisphere,
    base_output_dir: Path,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    land_spillover_alg: LAND_SPILL_ALGS,
    ancillary_source: ANCILLARY_SOURCES,
) -> None:
    # Note: It appears that click cannot set one argument based on another.
    #       For clarity, we handle the "None" arg condition here for end_<vars>
    if end_year is None:
        end_year = year
    if end_month is None:
        end_month = month

    make_monthly_25km_ecdr(
        year=year,
        month=month,
        end_year=end_year,
        end_month=end_month,
        hemisphere=hemisphere,
        base_output_dir=base_output_dir,
        resolution=resolution,
        land_spillover_alg=land_spillover_alg,
        ancillary_source=ancillary_source,
    )


if __name__ == "__main__":
    cli()
