from pathlib import Path
from typing import Final, Literal, get_args

import click
import pandas as pd
from pm_tb_data._types import Hemisphere

from seaice_ecdr.cli.util import CLI_EXE_PATH, run_cmd
from seaice_ecdr.constants import DEFAULT_BASE_NRT_OUTPUT_DIR
from seaice_ecdr.platforms.config import (
    NRT_AM2_PLATFORM_START_DATES_CONFIG_FILEPATH,
    NRT_F17_PLATFORM_START_DATES_CONFIG_FILEPATH,
)


def make_monthly_25km_ecdr(
    year: int,
    month: int,
    end_year: int | None,
    end_month: int | None,
    hemisphere: Hemisphere,
    base_output_dir: Path,
    nrt_platform_id: Literal["F17", "am2"],
):
    if end_year is None:
        end_year = year
    if end_month is None:
        end_month = month

    # TODO: consider extracting these to CLI options that default to these values.
    RESOLUTION: Final = "25"
    ANCILLARY_SOURCE: Final = "CDRv5"
    if nrt_platform_id == "F17":
        nrt_platform_start_dates_filepath = NRT_F17_PLATFORM_START_DATES_CONFIG_FILEPATH
    elif nrt_platform_id == "am2":
        nrt_platform_start_dates_filepath = NRT_AM2_PLATFORM_START_DATES_CONFIG_FILEPATH
    else:
        raise RuntimeError(f"NRT processing is not defined for {nrt_platform_id}")
    run_cmd(
        f"export PLATFORM_START_DATES_CONFIG_FILEPATH={nrt_platform_start_dates_filepath} &&"
        f" {CLI_EXE_PATH} intermediate-monthly"
        f" --year {year} --month {month}"
        f" --end-year {end_year} --end-month {end_month}"
        f" --hemisphere {hemisphere}"
        f" --base-output-dir {base_output_dir}"
        f" --resolution {RESOLUTION}"
        f" --ancillary-source {ANCILLARY_SOURCE}"
        " --is-nrt"
    )

    # Prepare the monthly data for publication
    for period in pd.period_range(
        start=pd.Period(year=year, month=month, freq="M"),
        end=pd.Period(year=end_year, month=end_month, freq="M"),
        freq="M",
    ):
        run_cmd(
            f"export PLATFORM_START_DATES_CONFIG_FILEPATH={nrt_platform_start_dates_filepath} &&"
            f" {CLI_EXE_PATH} prepare-monthly-for-publish"
            f" --year {period.year} --month {period.month}"
            f" --hemisphere {hemisphere}"
            f" --base-output-dir {base_output_dir}"
            " --is-nrt"
        )


@click.command(name="monthly-nrt")
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
    "--nrt-platform-id",
    # TODO: eventually we want to support AM2 as a separate monthly file. This
    # does not currently work: the code instead places the AMSR2 monthly data
    # into a `am2_prototype` group like the non-nrt files, and that's not what
    # we want because the lag on data for F17 is different from AM2.
    type=click.Choice(["F17"]),
    default="F17",
)
def cli(
    *,
    year: int,
    month: int,
    end_year: int | None,
    end_month: int | None,
    hemisphere: Hemisphere | Literal["both"],
    base_output_dir: Path,
    nrt_platform_id: Literal["F17"],
) -> None:
    base_output_dir = base_output_dir / "CDR"
    base_output_dir.mkdir(exist_ok=True)

    if hemisphere == "both":
        hemispheres: list[Hemisphere] = ["north", "south"]
    else:
        hemispheres = [hemisphere]

    for hemisphere in hemispheres:
        make_monthly_25km_ecdr(
            year=year,
            month=month,
            end_year=end_year,
            end_month=end_month,
            hemisphere=hemisphere,
            base_output_dir=base_output_dir,
            nrt_platform_id=nrt_platform_id,
        )


if __name__ == "__main__":
    cli()
