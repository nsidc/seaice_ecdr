"""Validate outputs of the seaice ECDR.

There are two output files each time this program is run:
   error_seaice_cdr*.txt - error code for each day in date range run
   log_seaice_cdr*.txt - log of pixel counts for various parameters

   error file values:
      error = -9999 --> no CDR file on that date
      error = -999  --> CDR file exists but is empty (no valid concentrations)
      error = -99   --> CDR file exists and has values but some concentration values are in error
      error = -9    --> Melt is being flagged on a date that it should not be
      error = -2    --> More than 1000 missing values, indicates at least one missing swath
      error = -1    --> 100-1000 missing values, likely indicates at least part of
                        a missing swath
      error = 0     --> no significant problems in CDR fields

   non-varying log file parameters:
              north    south
      total: 136192   104912  --> total number pixels in the grid
      land:   63212    21273  --> number of land pixels
      coast:   5052      907  --> number of coast pixels
      lake:     665        0  --> number of lake pixels
      pole:     468        0  --> number of pixels in pole hole
   ***these values should not vary for any day or month***

   varying log file parameters:
      ice       --> number pixels with non-zero sea ice concentrations
      oceanmask --> number of pixels with no ice that are masked by ocean mask
                    this number will vary by month but should be the same for a given month
      ice-free  --> pixels that aren't masked but have a value of 0
      missing   --> pixels that are missing (no brightness temperatures)
      bad       --> pixels that have an error (invalid sea ice value), should always be 0
      melt      --> pixels that have ice and are melting (north only, 1 March - 1 September only)
"""
import datetime as dt
from pathlib import Path
from typing import Literal, cast, get_args

import click
from pm_tb_data._types import Hemisphere

from seaice_ecdr.cli.util import datetime_to_date
from seaice_ecdr.constants import STANDARD_BASE_OUTPUT_DIR
from seaice_ecdr.util import date_range


def get_validation_dir(*, ecdr_data_dir: Path) -> Path:
    validation_dir = ecdr_data_dir / "validation"
    validation_dir.mkdir(exist_ok=True)

    return validation_dir


def validate_daily_outputs(
    *,
    hemisphere: Hemisphere,
    ecdr_data_dir: Path,
    start_date: dt.date,
    end_date: dt.date,
) -> None:
    """Create validation logs for daily outputs.

    Creates two space-delimited files (TODO: could be csv?):

    * log_seaice_{n|s}_daily_{start_year}_{end_year}.txt. Contains the following
      fields: [year, month, day, total, ice, land, coast, lake, pole, oceanmask,
      ice-free, missing, bad, melt].
    * error_seaice_{n|s}_daily_{start_year}_{end_year}.txt. Contains the
      following fields: [year, month, day, error_code]

    """
    for date in date_range(start_date=start_date, end_date=end_date):
        # TODO
        ...


def validate_monthly_outputs(
    *,
    hemisphere: Hemisphere,
    ecdr_data_dir: Path,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
) -> None:
    # TODO
    ...


@click.command(name="validate-outputs")
@click.option(
    "--ecdr-data-dir",
    required=True,
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
    default=STANDARD_BASE_OUTPUT_DIR,
    help=(
        "Base output directory for standard ECDR outputs."
        " Subdirectories are created for outputs of"
        " different stages of processing."
    ),
    show_default=True,
)
# hemisphere
@click.option(
    "--hemisphere",
    required=True,
    type=click.Choice([*get_args(Hemisphere), "both"]),
    default="both",
)
# product type (daily/monthly)
@click.option(
    "--product-type",
    required=True,
    type=click.Choice(["daily", "monthly", "both"]),
    default="both",
)
# start/end dates
@click.option(
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
    callback=datetime_to_date,
)
def cli(
    ecdr_data_dir: Path,
    hemisphere_str: Hemisphere | Literal["both"],
    product_type: Literal["daily", "monthly", "both"],
    start_date: dt.date,
    end_date: dt.date,
):
    daily = False
    monthly = False
    if product_type == "daily" or product_type == "both":
        daily = True
    elif product_type == "monthly" or product_type == "both":
        monthly = True

    hemispheres = get_args(Hemisphere) if hemisphere_str == "both" else [hemisphere_str]
    hemispheres = cast(list[Hemisphere], hemispheres)
    for hemisphere in hemispheres:
        # daily & monthly, NH and SH.
        if daily:
            validate_daily_outputs(
                hemisphere=hemisphere,
                ecdr_data_dir=ecdr_data_dir,
                start_date=start_date,
                end_date=end_date,
            )
        if monthly:
            validate_monthly_outputs(
                hemisphere=hemisphere,
                ecdr_data_dir=ecdr_data_dir,
                start_year=start_date.year,
                start_month=start_date.month,
                end_year=end_date.year,
                end_month=end_date.month,
            )
