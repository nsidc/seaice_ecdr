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
      total: total number pixels in the grid
      land: number of land pixels
      coast: number of coast pixels
      lake: number of lake pixels

   varying log file parameters:
      pole: number of pixels in pole hole
      ice: number pixels with non-zero sea ice concentrations
      oceanmask: number of pixels with no ice that are masked by ocean mask
                    this number will vary by month but should be the same for a given month
      ice-free: pixels that aren't masked but have a value of 0
      missing: pixels that are missing (no brightness temperatures)
      bad: pixels that have an error (invalid sea ice value), should always be 0
      melt: pixels that have ice and are melting (north only, 1 March - 1 September only)
"""
import datetime as dt
from pathlib import Path
from typing import Final, Literal, cast, get_args

import click
import xarray as xr
from loguru import logger
from pm_tb_data._types import Hemisphere

from seaice_ecdr.ancillary import bitmask_value_for_meaning, flag_value_for_meaning
from seaice_ecdr.cli.util import datetime_to_date
from seaice_ecdr.complete_daily_ecdr import get_ecdr_filepath
from seaice_ecdr.constants import STANDARD_BASE_OUTPUT_DIR
from seaice_ecdr.util import date_range

VALIDATION_RESOLUTION: Final = "12.5"

ERROR_FILE_CODES = dict(
    missing_file=-9999,
    file_exists_but_is_empty=-999,
    file_exists_but_conc_values_are_bad=-99,
    melt_flagged_on_wrong_day=-9,
    more_than_1000_missing_values=-2,
    between_100_and_1000_missing_values=-1,
    no_problems=0,
)


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
    validation_dir = get_validation_dir(ecdr_data_dir=ecdr_data_dir)
    log_filepath = (
        validation_dir
        / f"log_seaice_{hemisphere[0]}_daily_{start_date.year}_{end_date.year}.txt"
    )
    error_filepath = (
        validation_dir
        / f"error_seaice_{hemisphere[0]}_daily_{start_date.year}_{end_date.year}.txt"
    )
    with open(log_filepath, "w") as log_file, open(error_filepath, "w") as error_file:
        # Write headers
        log_file.write(
            "year month day total ice land coast lake pole oceanmask ice-free missing bad melt\n"
        )
        error_file.write("year month day error_code\n")
        for date in date_range(start_date=start_date, end_date=end_date):
            data_fp = get_ecdr_filepath(
                date=date,
                hemisphere=hemisphere,
                resolution=VALIDATION_RESOLUTION,
                ecdr_data_dir=ecdr_data_dir,
            )

            # Missing data file.
            # "no file exists" is set as the value in the
            # log file, after "year, month, day".
            # Error file is the same except it gives the error code.
            if not data_fp.is_file():
                log_file.write(f"{date.year} {date.month} {date.day} no file exists\n")
                error_value = ERROR_FILE_CODES["missing_file"]
                error_file.write(f"{date.year} {date.month} {date.day} {error_value}\n")

            # A file exists on disk. Read it.
            ds = xr.open_dataset(data_fp)

            # Handle the log file first. It contains information on the # of
            # pixels of each type in the surface_type_mask and the ocean mask
            # (via qa_of_cdr_seaice_conc)
            # total ice land coast lake pole oceanmask ice-free missing bad melt
            total_num_pixels = len(ds.x) * len(ds.y)
            # Areas where there is a concentration detected.
            num_ice_pixels = int(
                ((ds.cdr_seaice_conc > 0) & (ds.cdr_seaice_conc <= 1)).sum()
            )

            # Surface value counts. These should be the same for every day.
            surf_value_counts = {}
            for flag in ("land", "coast", "lake", "polehole_mask"):
                flag_value = flag_value_for_meaning(
                    var=ds.surface_type_mask,
                    meaning=flag,
                )
                num_flag_pixels = int((ds.surface_type_mask == flag_value).sum())
                surf_value_counts[flag] = num_flag_pixels

            # Number of oceanmask (invalid ice mask) pixels
            invalid_ice_bitmask_value = bitmask_value_for_meaning(
                var=ds.qa_of_cdr_seaice_conc,
                meaning="valid_ice_mask_applied",
            )
            invalid_ice_mask = (
                ds.qa_of_cdr_seaice_conc & invalid_ice_bitmask_value
            ) > 0
            num_oceanmask_pixels = int(invalid_ice_mask.sum())

            # Ice-free pixels (conc == 0)
            num_ice_free_pixels = int((ds.cdr_seaice_conc == 0).sum())

            # TODO: we don't have missing pixels. Remove? Some other measure?
            # Leave hard-codded to 0?
            num_missing_pixels = 0

            # Per CDR v4, "bad" ice pixels are outside the expected range.
            # TODO: do we cutoff conc < 10%?
            less_than_10_sic = int(
                ((ds.cdr_seaice_conc > 0) & (ds.cdr_seaice_conc < 0.1)).sum()
            )
            gt_100_sic = int((ds.cdr_seaice_conc > 1).sum())
            num_bad_pixels = less_than_10_sic | gt_100_sic

            # Number of melt pixels
            melt_start_detected_bitmask_value = bitmask_value_for_meaning(
                var=ds.qa_of_cdr_seaice_conc,
                meaning="melt_start_detected",
            )
            melt_start_detected_mask = (
                ds.qa_of_cdr_seaice_conc & melt_start_detected_bitmask_value
            ) > 0
            num_melt_pixels = int(melt_start_detected_mask.sum())

            log_file.write(
                f"{date.year} {date.month} {date.day}"
                f" {total_num_pixels} {num_ice_pixels} {surf_value_counts['land']}"
                f" {surf_value_counts['coast']} {surf_value_counts['lake']} {surf_value_counts['polehole_mask']}"
                f" {num_oceanmask_pixels} {num_ice_free_pixels} {num_missing_pixels}"
                f" {num_bad_pixels} {num_melt_pixels}\n"
            )

        logger.info(f"Wrote {log_filepath}")
        logger.info(f"Wrote {error_filepath}")


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
    hemisphere: Hemisphere | Literal["both"],
    product_type: Literal["daily", "monthly", "both"],
    start_date: dt.date,
    end_date: dt.date,
):
    daily = False
    if product_type == "daily" or product_type == "both":
        daily = True

    monthly = False
    if product_type == "monthly" or product_type == "both":
        monthly = True

    hemispheres = get_args(Hemisphere) if hemisphere == "both" else [hemisphere]
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
