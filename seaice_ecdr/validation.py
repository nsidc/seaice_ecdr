"""Validate outputs of the seaice ECDR.

This modlue provides code and a CLI for doing QC/validation of ECDR output
files.

The primary output from this code are two CSV files:

* error_seaice_{n|s}_daily_{start_year}_{end_year}.csv
* log_seaice_{n|s}_daily_{start_year}_{end_year}.csv

The error CSV file provides a code for each daily/monthly ECDR NetCDF file.

error file values:
    error = -9999 --> no CDR file on that date
    error = -999  --> CDR file exists but is empty (no valid concentrations)
    error = -99   --> CDR file exists and has values but some concentration values are in error
    error = -9    --> Melt is being flagged on a date that it should not be
    error = -2    --> More than 1000 missing values, indicates at least one missing swath
    error = -1    --> 100-1000 missing values, likely indicates at least part of
                    a missing swath
    error = 0     --> no significant problems in CDR fields

The log file provides more detail, giving the number of pixels of various
categories (e.g., "land", "melt", "ice-free").

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
    missing: pixels that are missing (no SIC). Should always be 0.
    bad: pixels that have an error (invalid sea ice value), should always be 0
    melt: pixels that have ice and are melting (north only, 1 March - 1 September only)
"""

import csv
import datetime as dt
import itertools
from collections import defaultdict
from pathlib import Path
from typing import Final, Literal, cast, get_args

import click
import xarray as xr
from loguru import logger
from pm_tb_data._types import Hemisphere

from seaice_ecdr.ancillary import (
    bitmask_value_for_meaning,
    flag_value_for_meaning,
)
from seaice_ecdr.cli.util import datetime_to_date
from seaice_ecdr.complete_daily_ecdr import get_ecdr_filepath
from seaice_ecdr.constants import DEFAULT_BASE_OUTPUT_DIR, ECDR_PRODUCT_VERSION
from seaice_ecdr.monthly import get_monthly_dir
from seaice_ecdr.util import date_range, get_num_missing_pixels

VALIDATION_RESOLUTION: Final = "12.5"

ERROR_FILE_BITMASK = dict(
    missing_file=-9999,
    file_exists_but_is_empty=-999,
    file_exists_but_conc_values_are_bad=-99,
    melt_flagged_on_wrong_day=-9,
    more_than_1000_missing_values=-2,
    between_100_and_1000_missing_values=-1,
    no_problems=0,
)


def get_validation_dir(*, base_output_dir: Path) -> Path:
    validation_dir = base_output_dir / "validation"
    validation_dir.mkdir(exist_ok=True)

    return validation_dir


MONTHLY_TEMPORAL_FIELDS = [
    "year",
    "month",
]

DAILY_TEMPORAL_FIELDS = [
    *MONTHLY_TEMPORAL_FIELDS,
    "day",
]

SHARED_LOG_FIELDS = [
    "total",
    "ice",
    "land",
    "coast",
    "lake",
    "pole",
    "oceanmask",
    "ice-free",
    "missing",
    "bad",
    "melt",
]

SHARED_ERROR_FIELDS = ["error_code"]

Product = Literal["daily", "monthly"]


def get_log_fields(*, product: Product) -> list[str]:
    if product == "daily":
        log_fields = [*DAILY_TEMPORAL_FIELDS, *SHARED_LOG_FIELDS]
    else:
        log_fields = [*MONTHLY_TEMPORAL_FIELDS, *SHARED_LOG_FIELDS]

    return log_fields


def write_log_entry(
    *,
    product: Product,
    csv_writer: csv.DictWriter,
    entry: dict,
):
    log_fields = get_log_fields(product=product)

    log_entry: dict = defaultdict(None)

    for k, v in entry.items():
        if k not in log_fields:
            raise RuntimeError(f"Unexpected log field {k} for {product=}")
        log_entry[k] = v

    csv_writer.writerow(log_entry)


def get_error_fields(*, product: Product) -> list[str]:
    if product == "daily":
        error_fields = [*DAILY_TEMPORAL_FIELDS, *SHARED_ERROR_FIELDS]
    else:
        error_fields = [*MONTHLY_TEMPORAL_FIELDS, *SHARED_ERROR_FIELDS]

    return error_fields


def write_error_entry(
    *,
    product: Product,
    csv_writer: csv.DictWriter,
    entry: dict,
):
    error_fields = get_error_fields(product=product)

    error_entry: dict = defaultdict(None)
    for k, v in entry.items():
        if k not in error_fields:
            raise RuntimeError(f"Unexpected error field {k}")
        error_entry[k] = v

    csv_writer.writerow(error_entry)


def make_validation_dict_for_missing_file() -> dict:
    log_dict: dict = {}
    error_value = ERROR_FILE_BITMASK["missing_file"]
    error_dict = dict(
        error_code=error_value,
    )
    validation_dict = dict(
        error=error_dict,
        log=log_dict,
    )
    return validation_dict


def get_error_code(
    *,
    num_total_pixels: int,
    num_bad_pixels: int,
    num_missing_pixels: int,
    num_melt_pixels: int,
    date: dt.date,
    data_fp: Path,
):
    error_code = ERROR_FILE_BITMASK["no_problems"]
    # This should never happen.
    if num_total_pixels == num_missing_pixels:
        error_code += ERROR_FILE_BITMASK["file_exists_but_is_empty"]

    if num_bad_pixels > 0:
        logger.warning(f"Found {num_bad_pixels} bad pixels for {data_fp}")
        error_code += ERROR_FILE_BITMASK["file_exists_but_conc_values_are_bad"]

    # melt flag on the wrong day
    melt_season_start_for_year = dt.date(date.year, 3, 1)
    melt_season_end_for_year = dt.date(date.year, 9, 1)
    date_in_melt_season = melt_season_start_for_year <= date <= melt_season_end_for_year
    if (num_melt_pixels > 0) and not date_in_melt_season:
        error_code += ERROR_FILE_BITMASK["melt_flagged_on_wrong_day"]

    if num_missing_pixels >= 1000:
        error_code += ERROR_FILE_BITMASK["more_than_1000_missing_values"]

    if (num_missing_pixels > 100) and (num_missing_pixels < 1000):
        error_code += ERROR_FILE_BITMASK["between_100_and_1000_missing_values"]

    return error_code


def get_pixel_counts(
    *,
    ds: xr.Dataset,
    product: Product,
    hemisphere: Hemisphere,
) -> dict[str, int]:
    """Return pixel counts from the daily or monthly ds.

    Each key of the resulting dictionary is an element of `SHARED_LOG_FIELDS`.
    """
    conc_var_name = "cdr_seaice_conc"
    if product == "monthly":
        conc_var_name = conc_var_name + "_monthly"
    seaice_conc_var = ds[conc_var_name]

    qa_var_name = "qa_of_" + conc_var_name
    qa_var = ds[qa_var_name]

    # Handle the log file first. It contains information on the # of
    # pixels of each type in the surface_type_mask and the ocean mask
    # (via qa_of_cdr_seaice_conc)
    # total ice land coast lake pole oceanmask ice-free missing bad melt
    total_num_pixels = len(ds.x) * len(ds.y)
    # Areas where there is a concentration detected.
    num_ice_pixels = int(((seaice_conc_var > 0) & (seaice_conc_var <= 1)).sum())

    # Surface value counts. These should be the same for every day.
    surf_value_counts = {}
    for flag in ("land", "coast", "lake", "polehole_mask"):
        if flag == "polehole_mask" and hemisphere == "south":
            surf_value_counts[flag] = 0
            continue

        flag_value = flag_value_for_meaning(
            var=ds.surface_type_mask,
            meaning=flag,
        )
        num_flag_pixels = int((ds.surface_type_mask == flag_value).sum())
        surf_value_counts[flag] = num_flag_pixels

    # Number of oceanmask (invalid ice mask) pixels
    invalid_ice_bitmask_value = bitmask_value_for_meaning(
        var=qa_var,
        meaning="invalid_ice_mask_applied",
    )
    invalid_ice_mask = (qa_var & invalid_ice_bitmask_value) > 0
    num_oceanmask_pixels = int(invalid_ice_mask.sum())

    # Ice-free pixels (conc == 0)
    num_ice_free_pixels = int((seaice_conc_var == 0).sum())

    # Get the number of missing pixels in the cdr conc field.
    num_missing_pixels = get_num_missing_pixels(
        seaice_conc_var=seaice_conc_var,
        hemisphere=hemisphere,
        resolution=VALIDATION_RESOLUTION,
    )

    # Per CDR v4, "bad" ice pixels are outside the expected range.
    # Note: we use 0.0999 instead of 0.1 because SIC values of 10% are
    # decoded from the integer value of 10 to 0.1, which is represented
    # as 0.099999 as a floating point data.
    gt_100_sic = int((seaice_conc_var > 1).sum())
    if product == "daily":
        less_than_10_sic = int(
            ((seaice_conc_var > 0) & (seaice_conc_var <= 0.0999)).sum()
        )
        num_bad_pixels = less_than_10_sic + gt_100_sic
    else:
        # We expect monthly data to contain values < 10%.
        num_bad_pixels = gt_100_sic

    # Number of melt pixels
    if hemisphere == "north":
        _melt_start_meaning = "melt_start_detected"
        if product == "monthly":
            _melt_start_meaning = "at_least_one_day_during_month_has_melt_detected"
        melt_start_detected_bitmask_value = bitmask_value_for_meaning(
            var=qa_var,
            meaning=_melt_start_meaning,
        )
        melt_start_detected_mask = (qa_var & melt_start_detected_bitmask_value) > 0
        num_melt_pixels = int(melt_start_detected_mask.sum())
    else:
        num_melt_pixels = 0

    pixel_counts = {
        "total": total_num_pixels,
        "ice": num_ice_pixels,
        "land": surf_value_counts["land"],
        "coast": surf_value_counts["coast"],
        "lake": surf_value_counts["lake"],
        "pole": surf_value_counts["polehole_mask"],
        "oceanmask": num_oceanmask_pixels,
        "ice-free": num_ice_free_pixels,
        "missing": num_missing_pixels,
        "bad": num_bad_pixels,
        "melt": num_melt_pixels,
    }

    return pixel_counts


def make_validation_dict(
    *,
    data_fp: Path,
    product: Product,
    date: dt.date,
    hemisphere: Hemisphere,
) -> dict:
    ds = xr.open_dataset(data_fp)

    pixel_counts = get_pixel_counts(
        ds=ds,
        product=product,
        hemisphere=hemisphere,
    )

    error_code = get_error_code(
        date=date,
        data_fp=data_fp,
        num_total_pixels=pixel_counts["total"],
        num_bad_pixels=pixel_counts["bad"],
        num_missing_pixels=pixel_counts["missing"],
        num_melt_pixels=pixel_counts["melt"],
    )

    validation_dict = dict(
        error=dict(error_code=error_code),
        log=pixel_counts,
    )

    return validation_dict


def validate_outputs(
    *,
    hemisphere: Hemisphere,
    base_output_dir: Path,
    start_date: dt.date,
    end_date: dt.date,
    product: Product,
) -> dict[str, Path]:
    """Create validation logs for daily outputs.

    Creates two CSV files:

    * log_seaice_{n|s}_daily_{start_year}_{end_year}.csv. Contains the following
      fields: [year, month, day, total, ice, land, coast, lake, pole, oceanmask,
      ice-free, missing, bad, melt].
    * error_seaice_{n|s}_daily_{start_year}_{end_year}.csv. Contains the
      following fields: [year, month, day, error_code]
    """
    validation_dir = get_validation_dir(base_output_dir=base_output_dir)
    log_filepath = (
        validation_dir
        / f"log_seaice_{hemisphere[0]}_{product}_{start_date.year}_{end_date.year}.csv"
    )
    error_filepath = (
        validation_dir
        / f"error_seaice_{hemisphere[0]}_{product}_{start_date.year}_{end_date.year}.csv"
    )
    log_fields = get_log_fields(product=product)
    error_fields = get_error_fields(product=product)
    with open(log_filepath, "w") as log_file, open(error_filepath, "w") as error_file:
        # Write headers
        log_writer = csv.DictWriter(log_file, fieldnames=log_fields, delimiter=",")
        log_writer.writeheader()
        error_writer = csv.DictWriter(
            error_file, fieldnames=error_fields, delimiter=","
        )
        error_writer.writeheader()
        if product == "daily":
            for date in date_range(start_date=start_date, end_date=end_date):
                data_fp = get_ecdr_filepath(
                    date=date,
                    hemisphere=hemisphere,
                    resolution=VALIDATION_RESOLUTION,
                    base_output_dir=base_output_dir,
                    is_nrt=False,
                )

                if not data_fp.is_file():
                    logger.warning(f"Expected daily file is missing: {data_fp}")
                    validation_dict = make_validation_dict_for_missing_file()
                else:
                    validation_dict = make_validation_dict(
                        data_fp=data_fp,
                        product=product,
                        date=date,
                        hemisphere=hemisphere,
                    )

                write_error_entry(
                    product=product,
                    csv_writer=error_writer,
                    entry=dict(
                        year=date.year,
                        month=date.month,
                        day=date.day,
                        **validation_dict["error"],
                    ),
                )
                write_log_entry(
                    product=product,
                    csv_writer=log_writer,
                    entry=dict(
                        year=date.year,
                        month=date.month,
                        day=date.day,
                        **validation_dict["log"],
                    ),
                )
        else:
            years = range(start_date.year, end_date.year + 1)
            months = range(start_date.month, end_date.month + 1)
            for year, month in itertools.product(years, months):
                monthly_dir = get_monthly_dir(base_output_dir=base_output_dir)

                # monthly filepaths should have the form
                # "sic_ps{n|s}12.5_{YYYYMM}_{sat}_v05r00.nc"
                expected_fn_glob = f"sic_ps{hemisphere[0]}12.5_{year}{month:02}_*_{ECDR_PRODUCT_VERSION}.nc"
                results = list(monthly_dir.glob(expected_fn_glob))
                if not results:
                    validation_dict = make_validation_dict_for_missing_file()
                else:
                    validation_dict = make_validation_dict(
                        data_fp=results[0],
                        product=product,
                        date=dt.date(year, month, 1),
                        hemisphere=hemisphere,
                    )
                write_error_entry(
                    product=product,
                    csv_writer=error_writer,
                    entry=dict(
                        year=year,
                        month=month,
                        **validation_dict["error"],
                    ),
                )
                write_log_entry(
                    product=product,
                    csv_writer=log_writer,
                    entry=dict(
                        year=year,
                        month=month,
                        **validation_dict["log"],
                    ),
                )

    logger.success(f"Wrote {log_filepath}")
    logger.success(f"Wrote {error_filepath}")

    return dict(
        log_filepath=log_filepath,
        error_filepath=error_filepath,
    )


@click.command(
    name="validate-outputs",
    help="Create CSV files used to validate ECDR outputs.",
)
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
    default=DEFAULT_BASE_OUTPUT_DIR,
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
    base_output_dir: Path,
    hemisphere: Hemisphere | Literal["both"],
    product_type: Literal["daily", "monthly", "both"],
    start_date: dt.date,
    end_date: dt.date,
):
    # The data should be organized by hemisphere.
    base_output_dir = base_output_dir / hemisphere
    base_output_dir.mkdir(exist_ok=True)

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
            validate_outputs(
                hemisphere=hemisphere,
                base_output_dir=base_output_dir,
                start_date=start_date,
                end_date=end_date,
                product="daily",
            )
        if monthly:
            validate_outputs(
                hemisphere=hemisphere,
                base_output_dir=base_output_dir,
                start_date=start_date,
                end_date=end_date,
                product="monthly",
            )
