import copy
import datetime as dt
from pathlib import Path
from typing import get_args

import click
import dateutil
import numpy as np
import xarray as xr
from datatree import DataTree
from loguru import logger
from pm_tb_data._types import NORTH, Hemisphere

from seaice_ecdr._types import ECDR_SUPPORTED_RESOLUTIONS
from seaice_ecdr.checksum import write_checksum_file
from seaice_ecdr.cli.util import datetime_to_date
from seaice_ecdr.constants import DEFAULT_BASE_OUTPUT_DIR
from seaice_ecdr.days_treated_differently import day_has_all_empty_fields
from seaice_ecdr.intermediate_daily import read_cdecdr_ds
from seaice_ecdr.nc_util import (
    add_coordinate_coverage_content_type,
    add_coordinates_attr,
    remove_FillValue_from_coordinate_vars,
)
from seaice_ecdr.platforms import PLATFORM_CONFIG, SUPPORTED_PLATFORM_ID
from seaice_ecdr.platforms.config import get_platform_id_from_name
from seaice_ecdr.util import (
    date_range,
    get_complete_output_dir,
    get_intermediate_output_dir,
    nrt_daily_filename,
    standard_daily_filename,
)

# TODO: consider extracting to config or a kwarg of this function for more
# flexible use with other platforms in the future.
# TODO: this is duplicated in `publish_monthly` and other locations. If this
# gets updated, update them all!
PROTOTYPE_PLATFORM_ID: SUPPORTED_PLATFORM_ID | None = None
PROTOTYPE_PLATFORM_DATA_GROUP_NAME: str | None = None
if PROTOTYPE_PLATFORM_ID:
    PROTOTYPE_PLATFORM_DATA_GROUP_NAME = "prototype_{PROTOTYPE_PLATFORM_ID}"


# TODO: this and `get_complete_daily_filepath` are identical (aside from var
# names) to `get_ecdr_filepath` and `get_ecdr_dir`.
def get_complete_daily_dir(
    *,
    complete_output_dir: Path,
    year: int,
) -> Path:
    ecdr_dir = complete_output_dir / "daily" / str(year)
    ecdr_dir.mkdir(parents=True, exist_ok=True)

    return ecdr_dir


def get_complete_daily_filepath(
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    complete_output_dir: Path,
    is_nrt: bool,
    platform_id: SUPPORTED_PLATFORM_ID,
):
    if is_nrt:
        ecdr_filename = nrt_daily_filename(
            hemisphere=hemisphere,
            date=date,
            platform_id=platform_id,
            resolution=resolution,
        )
    else:
        ecdr_filename = standard_daily_filename(
            hemisphere=hemisphere,
            date=date,
            platform_id=platform_id,
            resolution=resolution,
        )

    ecdr_dir = get_complete_daily_dir(
        complete_output_dir=complete_output_dir,
        year=date.year,
    )

    ecdr_filepath = ecdr_dir / ecdr_filename

    return ecdr_filepath


def make_publication_ready_ds(
    intermediate_daily_ds: xr.Dataset,
    hemisphere: Hemisphere,
) -> DataTree:
    """Take an intermediate daily dataset and prepare for publication.

    * Moves supplementary fields into "cdr_supplementary" group
    * Removes `valid_range` attr from coordinate vars
    * Adds `coverage_content_type: coordinate` attr to coordinate vars
    * Adds  `coordinates` attr to data variables
    """
    # For long data gaps, we explicitly require the CDR concentration field
    # to be filled with no-data values.  However, we still want to allow
    # other fields -- eg raw BT and NT fields and melt onset -- to get
    # calculated.  For those reasons, it is easiest to post-apply the
    # no-data values to the cdr_seaice_conc-related fields here.
    date = dateutil.parser.parse(intermediate_daily_ds.time_coverage_start).date()
    platform_id = get_platform_id_from_name(intermediate_daily_ds.platform)
    if day_has_all_empty_fields(
        platform_id=platform_id,
        hemisphere=hemisphere,
        date=date,
    ):
        intermediate_daily_ds["cdr_seaice_conc"].data[:] = np.nan
        intermediate_daily_ds["cdr_seaice_conc_stdev"].data[:] = -1
        intermediate_daily_ds["cdr_seaice_conc_qa_flag"].data[:] = 8
        intermediate_daily_ds["cdr_seaice_conc_interp_spatial_flag"].data[:] = 0
        intermediate_daily_ds["cdr_seaice_conc_interp_temporal_flag"].data[:] = 255
    # publication-ready daily data are grouped using `DataTree`.
    # Create a `cdr_supplementary` group for "supplemntary" fields
    cdr_supplementary_fields = [
        "raw_bt_seaice_conc",
        "raw_nt_seaice_conc",
        "surface_type_mask",
    ]
    # Melt onset only occurs in the NH.
    if hemisphere == NORTH:
        cdr_supplementary_fields.append("cdr_melt_onset_day")

    # Drop x, y, time coordinate variables. These will be inherited from the
    # root group.
    cdr_supplementary_group = intermediate_daily_ds[cdr_supplementary_fields].drop_vars(
        ["x", "y", "time"]
    )
    # remove attrs from supplementary group. These will be inherted from the
    # root group.
    cdr_supplementary_group.attrs = {}

    complete_daily_ds: DataTree = DataTree.from_dict(
        {
            "/": intermediate_daily_ds[
                [k for k in intermediate_daily_ds if k not in cdr_supplementary_fields]
            ],
            "cdr_supplementary": cdr_supplementary_group,
        }
    )

    # Remove `valid_range` from coordinate attrs
    # remove_valid_range_from_coordinate_vars(complete_daily_ds)
    remove_FillValue_from_coordinate_vars(complete_daily_ds)
    add_coordinate_coverage_content_type(complete_daily_ds)
    add_coordinates_attr(complete_daily_ds)

    return complete_daily_ds


# TODO: consider a better name. `publish` implies this function might actually
# publish it to a publicly accessible archive. That's something ops will do
# separately. This just generates the publication-ready nc file to it's expected
# publication staging location.
def publish_daily_nc(
    *,
    base_output_dir: Path,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
) -> Path:
    """Prepares a daily nc file for publication.

    If data for the provided prototype platform is available for the given date,
    this function adds that data to a prototype group in the output NC file. The
    output NC file's root-group variables are all taken from the default
    platforms given by the platform start date configuration.
    """

    intermediate_output_dir = get_intermediate_output_dir(
        base_output_dir=base_output_dir,
        hemisphere=hemisphere,
    )
    default_platform = PLATFORM_CONFIG.get_platform_by_date(date)
    default_daily_ds = read_cdecdr_ds(
        intermediate_output_dir=intermediate_output_dir,
        resolution=resolution,
        date=date,
        platform_id=default_platform.id,
        hemisphere=hemisphere,
        is_nrt=False,
    )

    # Prepare a dataset that's ready for publication.
    complete_daily_ds = make_publication_ready_ds(
        intermediate_daily_ds=default_daily_ds,
        hemisphere=hemisphere,
    )

    # Add the prototype group if there's data.
    if PROTOTYPE_PLATFORM_ID and PLATFORM_CONFIG.platform_available_for_date(
        platform_id=PROTOTYPE_PLATFORM_ID,
        date=date,
    ):
        try:
            prototype_daily_ds = read_cdecdr_ds(
                intermediate_output_dir=intermediate_output_dir,
                resolution=resolution,
                date=date,
                platform_id=PROTOTYPE_PLATFORM_ID,
                hemisphere=hemisphere,
                is_nrt=False,
            )
            cdr_var_fieldnames = [
                "cdr_seaice_conc",
                "cdr_seaice_conc_qa_flag",
                "cdr_seaice_conc_interp_spatial_flag",
                "cdr_seaice_conc_interp_temporal_flag",
                "cdr_seaice_conc_stdev",
            ]
            remap_names = {
                cdr_var_fieldname: cdr_var_fieldname.replace(
                    "cdr_", f"{PROTOTYPE_PLATFORM_ID}_"
                )
                for cdr_var_fieldname in cdr_var_fieldnames
            }
            prototype_subgroup = prototype_daily_ds[cdr_var_fieldnames].rename_vars(
                remap_names
            )
            # Rename ancillary variables.
            for var in prototype_subgroup.values():
                if "ancillary_variables" in var.attrs:
                    var.attrs["ancillary_variables"] = var.attrs[
                        "ancillary_variables"
                    ].replace("cdr_", f"{PROTOTYPE_PLATFORM_ID}_")
                if var.dims == ("time", "y", "x"):
                    var.attrs["coordinates"] = "time y x"
                if "long_name" in var.attrs:
                    if PROTOTYPE_PLATFORM_ID == "am2":
                        var.attrs["long_name"] = var.attrs["long_name"].replace(
                            "NOAA/NSIDC CDR of Passive Microwave", "AMSR2 Prototype"
                        )
                    else:
                        raise RuntimeError(
                            f"Unknown platform ID for naming: {PROTOTYPE_PLATFORM_ID}"
                        )

            # Drop x, y, and time coordinate variables. These will be inherited from the parent.
            prototype_subgroup = prototype_subgroup.drop_vars(["x", "y", "time"])
            # Retain only the group-specific global attrs
            prototype_subgroup.attrs = {
                k: v
                for k, v in prototype_subgroup.attrs.items()
                if k in ["source", "sensor", "platform"]
            }
            assert PROTOTYPE_PLATFORM_DATA_GROUP_NAME is not None
            complete_daily_ds[PROTOTYPE_PLATFORM_DATA_GROUP_NAME] = DataTree(
                data=prototype_subgroup,
            )
        except FileNotFoundError:
            logger.warning(
                f"Failed to find prototype daily file for {date=} {PROTOTYPE_PLATFORM_ID=}"
            )

    # write out finalized nc file.
    complete_output_dir = get_complete_output_dir(
        base_output_dir=base_output_dir,
        hemisphere=hemisphere,
    )
    platform = PLATFORM_CONFIG.get_platform_by_date(date)
    complete_daily_filepath = get_complete_daily_filepath(
        date=date,
        resolution=resolution,
        complete_output_dir=complete_output_dir,
        hemisphere=hemisphere,
        is_nrt=False,
        platform_id=platform.id,
    )

    # Ensure consistency of time units
    complete_daily_ds.time.encoding["units"] = "days since 1970-01-01"
    complete_daily_ds.time.encoding["calendar"] = "standard"

    complete_daily_ds = remove_FillValue_from_coordinate_vars(complete_daily_ds)
    complete_daily_ds.to_netcdf(complete_daily_filepath)
    logger.success(f"Staged NC file for publication: {complete_daily_filepath}")

    # Write checksum file for the complete daily output.
    checksums_subdir = complete_daily_filepath.relative_to(complete_output_dir).parent
    write_checksum_file(
        input_filepath=complete_daily_filepath,
        output_dir=complete_output_dir / "checksums" / checksums_subdir,
    )

    return complete_daily_filepath


def publish_daily_nc_for_dates(
    *,
    base_output_dir: Path,
    start_date: dt.date,
    end_date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
) -> list[Path]:
    dates = date_range(start_date=start_date, end_date=end_date)
    output_filepaths = []
    for date in dates:
        output_filepath = publish_daily_nc(
            base_output_dir=base_output_dir,
            hemisphere=hemisphere,
            resolution=resolution,
            date=date,
        )
        output_filepaths.append(output_filepath)

    return output_filepaths


@click.command(name="stage-for-publication")
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
@click.option(
    "-r",
    "--resolution",
    required=True,
    type=click.Choice(get_args(ECDR_SUPPORTED_RESOLUTIONS)),
)
def cli(
    *,
    date: dt.date,
    end_date: dt.date | None,
    hemisphere: Hemisphere,
    base_output_dir: Path,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
) -> None:
    """Stage daily NC files for publication."""
    if end_date is None:
        end_date = copy.copy(date)

    publish_daily_nc_for_dates(
        start_date=date,
        end_date=end_date,
        base_output_dir=base_output_dir,
        hemisphere=hemisphere,
        resolution=resolution,
    )
