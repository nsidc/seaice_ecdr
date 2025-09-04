"""Code to run daily NRT ECDR processing.


 The code must be run with the environment variable
`PLATFORM_START_DATES_CONFIG_FILEPATH` set to the NRT config file
(`nrt_platform_start_dates.yml`) for the code to run properly. Ideally in the
future we can refactor the code to support configuring the platform start dates
at any point rather than needing an at-import-time setup as we currently do.
"""

import copy
import datetime as dt
from functools import cache
from pathlib import Path
from typing import Final, Literal, cast, get_args

import click
import xarray as xr
from loguru import logger
from pm_tb_data._types import Hemisphere

from seaice_ecdr._types import ECDR_SUPPORTED_RESOLUTIONS
from seaice_ecdr.checksum import write_checksum_file
from seaice_ecdr.cli.util import datetime_to_date
from seaice_ecdr.constants import DEFAULT_BASE_NRT_OUTPUT_DIR, ECDR_NRT_PRODUCT_VERSION
from seaice_ecdr.initial_daily_ecdr import (
    compute_initial_daily_ecdr_dataset,
    get_idecdr_filepath,
    write_ide_netcdf,
)
from seaice_ecdr.intermediate_daily import (
    complete_daily_ecdr_ds,
    get_ecdr_filepath,
)
from seaice_ecdr.platforms import PLATFORM_CONFIG
from seaice_ecdr.publish_daily import (
    get_complete_daily_filepath,
    make_publication_ready_ds,
)
from seaice_ecdr.tb_data import (
    get_25km_am2_tbs_from_nsidc_0802,
)
from seaice_ecdr.temporal_composite_daily import (
    get_tie_filepath,
    temporal_interpolation,
    write_tie_netcdf,
)
from seaice_ecdr.util import (
    date_range,
    get_complete_output_dir,
    get_intermediate_output_dir,
)

NRT_RESOLUTION: Final = "25"
# Number of days to look previously for temporal interpolation (forward
# gap-filling)
NRT_DAYS_TO_LOOK_PREVIOUSLY: Final = 5
NRT_LAND_SPILLOVER_ALG: Final = "NT2_BT"
NRT_SUPPORTED_PLATFORM_ID = Literal["am2"]
# TODO: this variable must be kept in sync with other simliar variables in other
# modules! If this changes here, make sure it is updated elsewhere too! In the
# future, this should be config-driven.
PROTOTYPE_PLATFORM_ID: str | None = None


@cache
def _get_nrt_platform_id() -> NRT_SUPPORTED_PLATFORM_ID:
    # NRT processing considers only a single platform at a time. Raise an error
    # if more or less than 1 is given.
    if (num_start_dates := len(PLATFORM_CONFIG.cdr_platform_start_dates)) != 1:
        raise RuntimeError(
            "NRT Processing expects a single platform start date."
            f" Got {num_start_dates}."
        )

    nrt_platform_id = PLATFORM_CONFIG.cdr_platform_start_dates[0].platform_id
    if nrt_platform_id not in get_args(NRT_SUPPORTED_PLATFORM_ID):
        raise RuntimeError(f"NRT processing is not defined for {nrt_platform_id}")

    nrt_platform_id = cast(NRT_SUPPORTED_PLATFORM_ID, nrt_platform_id)

    return nrt_platform_id


def compute_nrt_initial_daily_ecdr_dataset(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
):
    """Create an initial daily ECDR NetCDF using NRT data"""
    platform_id = _get_nrt_platform_id()

    if platform_id == "am2":
        tb_data = get_25km_am2_tbs_from_nsidc_0802(
            date=date,
            hemisphere=hemisphere,
        )
    else:
        raise NotImplementedError(
            f"Daily NRT processing is not defined for {platform_id}"
        )

    tb_data = get_25km_am2_tbs_from_nsidc_0802(
        date=date,
        hemisphere=hemisphere,
    )

    nrt_initial_ecdr_ds = compute_initial_daily_ecdr_dataset(
        date=date,
        hemisphere=hemisphere,
        tb_data=tb_data,
        land_spillover_alg=NRT_LAND_SPILLOVER_ALG,
    )

    return nrt_initial_ecdr_ds


def read_or_create_and_read_nrt_idecdr_ds(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    intermediate_output_dir: Path,
    overwrite: bool,
):
    idecdr_filepath = get_idecdr_filepath(
        hemisphere=hemisphere,
        date=date,
        platform_id="am2",
        intermediate_output_dir=intermediate_output_dir,
        resolution=NRT_RESOLUTION,
    )

    if overwrite or not idecdr_filepath.is_file():
        nrt_initial_ecdr_ds = compute_nrt_initial_daily_ecdr_dataset(
            date=date,
            hemisphere=hemisphere,
        )

        excluded_idecdr_fields = [
            "h19_day",
            "v19_day",
            "v22_day",
            "h37_day",
            "v37_day",
            # "h19_day_si",  # include this field for melt onset calculation
            "v19_day_si",
            "v22_day_si",
            # "h37_day_si",  # include this field for melt onset calculation
            "v37_day_si",
            "non_ocean_mask",
            "invalid_ice_mask",
            "pole_mask",
            "bt_weather_mask",
            "nt_weather_mask",
            "missing_tb_mask",
        ]
        write_ide_netcdf(
            ide_ds=nrt_initial_ecdr_ds,
            output_filepath=idecdr_filepath,
            excluded_fields=excluded_idecdr_fields,
        )

    ide_ds = xr.load_dataset(idecdr_filepath)
    return ide_ds


def temporally_interpolated_nrt_ecdr_dataset(
    *,
    hemisphere: Hemisphere,
    date: dt.date,
    intermediate_output_dir: Path,
    overwrite: bool,
) -> xr.Dataset:
    init_datasets = []
    for date in date_range(
        start_date=date - dt.timedelta(days=NRT_DAYS_TO_LOOK_PREVIOUSLY), end_date=date
    ):
        init_dataset = read_or_create_and_read_nrt_idecdr_ds(
            date=date,
            hemisphere=hemisphere,
            overwrite=overwrite,
            intermediate_output_dir=intermediate_output_dir,
        )
        init_datasets.append(init_dataset)

    data_stack = xr.concat(init_datasets, dim="time").sortby("time")

    temporally_interpolated_ds = temporal_interpolation(
        date=date,
        hemisphere=hemisphere,
        resolution=NRT_RESOLUTION,
        data_stack=data_stack,
        interp_range=NRT_DAYS_TO_LOOK_PREVIOUSLY,
        one_sided_limit=NRT_DAYS_TO_LOOK_PREVIOUSLY,
    )

    return temporally_interpolated_ds


def read_or_create_and_read_nrt_tiecdr_ds(
    *,
    hemisphere: Hemisphere,
    date: dt.date,
    intermediate_output_dir: Path,
    overwrite: bool,
) -> xr.Dataset:
    tie_filepath = get_tie_filepath(
        date=date,
        hemisphere=hemisphere,
        resolution=NRT_RESOLUTION,
        intermediate_output_dir=intermediate_output_dir,
    )

    if overwrite or not tie_filepath.is_file():
        nrt_temporally_interpolated = temporally_interpolated_nrt_ecdr_dataset(
            hemisphere=hemisphere,
            date=date,
            intermediate_output_dir=intermediate_output_dir,
            overwrite=overwrite,
        )

        write_tie_netcdf(
            tie_ds=nrt_temporally_interpolated,
            output_filepath=tie_filepath,
        )

    tie_ds = xr.load_dataset(tie_filepath)
    return tie_ds


def get_nrt_complete_daily_filepath(
    *, base_output_dir: Path, hemisphere: Hemisphere, date: dt.date
) -> Path:
    complete_output_dir = get_complete_output_dir(
        base_output_dir=base_output_dir,
        hemisphere=hemisphere,
    )
    nrt_output_filepath = get_complete_daily_filepath(
        date=date,
        hemisphere=hemisphere,
        resolution=NRT_RESOLUTION,
        complete_output_dir=complete_output_dir,
        platform_id="am2",
        is_nrt=True,
    )

    return nrt_output_filepath


def override_attrs_for_nrt(
    *,
    publication_ready_ds: xr.DataTree,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
) -> xr.DataTree:
    platform_id = _get_nrt_platform_id()

    if platform_id != "am2":
        raise NotImplementedError(f"NRT Processing is not defined for {platform_id}")

    override_for_nrt = publication_ready_ds.copy()

    override_for_nrt.attrs["summary"] = (
        f"This data set provides a near-real-time (NRT) passive microwave sea ice concentration climate data record (CDR) based on gridded brightness temperatures (TBs) from the Global Change Observation Mission 1st-Water (GCOM-W1) passive microwave radiometer: Advanced Microwave Scanning Radiometer 2 (AMSR2). The sea ice concentration CDR is an estimate of sea ice concentration that is produced by combining concentration estimates from two algorithms developed at the NASA Goddard Space Flight Center (GSFC): the NASA Team algorithm and the Bootstrap algorithm. The individual algorithms are used to process and combine brightness temperature data at NSIDC. This product is designed to provide an NRT time series of sea ice concentrations (the fraction, or percentage, of ocean area covered by sea ice). The data are gridded on the NSIDC polar stereographic grid with {resolution} x {resolution} km grid cells and are available in NetCDF file format. Each file contains a variable with the CDR concentration values as well as variables that hold the NASA Team and Bootstrap processed concentrations for reference. Variables containing standard deviation, quality flags, and projection information are also included."
    )

    override_for_nrt.attrs["id"] = "https://doi.org/10.7265/j0z0-4h87"
    link_to_dataproduct = f"https://nsidc.org/data/g10016/versions/{ECDR_NRT_PRODUCT_VERSION.major_version_number}"
    override_for_nrt.attrs["metadata_link"] = link_to_dataproduct
    override_for_nrt.attrs["title"] = (
        "Near-Real-Time NOAA-NSIDC Climate Data Record of Passive Microwave"
        f" Sea Ice Concentration Version {ECDR_NRT_PRODUCT_VERSION.major_version_number}"
    )
    override_for_nrt.attrs["product_version"] = ECDR_NRT_PRODUCT_VERSION.version_str

    return override_for_nrt


def hack_daily_cdr_vars_for_prototype_platform(
    daily_ds: xr.DataTree,
):
    """Hack prepare daily prototype platform files for publication.
    * rename `cdr_` variables with `{platform_id}_`
    * Remove from cdr_supplementary group:
      * raw_bt_seaice_conc
      * raw_nt_seaice_conc
      * cdr_melt_onset_day
    """
    platform_id = _get_nrt_platform_id()
    if not PROTOTYPE_PLATFORM_ID or platform_id != PROTOTYPE_PLATFORM_ID:
        # Make no changes. The given platform should not be considered a
        # "prototype".
        return daily_ds

    root_as_xr = daily_ds.root.to_dataset()

    suppl_as_xr = daily_ds.cdr_supplementary.to_dataset()
    vars_to_drop = ["raw_bt_seaice_conc", "raw_nt_seaice_conc"]
    if "cdr_melt_onset_day" in suppl_as_xr.variables:
        vars_to_drop.append("cdr_melt_onset_day")

    suppl_as_xr = suppl_as_xr.drop_vars(vars_to_drop)

    root_remapping = {}
    for var in root_as_xr.variables:
        if var.startswith("cdr_"):
            new_name = var.replace("cdr_", f"{platform_id}_")
            root_remapping[var] = new_name

    root_as_xr = root_as_xr.rename(root_remapping)

    renamed_vars_ds = xr.DataTree.from_dict(
        {
            "/": root_as_xr,
            "cdr_supplementary": suppl_as_xr,
        }
    )

    return renamed_vars_ds


def nrt_ecdr_for_day(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    base_output_dir: Path,
    overwrite: bool,
):
    """Create an initial daily ECDR NetCDF using NRT NSIDC-0802 AMSR2 data."""
    nrt_output_filepath = get_nrt_complete_daily_filepath(
        base_output_dir=base_output_dir,
        hemisphere=hemisphere,
        date=date,
    )
    if nrt_output_filepath.is_file() and not overwrite:
        logger.info(f"File for {date=} already exists ({nrt_output_filepath}).")
        return

    if not nrt_output_filepath.is_file() or overwrite:
        intermediate_output_dir = get_intermediate_output_dir(
            base_output_dir=base_output_dir,
            hemisphere=hemisphere,
        )
        try:
            tiecdr_ds = read_or_create_and_read_nrt_tiecdr_ds(
                hemisphere=hemisphere,
                date=date,
                intermediate_output_dir=intermediate_output_dir,
                overwrite=overwrite,
            )

            cde_ds = complete_daily_ecdr_ds(
                tie_ds=tiecdr_ds,
                date=date,
                hemisphere=hemisphere,
                resolution=NRT_RESOLUTION,
                intermediate_output_dir=intermediate_output_dir,
                is_nrt=True,
            )
            # Write the daily intermediate file. This is used by the monthly NRT
            # processing to produce the monthly fields.
            cde_ds_filepath = get_ecdr_filepath(
                date=date,
                hemisphere=hemisphere,
                resolution=NRT_RESOLUTION,
                intermediate_output_dir=intermediate_output_dir,
                platform_id="am2",
                is_nrt=True,
            )
            cde_ds.to_netcdf(
                cde_ds_filepath,
            )

            # Prepare the ds for publication
            daily_ds = make_publication_ready_ds(
                intermediate_daily_ds=cde_ds,
                hemisphere=hemisphere,
            )

            # Update global attrs to reflect G10016 instead of G02202:
            daily_ds = override_attrs_for_nrt(
                publication_ready_ds=daily_ds,
                resolution=NRT_RESOLUTION,
            )

            # If a prototype platform is defined, hack the variables so that
            # they have the correct names.
            daily_ds = hack_daily_cdr_vars_for_prototype_platform(
                daily_ds=daily_ds,
            )

            daily_ds.to_netcdf(nrt_output_filepath)
            logger.success(f"Wrote complete daily NRT NC file: {nrt_output_filepath}")

            # write checksum file for NRTs
            complete_output_dir = get_complete_output_dir(
                base_output_dir=base_output_dir,
                hemisphere=hemisphere,
            )
            checksums_dir = (
                complete_output_dir
                / "checksums"
                / nrt_output_filepath.relative_to(complete_output_dir).parent
            )
            write_checksum_file(
                input_filepath=nrt_output_filepath,
                output_dir=checksums_dir,
            )
        except Exception as e:
            logger.exception(f"Failed to create NRT ECDR for {date=} {hemisphere=}")
            raise e


@click.command(name="nrt")
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
def nrt_ecdr_for_dates(
    *,
    date: dt.date,
    end_date: dt.date | None,
    hemisphere: Hemisphere,
    base_output_dir: Path,
    overwrite: bool,
):
    if end_date is None:
        end_date = copy.copy(date)

    for date in date_range(start_date=date, end_date=end_date):
        nrt_ecdr_for_day(
            date=date,
            hemisphere=hemisphere,
            base_output_dir=base_output_dir,
            overwrite=overwrite,
        )
