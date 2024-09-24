"""Code to run NRT ECDR processing."""

import copy
import datetime as dt
from pathlib import Path
from typing import Final, get_args

import click
import xarray as xr
from loguru import logger
from pm_tb_data._types import Hemisphere
from pm_tb_data.fetch.nsidc_0080 import get_nsidc_0080_tbs_from_disk

from seaice_ecdr.ancillary import ANCILLARY_SOURCES
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
)
from seaice_ecdr.platforms import (
    PLATFORM_CONFIG,
)
from seaice_ecdr.publish_daily import (
    get_complete_daily_filepath,
    make_publication_ready_ds,
)
from seaice_ecdr.tb_data import EcdrTbData, map_tbs_to_ecdr_channels
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
NRT_PLATFORM_ID: Final = "F18"
# Number of days to look previously for temporal interpolation (forward
# gap-filling)
NRT_DAYS_TO_LOOK_PREVIOUSLY: Final = 5
NRT_LAND_SPILLOVER_ALG: Final = "NT2_BT"


def compute_nrt_initial_daily_ecdr_dataset(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    ancillary_source: ANCILLARY_SOURCES = "CDRv5",
):
    """Create an initial daily ECDR NetCDF using NRT data"""
    # TODO: consider extracting the fetch-related code here to `tb_data` module.
    xr_tbs = get_nsidc_0080_tbs_from_disk(
        date=date,
        hemisphere=hemisphere,
        resolution=NRT_RESOLUTION,
        platform_id=NRT_PLATFORM_ID,
    )
    data_source: Final = "NSIDC-0080"

    ecdr_tbs = map_tbs_to_ecdr_channels(
        mapping=dict(
            v19="v19",
            h19="h19",
            v22="v22",
            v37="v37",
            h37="h37",
        ),
        xr_tbs=xr_tbs,
        hemisphere=hemisphere,
        resolution=NRT_RESOLUTION,
        date=date,
        data_source=data_source,
    )

    tb_data = EcdrTbData(
        tbs=ecdr_tbs,
        resolution="25",
        data_source=data_source,
        platform_id=NRT_PLATFORM_ID,
    )

    nrt_initial_ecdr_ds = compute_initial_daily_ecdr_dataset(
        date=date,
        hemisphere=hemisphere,
        tb_data=tb_data,
        land_spillover_alg=NRT_LAND_SPILLOVER_ALG,
        ancillary_source=ancillary_source,
    )

    return nrt_initial_ecdr_ds


def read_or_create_and_read_nrt_idecdr_ds(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    intermediate_output_dir: Path,
    overwrite: bool,
):
    platform = PLATFORM_CONFIG.get_platform_by_date(date)
    idecdr_filepath = get_idecdr_filepath(
        hemisphere=hemisphere,
        date=date,
        platform_id=platform.id,
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
    ancillary_source: ANCILLARY_SOURCES = "CDRv5",
) -> xr.Dataset:
    init_datasets = []
    for date in date_range(
        start_date=date - dt.timedelta(days=NRT_DAYS_TO_LOOK_PREVIOUSLY), end_date=date
    ):
        # TODO: support missing data periods (e.g., running for 2024-09-22 does
        # not work atm).
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
        ancillary_source=ancillary_source,
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
        is_nrt=True,
    )
    nrt_output_filepath = get_complete_daily_filepath(
        date=date,
        hemisphere=hemisphere,
        resolution=NRT_RESOLUTION,
        complete_output_dir=complete_output_dir,
        platform_id=NRT_PLATFORM_ID,
        is_nrt=True,
    )

    return nrt_output_filepath


def nrt_ecdr_for_day(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    base_output_dir: Path,
    overwrite: bool,
    ancillary_source: ANCILLARY_SOURCES = "CDRv5",
):
    """Create an initial daily ECDR NetCDF using NRT NSIDC-0080 F18 data."""
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
            is_nrt=True,
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
                ancillary_source=ancillary_source,
            )
            daily_ds = make_publication_ready_ds(
                intermediate_daily_ds=cde_ds,
                hemisphere=hemisphere,
            )

            # Update global attrs to reflect G10016 instead of G02202:
            daily_ds.attrs["id"] = "https://doi.org/10.7265/j0z0-4h87"
            daily_ds.attrs["metadata_link"] = (
                f"https://nsidc.org/data/g10016/versions/{ECDR_NRT_PRODUCT_VERSION.major_version_number}"
            )
            daily_ds.attrs["title"] = (
                "Near-Real-Time NOAA-NSIDC Climate Data Record of Passive Microwave"
                f" Sea Ice Concentration Version {ECDR_NRT_PRODUCT_VERSION.major_version_number}"
            )
            daily_ds.attrs["product_version"] = ECDR_NRT_PRODUCT_VERSION.version_str

            daily_ds.to_netcdf(nrt_output_filepath)
            logger.success(f"Wrote complete daily NRT NC file: {nrt_output_filepath}")

            # write checksum file for NRTs
            checksums_dir = nrt_output_filepath.parent / "checksums"
            write_checksum_file(
                input_filepath=nrt_output_filepath,
                output_dir=checksums_dir,
            )
        except Exception as e:
            logger.exception(f"Failed to create NRT ECDR for {date=} {hemisphere=}")
            raise e


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


@click.group(name="nrt")
def nrt_cli():
    """Run NRT Sea Ice ECDR."""
    ...


nrt_cli.add_command(nrt_ecdr_for_dates)
