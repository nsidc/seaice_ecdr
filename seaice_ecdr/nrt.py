"""Code to run NRT ECDR processing.

TODO:

* Think about a missing value. The nrt data will potentially contain missing
  data since it can only look to the past for filling missing data. A user could
  probably look at the QA field to find out that spatial/temporal interp hasn't
  happened for a `np.nan` cell, but this would be extra work. Maybe a fine
  trade-off for now.
* Figure out how to leverage the temporal composite and finalization
  code. Currently most of our code relies on hard-coded expectations about the
  platform (based on date) and where the data live (ecdr_data_dir). The data dir
  is easy enough to change, but we may need to override the platform for NRT
  processing. Some way to tell the code on a global level that it's dealing w/
  NRT maybe?
"""

import datetime as dt
from pathlib import Path
from typing import Final, get_args

import click
import xarray as xr
from loguru import logger
from pm_tb_data._types import Hemisphere
from pm_tb_data.fetch.amsr.lance_amsr2 import (
    access_local_lance_data,
    download_latest_lance_files,
)

from seaice_ecdr.cli.util import datetime_to_date
from seaice_ecdr.complete_daily_ecdr import (
    complete_daily_ecdr_ds,
    get_ecdr_filepath,
    write_cde_netcdf,
)
from seaice_ecdr.constants import LANCE_NRT_DATA_DIR, NRT_BASE_OUTPUT_DIR
from seaice_ecdr.initial_daily_ecdr import (
    compute_initial_daily_ecdr_dataset,
    get_idecdr_filepath,
    write_ide_netcdf,
)
from seaice_ecdr.platforms import (
    get_platform_by_date,
)
from seaice_ecdr.tb_data import EcdrTbData, map_tbs_to_ecdr_channels
from seaice_ecdr.temporal_composite_daily import (
    get_tie_filepath,
    temporal_interpolation,
    write_tie_netcdf,
)
from seaice_ecdr.util import create_err_logfile, date_range

LANCE_RESOLUTION: Final = "12.5"


def compute_nrt_initial_daily_ecdr_dataset(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
):
    """Create an initial daily ECDR NetCDF using NRT LANCE AMSR2 data."""
    # TODO: handle missing data case.
    xr_tbs = access_local_lance_data(
        date=date,
        hemisphere=hemisphere,
        data_dir=LANCE_NRT_DATA_DIR,
    )
    data_source: Final = "LANCE AU_SI12"
    platform: Final = "am2"

    ecdr_tbs = map_tbs_to_ecdr_channels(
        # TODO/Note: this mapping is the same as used for `am2`.
        mapping=dict(
            v19="v18",
            h19="h18",
            v22="v23",
            v37="v36",
            h37="h36",
        ),
        xr_tbs=xr_tbs,
        hemisphere=hemisphere,
        resolution=LANCE_RESOLUTION,
        date=date,
        data_source=data_source,
    )

    tb_data = EcdrTbData(
        tbs=ecdr_tbs,
        resolution=LANCE_RESOLUTION,
        data_source=data_source,
        platform=platform,
    )

    nrt_initial_ecdr_ds = compute_initial_daily_ecdr_dataset(
        date=date,
        hemisphere=hemisphere,
        tb_data=tb_data,
    )

    return nrt_initial_ecdr_ds


def read_or_create_and_read_nrt_idecdr_ds(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    ecdr_data_dir: Path,
    overwrite: bool,
):
    platform = get_platform_by_date(date)
    idecdr_filepath = get_idecdr_filepath(
        hemisphere=hemisphere,
        date=date,
        platform=platform,
        ecdr_data_dir=ecdr_data_dir,
        resolution="12.5",
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
    ecdr_data_dir: Path,
    overwrite: bool,
    days_to_look_previously: int = 5,
) -> xr.Dataset:
    init_datasets = []
    for date in date_range(
        start_date=date - dt.timedelta(days=days_to_look_previously), end_date=date
    ):
        init_dataset = read_or_create_and_read_nrt_idecdr_ds(
            date=date,
            hemisphere=hemisphere,
            overwrite=overwrite,
            ecdr_data_dir=ecdr_data_dir,
        )
        init_datasets.append(init_dataset)

    data_stack = xr.concat(init_datasets, dim="time").sortby("time")

    temporally_interpolated_ds = temporal_interpolation(
        date=date,
        hemisphere=hemisphere,
        resolution=LANCE_RESOLUTION,
        data_stack=data_stack,
        interp_range=days_to_look_previously,
        one_sided_limit=days_to_look_previously,
    )

    return temporally_interpolated_ds


def read_or_create_and_read_nrt_tiecdr_ds(
    *,
    hemisphere: Hemisphere,
    date: dt.date,
    ecdr_data_dir: Path,
    overwrite: bool,
    days_to_look_previously: int = 4,
) -> xr.Dataset:
    tie_filepath = get_tie_filepath(
        date=date,
        hemisphere=hemisphere,
        resolution=LANCE_RESOLUTION,
        ecdr_data_dir=ecdr_data_dir,
    )

    if overwrite or not tie_filepath.is_file():
        nrt_temporally_interpolated = temporally_interpolated_nrt_ecdr_dataset(
            hemisphere=hemisphere,
            date=date,
            ecdr_data_dir=ecdr_data_dir,
            overwrite=overwrite,
            days_to_look_previously=days_to_look_previously,
        )

        write_tie_netcdf(
            tie_ds=nrt_temporally_interpolated,
            output_filepath=tie_filepath,
        )

    tie_ds = xr.load_dataset(tie_filepath)
    return tie_ds


@click.command(name="download-latest-nrt-data")
@click.option(
    "-o",
    "--output-dir",
    required=True,
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
    show_default=True,
    default=LANCE_NRT_DATA_DIR,
    help="Directory in which LANCE AMSR2 NRT files will be downloaded to.",
)
@click.option(
    "--overwrite",
    is_flag=True,
    show_default=True,
    default=False,
    help="Overwrite existing LANCE files.",
)
def download_latest_nrt_data(*, output_dir: Path, overwrite: bool) -> None:
    """Download the latest NRT LANCE AMSR2 data to the specified output directory.

    Files are only downloaded if they are considered 'complete' and ready for
    NRT processing for the ECDR product. This means that the latest available
    date of data is never downloaded, as it is considered provisional/subject to
    change until a new day's worth of data is available.
    """
    download_latest_lance_files(output_dir=output_dir, overwrite=overwrite)


@click.command(name="nrt-ecdr-for-day")
@click.option(
    "-d",
    "--date",
    required=True,
    type=click.DateTime(formats=("%Y-%m-%d", "%Y%m%d", "%Y.%m.%d")),
    callback=datetime_to_date,
)
@click.option(
    "-h",
    "--hemisphere",
    required=True,
    type=click.Choice(get_args(Hemisphere)),
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
    default=NRT_BASE_OUTPUT_DIR,
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
    help=(
        "Overwrite intermediate and final outputs. CAUTION: because lance data is temporary,"
        " this action could be destructive in a permenant way. E.g,. if input data for a"
        " day that this CLI is being run for was previously generated with available"
        " lance data, but that data no longer exists, the resulting file may be empty or"
        " have significant data gaps. Use this primarily in a development environment."
    ),
)
def nrt_ecdr_for_day(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    ecdr_data_dir: Path,
    overwrite: bool,
):
    """Create an initial daily ECDR NetCDF using NRT LANCE AMSR2 data."""
    cde_filepath = get_ecdr_filepath(
        date=date,
        hemisphere=hemisphere,
        resolution=LANCE_RESOLUTION,
        ecdr_data_dir=ecdr_data_dir,
    )

    if cde_filepath.is_file() and not overwrite:
        logger.info(f"File for {date=} already exists ({cde_filepath}).")
        return

    if not cde_filepath.is_file() or overwrite:
        try:
            tiecdr_ds = read_or_create_and_read_nrt_tiecdr_ds(
                hemisphere=hemisphere,
                date=date,
                ecdr_data_dir=ecdr_data_dir,
                overwrite=overwrite,
            )

            cde_ds = complete_daily_ecdr_ds(
                tie_ds=tiecdr_ds,
                date=date,
                hemisphere=hemisphere,
                resolution=LANCE_RESOLUTION,
                ecdr_data_dir=ecdr_data_dir,
            )

            written_cde_ncfile = write_cde_netcdf(
                cde_ds=cde_ds,
                output_filepath=cde_filepath,
                ecdr_data_dir=ecdr_data_dir,
            )
            logger.info(f"Wrote complete daily ncfile: {written_cde_ncfile}")
        except Exception as e:
            logger.exception(f"Failed to create NRT ECDR for {date=} {hemisphere=}")
            create_err_logfile(
                filename=cde_filepath.name,
                ecdr_data_dir=ecdr_data_dir,
                product_type="complete_daily",
            )
            raise e


@click.group(name="nrt")
def nrt_cli():
    """Run NRT Sea Ice ECDR."""
    ...


nrt_cli.add_command(download_latest_nrt_data)
nrt_cli.add_command(nrt_ecdr_for_day)
