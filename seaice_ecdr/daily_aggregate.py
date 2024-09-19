"""Code to produce daily aggregate files from daily complete data.


TODO: The daily-aggregate processing is very parallelizable because
      each year is indendent of every other year.  It could be
      implemented with multi-processing to speed up production
      on a multi-core machine.  Perhaps as a cmdline arg to this
      CLI API?
"""

import datetime as dt
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import get_args

import click
import datatree
import pandas as pd
from loguru import logger
from pm_tb_data._types import Hemisphere

from seaice_ecdr._types import ECDR_SUPPORTED_RESOLUTIONS
from seaice_ecdr.ancillary import ANCILLARY_SOURCES, get_ancillary_ds
from seaice_ecdr.checksum import write_checksum_file
from seaice_ecdr.constants import DEFAULT_BASE_OUTPUT_DIR
from seaice_ecdr.nc_attrs import get_global_attrs
from seaice_ecdr.nc_util import concatenate_nc_files
from seaice_ecdr.platforms import PLATFORM_CONFIG
from seaice_ecdr.publish_daily import get_complete_daily_filepath
from seaice_ecdr.util import (
    get_complete_output_dir,
    platform_id_from_filename,
    standard_daily_aggregate_filename,
)


# TODO: very similar to `monthly._get_daily_complete_filepaths_for_month`. DRY
# out/move into module/subpackage related to daily data.
def _get_daily_complete_filepaths_for_year(
    *,
    year: int,
    complete_output_dir: Path,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
) -> list[Path]:
    data_list = []
    start_date = max(
        dt.date(year, 1, 1), PLATFORM_CONFIG.get_first_platform_start_date()
    )
    for period in pd.period_range(start=start_date, end=dt.date(year, 12, 31)):
        expected_fp = get_complete_daily_filepath(
            date=period.to_timestamp().date(),
            hemisphere=hemisphere,
            resolution=resolution,
            complete_output_dir=complete_output_dir,
            is_nrt=False,
        )
        if expected_fp.is_file():
            data_list.append(expected_fp)
        else:
            logger.warning(f"Expected to find {expected_fp} but found none.")

    if len(data_list) == 0:
        raise RuntimeError("No daily data files found.")

    return data_list


def get_daily_aggregate_filepath(
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    complete_output_dir: Path,
    start_date: dt.date,
    end_date: dt.date,
) -> Path:
    output_dir = complete_output_dir / "aggregate"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_fn = standard_daily_aggregate_filename(
        hemisphere=hemisphere,
        resolution=resolution,
        start_date=start_date,
        end_date=end_date,
    )

    output_filepath = output_dir / output_fn

    return output_filepath


def _update_ncrcat_daily_ds(
    *,
    ds: datatree.DataTree,
    daily_filepaths: list[Path],
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ancillary_source: ANCILLARY_SOURCES = "CDRv5",
):
    """Update the aggregate dataset created by `ncrcat`.

    Adds lat/lon fields and sets global attrs.
    """
    surf_geo_ds = get_ancillary_ds(
        hemisphere=hemisphere,
        resolution=resolution,
        ancillary_source=ancillary_source,
    )
    # Lat and lon fields are placed under the "cdr_supplementary" group.
    for sup_var in ("latitude", "longitude"):
        ds["cdr_supplementary"][sup_var] = surf_geo_ds[sup_var]
        # Add in the `coordinates` attr to lat and lon.
        ds["cdr_supplementary"][sup_var].attrs["coordinates"] = "y x"

    # lat and lon fields have x and y coordinate variables associated with them
    # and get added automatically when adding those fields above. This drops
    # those unnecessary vars that will be inherited from the root group.
    ds["cdr_supplementary"] = ds["cdr_supplementary"].drop_vars(["x", "y"])

    # Remove the "number of missing pixels" attr from the daily aggregate conc
    # variables.
    ds["cdr_seaice_conc"].attrs = {
        k: v
        for k, v in ds["cdr_seaice_conc"].attrs.items()
        if k != "number_of_missing_pixels"
    }

    # Remove "number_of_missing_pixels" attr from seaice conc var in any
    # prototype subgroups
    prototype_groups = [
        group_name for group_name in ds.groups if "/prototype_" in group_name
    ]
    if len(prototype_groups) > 0:
        for prototype_group in prototype_groups:
            # We expect prototype group name to have the form
            # `prototype_{platform_id}`
            platform_id = prototype_group.split("_")[1]
            prototype_group_name = f"prototype_{platform_id}"
            prototype_seaice_conc_name = f"{platform_id}_seaice_conc"
            ds[prototype_group_name][prototype_seaice_conc_name].attrs = {
                k: v
                for k, v in ds[prototype_group_name][
                    prototype_seaice_conc_name
                ].attrs.items()
                if k != "number_of_missing_pixels"
            }

    # Set global attributes. Only updates to the root group are necessary. The
    # `prototype_{platform_id}` and `cdr_supplementary` groups will keep its attrs unchanged.
    daily_aggregate_ds_global_attrs = get_global_attrs(
        time=ds.time,
        temporality="daily",
        aggregate=True,
        source=", ".join([fp.name for fp in daily_filepaths]),
        platform_ids=[platform_id_from_filename(fp.name) for fp in daily_filepaths],
        resolution=resolution,
    )
    ds.attrs = daily_aggregate_ds_global_attrs  # type: ignore[assignment]

    return ds


def make_daily_aggregate_netcdf_for_year(
    *,
    year: int,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    complete_output_dir: Path,
) -> None:
    try:
        daily_filepaths = _get_daily_complete_filepaths_for_year(
            year=year,
            complete_output_dir=complete_output_dir,
            hemisphere=hemisphere,
            resolution=resolution,
        )

        # Create a temporary dir to store a WIP netcdf file. We do this because
        # using the `ncrcat` CLI tool (included with `nco` dep) is much faster than
        # xarray at concatenating the data. Then we do some modifications (e.g.,
        # adding `latitude` and `longitude`, global attrs, etc.) before saving the
        # data in it's final location.
        with TemporaryDirectory() as tmpdir:
            tmp_output_fp = Path(tmpdir) / "temp.nc"
            concatenate_nc_files(
                input_filepaths=daily_filepaths,
                output_filepath=tmp_output_fp,
            )
            daily_ds = datatree.open_datatree(tmp_output_fp, chunks=dict(time=1))
            daily_ds = _update_ncrcat_daily_ds(
                ds=daily_ds,
                daily_filepaths=daily_filepaths,
                hemisphere=hemisphere,
                resolution=resolution,
            )

            output_path = get_daily_aggregate_filepath(
                hemisphere=hemisphere,
                resolution=resolution,
                start_date=pd.Timestamp(daily_ds.time.min().item()).date(),
                end_date=pd.Timestamp(daily_ds.time.max().item()).date(),
                complete_output_dir=complete_output_dir,
            )

            # The daily ds should already be set to handle `time` as an
            # unlimited dim.
            assert daily_ds.encoding["unlimited_dims"] == {"time"}

            # Write out the aggregate daily file.
            daily_ds.to_netcdf(
                output_path,
            )

        logger.success(f"Wrote daily aggregate file for year={year} to {output_path}")

        # Write checksum file for the aggregate daily output.
        checksum_output_dir = complete_output_dir / "checksums" / "aggregate"
        write_checksum_file(
            input_filepath=output_path,
            output_dir=checksum_output_dir,
        )
    except Exception as e:
        logger.exception(f"Failed to create daily aggregate for {year=} {hemisphere=}")
        raise e


@click.command(name="daily-aggregate")
@click.option(
    "--year",
    required=True,
    type=int,
    help="Year for which to create the daily-aggregate file.",
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
    default="25",
    type=click.Choice(get_args(ECDR_SUPPORTED_RESOLUTIONS)),
)
@click.option(
    "--end-year",
    required=False,
    type=int,
    help="Last year for which to create the daily-aggregate file.",
    default=None,
)
def cli(
    *,
    year: int,
    hemisphere: Hemisphere,
    base_output_dir: Path,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    end_year: int | None,
) -> None:
    if end_year is None:
        end_year = year

    complete_output_dir = get_complete_output_dir(
        base_output_dir=base_output_dir,
        hemisphere=hemisphere,
        is_nrt=False,
    )
    failed_years = []
    for year_to_process in range(year, end_year + 1):
        try:
            make_daily_aggregate_netcdf_for_year(
                year=year_to_process,
                hemisphere=hemisphere,
                resolution=resolution,
                complete_output_dir=complete_output_dir,
            )
        except Exception:
            failed_years.append(year_to_process)

    if failed_years:
        str_formatted_years = "\n".join(str(year) for year in failed_years)
        raise RuntimeError(
            f"Encountered {len(failed_years)} failures."
            f" Daily aggregates for the following years were not created:\n{str_formatted_years}"
        )
