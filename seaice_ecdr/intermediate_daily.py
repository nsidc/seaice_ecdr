"""Routines for generating completely filled daily eCDR files.

This code takes temporally interpolated daily data (`temporal_composite_daily.py`) and adds:

* melt onset (for NH)
* The surface-type field
* Sets necessary NC attrs
"""

import copy
import datetime as dt
from functools import cache
from pathlib import Path
from typing import Iterable, get_args

import click
import numpy as np
import numpy.typing as npt
import xarray as xr
from loguru import logger
from pm_tb_data._types import NORTH, Hemisphere

from seaice_ecdr._types import ECDR_SUPPORTED_RESOLUTIONS
from seaice_ecdr.ancillary import (
    ANCILLARY_SOURCES,
    get_non_ocean_mask,
    get_surfacetype_da,
)
from seaice_ecdr.cli.util import datetime_to_date
from seaice_ecdr.constants import DEFAULT_BASE_OUTPUT_DIR
from seaice_ecdr.melt import (
    MELT_ONSET_FILL_VALUE,
    MELT_SEASON_FIRST_DOY,
    MELT_SEASON_LAST_DOY,
    date_in_nh_melt_season,
    melting,
)
from seaice_ecdr.platforms import PLATFORM_CONFIG, SUPPORTED_PLATFORM_ID
from seaice_ecdr.set_daily_ncattrs import finalize_cdecdr_ds
from seaice_ecdr.spillover import LAND_SPILL_ALGS
from seaice_ecdr.temporal_composite_daily import (
    read_or_create_and_read_standard_tiecdr_ds,
    read_tiecdr_ds,
)
from seaice_ecdr.util import (
    date_range,
    get_ecdr_grid_shape,
    get_intermediate_output_dir,
    nrt_daily_filename,
    raise_error_for_dates,
    standard_daily_filename,
)


@cache
def get_ecdr_dir(
    *,
    intermediate_output_dir: Path,
    year: int,
    # TODO: extract nrt handling and make responsiblity for defining the output
    # dir a higher-level concern.
    is_nrt: bool,
) -> Path:
    """Daily output dir for ECDR processing"""
    if is_nrt:
        # NRT daily data just lives under the complete output dir.
        ecdr_dir = intermediate_output_dir
    else:
        ecdr_dir = intermediate_output_dir / "daily" / str(year)
    ecdr_dir.mkdir(parents=True, exist_ok=True)

    return ecdr_dir


def get_ecdr_filepath(
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    intermediate_output_dir: Path,
    platform_id: SUPPORTED_PLATFORM_ID,
    is_nrt: bool,
) -> Path:
    """Return the daily eCDR file path."""
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

    ecdr_dir = get_ecdr_dir(
        intermediate_output_dir=intermediate_output_dir,
        year=date.year,
        is_nrt=is_nrt,
    )

    ecdr_filepath = ecdr_dir / ecdr_filename

    return ecdr_filepath


def _empty_melt_onset_field(
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
) -> npt.NDArray:
    """Return an array of the shape for this hem/res filled with fill_value."""
    # `grid_shape` is 2D.
    grid_shape = get_ecdr_grid_shape(
        hemisphere=hemisphere,
        resolution=resolution,
    )
    # Add a dim for `time`, making the melt onset field 3D.
    melt_onset_field_shape = (1, *grid_shape)
    empty_melt_onset_field = np.full(
        melt_onset_field_shape,
        MELT_ONSET_FILL_VALUE,
        dtype=np.uint8,
    )

    return empty_melt_onset_field


def read_melt_onset_field_from_complete_daily(
    *,
    date,
    hemisphere,
    resolution,
    intermediate_output_dir: Path,
    is_nrt: bool,
) -> npt.NDArray:
    platform = PLATFORM_CONFIG.get_platform_by_date(date)
    cde_ds = read_cdecdr_ds(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        intermediate_output_dir=intermediate_output_dir,
        is_nrt=is_nrt,
        platform_id=platform.id,
    )

    # TODO: Perhaps these field names should be in a dictionary somewhere?
    melt_onset_from_ds = cde_ds["cdr_melt_onset_day"].to_numpy()

    return melt_onset_from_ds


def read_melt_elements_from_tiecdr(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    intermediate_output_dir: Path,
):
    tie_ds = read_tiecdr_ds(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        intermediate_output_dir=intermediate_output_dir,
    )

    return (
        np.squeeze(tie_ds["cdr_conc"].to_numpy()),
        tie_ds["h19_day_si"].to_numpy(),
        tie_ds["h37_day_si"].to_numpy(),
    )


def update_melt_onset_for_day(
    *,
    prior_melt_onset_field: npt.NDArray,
    date: dt.date,
    cdr_conc: npt.NDArray,
    tb_h19: npt.NDArray,
    tb_h37: npt.NDArray,
    non_ocean_mask: npt.NDArray[np.bool_],
) -> npt.NDArray[np.uint8]:
    """Return an updated melt onset field given data for the given day."""
    is_melted_today = melting(
        concentrations=cdr_conc,
        tb_h19=tb_h19,
        tb_h37=tb_h37,
    )
    # Apply non-ocean mask
    is_melted_today[0, non_ocean_mask] = False

    # Note: zero counts as a prior value
    have_prior_melt_values = prior_melt_onset_field != MELT_ONSET_FILL_VALUE

    # Note: zero is also a "missing prior" day that can be updated
    is_missing_prior = (prior_melt_onset_field == MELT_ONSET_FILL_VALUE) | (
        prior_melt_onset_field == 0
    )
    has_new_melt = is_missing_prior & is_melted_today

    melt_onset_field = np.zeros(prior_melt_onset_field.shape, dtype=np.uint8)
    melt_onset_field[:] = MELT_ONSET_FILL_VALUE
    melt_onset_field[have_prior_melt_values] = prior_melt_onset_field[
        have_prior_melt_values
    ]

    # Note: this can replace zeros with a value
    day_of_year = int(date.strftime("%j"))
    melt_onset_field[has_new_melt] = day_of_year

    return melt_onset_field


def create_melt_onset_field(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ancillary_source: ANCILLARY_SOURCES,
    intermediate_output_dir: Path,
    is_nrt: bool,
) -> npt.NDArray[np.uint8]:
    """Return a uint8 melt onset field.

    It is expected that melt only be calculated in the northern hemisphere. A
    `RuntimeError` is raised if `south` is passed for `hemisphere`.

    Note: this routine creates the melt onset field using input data
    from today (and yesterday if needed).

    For dates with day of year prior to first_melt_doy or after last_melt_doy,
    this will be an uint8 array with all values set to no_melt_flag.

    For date with day of year equal to first_melt_doy, this field will be
    computed.

    For date between first_melt_doy and last_melt_doy, this will read the
    melt onset field from the prior day's cdecdr file and compute melt
    for the current day.
    """
    if hemisphere != NORTH:
        raise RuntimeError(
            "The CDR melt algorithm is not designed for use in the Southern hemisphere."
        )

    day_of_year = int(date.strftime("%j"))
    # Determine if the given day of year is within the melt season. If it's not,
    # return an empty melt onset field.
    if not date_in_nh_melt_season(date=date):
        logger.debug(
            f"returning empty melt_onset_field for {date:%Y-%m-%d} ({day_of_year=})"
        )
        return _empty_melt_onset_field(
            hemisphere=hemisphere,
            resolution=resolution,
        )

    is_first_day_of_melt = day_of_year == MELT_SEASON_FIRST_DOY
    if is_first_day_of_melt:
        # The first day of the melt seasion should have a prior_melt field
        # with fill value where there is siconc and zeros in non-siconc ocean
        # The fill values are set here.  The zeros are set later in this routine
        prior_melt_onset_field = _empty_melt_onset_field(
            hemisphere=hemisphere,
            resolution=resolution,
        )
    else:
        # During the melt season, try to read the previous day's input as a
        # starting point. Use an empty melt onset field if no data for the
        # previous day is available.
        try:
            prior_melt_onset_field = read_melt_onset_field_from_complete_daily(
                date=date - dt.timedelta(days=1),
                hemisphere=hemisphere,
                resolution=resolution,
                intermediate_output_dir=intermediate_output_dir,
                is_nrt=is_nrt,
            )
            logger.debug(f"using read melt_onset_field for prior for {day_of_year}")
        except FileNotFoundError:
            logger.warning(
                f"Tried to read previous melt field for {day_of_year} but the file was not found."
            )
            prior_melt_onset_field = _empty_melt_onset_field(
                hemisphere=hemisphere,
                resolution=resolution,
            )
            logger.debug(f"using empty melt_onset_field for prior for {day_of_year}")

    cdr_conc_ti, tb_h19, tb_h37 = read_melt_elements_from_tiecdr(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        intermediate_output_dir=intermediate_output_dir,
    )

    non_ocean_mask = get_non_ocean_mask(
        hemisphere=hemisphere,
        resolution=resolution,
        ancillary_source=ancillary_source,
    )

    updated_melt_onset = update_melt_onset_for_day(
        prior_melt_onset_field=prior_melt_onset_field,
        cdr_conc=cdr_conc_ti,
        tb_h19=tb_h19,
        tb_h37=tb_h37,
        non_ocean_mask=non_ocean_mask.data,
        date=date,
    )

    # On first day of melt, set to 0 if siconc < 50% and didn't melt today
    # Note: values of 0 can be replaced later if they melt in the melt season
    if is_first_day_of_melt:
        is_nosiconc_ti = (
            np.isfinite(cdr_conc_ti)
            & (cdr_conc_ti < 0.5)
            & (updated_melt_onset[0, :, :] == MELT_ONSET_FILL_VALUE)
        )
        updated_melt_onset[0, is_nosiconc_ti] = 0
        logger.debug(
            f"using zeros and fill values in ocean for first melt_onset field on day {day_of_year}"
        )

    return updated_melt_onset


def _add_melt_onset_for_nh(
    *,
    cde_ds: xr.Dataset,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ancillary_source: ANCILLARY_SOURCES,
    intermediate_output_dir: Path,
    is_nrt: bool,
) -> xr.Dataset:
    """Add the melt onset field to the complete daily dataset for the given date."""
    cde_ds_with_melt_onset = cde_ds.copy()

    melt_onset_field = create_melt_onset_field(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        intermediate_output_dir=intermediate_output_dir,
        is_nrt=is_nrt,
        ancillary_source=ancillary_source,
    )

    # Update cde_ds with melt onset info
    cde_ds_with_melt_onset["cdr_melt_onset_day"] = (
        ("time", "y", "x"),
        melt_onset_field,
        {
            "grid_mapping": "crs",
            "standard_name": "status_flag",
            "comment": (
                "Value of 255 means no melt detected yet or the date is"
                " outside the melt season.  Other values indicate the day"
                " of year when melt was first detected at this location."
            ),
        },
        {
            "zlib": True,
        },
    )

    # the np.squeeze() function here is to remove the time dim so that
    # this becomes a 2d array for updating the qa_... field
    is_melt_has_occurred = np.squeeze(
        (MELT_SEASON_FIRST_DOY <= cde_ds_with_melt_onset["cdr_melt_onset_day"].data)
        & (cde_ds_with_melt_onset["cdr_melt_onset_day"].data <= MELT_SEASON_LAST_DOY)
    )
    # TODO: the flag value being "or"ed to the bitmask should be looked
    #       up as the temporally-interpolation-has-occured value
    #       rather than hardcoded as '128'.
    cde_ds_with_melt_onset["cdr_seaice_conc_qa_flag"] = cde_ds_with_melt_onset[
        "cdr_seaice_conc_qa_flag"
    ].where(
        ~is_melt_has_occurred,
        other=np.bitwise_or(cde_ds_with_melt_onset["cdr_seaice_conc_qa_flag"], 128),
    )

    return cde_ds_with_melt_onset


def _add_surfacetype_da(
    *,
    date: dt.date,
    cde_ds: xr.Dataset,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ancillary_source: ANCILLARY_SOURCES,
) -> xr.Dataset:
    """Add the surface_type field to the complete daily dataset for the given date."""
    cde_ds_with_surfacetype = cde_ds.copy()
    # Add the surface-type field
    # TODO: Setting a DataArray directly into a Dataset changes the
    #       coordinate variables of the Dataset.  Eg, here, if the
    #       surfacetype dataarray is set directly, the x, y, and time
    #       variables/coords/dims of the cde_ds are reset.
    #       The methodology here should be reviewed to see if there is
    #       a "better" way to add a geo-referenced dataarray to an existing
    #       xr Dataset.
    surfacetype_da = get_surfacetype_da(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        ancillary_source=ancillary_source,
    )
    # Force use of the cde_ds coords instead of the x, y, time vars
    # from the ancillary file (which *should* be compatible...but we
    # don't want coords changing in cde_ds as a result of external files).
    surfacetype_da.assign_coords(
        {
            "time": cde_ds_with_surfacetype.time,
            "y": cde_ds_with_surfacetype.y,
            "x": cde_ds_with_surfacetype.x,
        }
    )
    cde_ds_with_surfacetype = cde_ds_with_surfacetype.merge(surfacetype_da)

    return cde_ds_with_surfacetype


def complete_daily_ecdr_ds(
    *,
    tie_ds: xr.Dataset,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ancillary_source: ANCILLARY_SOURCES,
    intermediate_output_dir: Path,
    is_nrt: bool,
) -> xr.Dataset:
    """Create xr dataset containing the complete daily enhanced CDR.

    This function returns
    - a Dataset containing
      - The melt onset field
      - All appropriate QA fields
      - sets variable encoding for writting data to NetCDF.
    """
    # Initialize the complete daily ECDR dataset (cde) using the temporally
    # interpolated ECDR (tie) dataset provided to this function.
    cde_ds = tie_ds.copy()

    # Add the surface-type field
    cde_ds = _add_surfacetype_da(
        date=date,
        cde_ds=cde_ds,
        hemisphere=hemisphere,
        resolution=resolution,
        ancillary_source=ancillary_source,
    )

    # For the northern hemisphere, create the melt onset field and add it to the
    # dataset. The southern hemisphere does not include a melt onset field.
    if hemisphere == NORTH:
        cde_ds = _add_melt_onset_for_nh(
            cde_ds=cde_ds,
            date=date,
            hemisphere=hemisphere,
            resolution=resolution,
            intermediate_output_dir=intermediate_output_dir,
            is_nrt=is_nrt,
            ancillary_source=ancillary_source,
        )

    cde_ds = finalize_cdecdr_ds(
        cde_ds, hemisphere, resolution, ancillary_source=ancillary_source
    )

    # Set nc encoding for variables
    conc_fields: Iterable[str] = [
        "raw_bt_seaice_conc",
        "raw_nt_seaice_conc",
        "cdr_seaice_conc",
    ]
    for conc_varname in conc_fields:
        cde_ds[conc_varname].encoding = {
            "zlib": True,
            "dtype": "uint8",
            "scale_factor": 0.01,
            "add_offset": 0.0,
            "_FillValue": 255,
        }
    for coord_var_name in ("crs", "time", "y", "x"):
        cde_ds[coord_var_name].encoding = {"zlib": True}

    cde_ds.encoding = {"unlimited_dims": "time"}

    # TODO: Need to ensure that the cdr_seaice_conc field does not have values
    #       where seaice cannot occur, eg over land or lakes

    return cde_ds


def make_standard_cdecdr_netcdf(
    date: dt.date,
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    base_output_dir: Path,
    land_spillover_alg: LAND_SPILL_ALGS,
    ancillary_source: ANCILLARY_SOURCES,
    overwrite_cde: bool = False,
) -> Path:
    """Create a 'standard', daily CDR NetCDF file.

    'standard' files are those that use the non-NRT input sources.

    This function is recursive. It will attempt to create earlier complete daily
    files if they do not exist.

    The output of this function is a "complete" daily file for a given platform
    start-dates configuration.
    """
    intermediate_output_dir = get_intermediate_output_dir(
        base_output_dir=base_output_dir,
        hemisphere=hemisphere,
    )
    platform = PLATFORM_CONFIG.get_platform_by_date(date)
    cde_filepath = get_ecdr_filepath(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
        intermediate_output_dir=intermediate_output_dir,
        platform_id=platform.id,
        is_nrt=False,
    )

    if cde_filepath.is_file() and not overwrite_cde:
        logger.info(
            f"Complete daily ECDR already exists for {date=} {hemisphere=} {resolution=}: {cde_filepath}"
        )
        return cde_filepath

    try:
        logger.info(f"Creating cdecdr for {date=}, {hemisphere=}, {resolution=}")

        intermediate_output_dir = get_intermediate_output_dir(
            base_output_dir=base_output_dir,
            hemisphere=hemisphere,
        )
        tie_ds = read_or_create_and_read_standard_tiecdr_ds(
            date=date,
            hemisphere=hemisphere,
            resolution=resolution,
            intermediate_output_dir=intermediate_output_dir,
            land_spillover_alg=land_spillover_alg,
            ancillary_source=ancillary_source,
        )

        # Ensure the previous day's complete daily field exists for the melt
        # onset calculation/update (NH only!).
        # This is a recursive function! Ideally we'd just have code that
        # generates the necessary intermediate files for the target date, and
        # then this code would be solely responsible for reading the previous
        # day's complete field.
        if hemisphere == NORTH and date_in_nh_melt_season(
            date=date - dt.timedelta(days=1)
        ):
            make_standard_cdecdr_netcdf(
                date=date - dt.timedelta(days=1),
                hemisphere=hemisphere,
                resolution=resolution,
                base_output_dir=base_output_dir,
                overwrite_cde=overwrite_cde,
                land_spillover_alg=land_spillover_alg,
                ancillary_source=ancillary_source,
            )

        cde_ds = complete_daily_ecdr_ds(
            tie_ds=tie_ds,
            date=date,
            hemisphere=hemisphere,
            resolution=resolution,
            intermediate_output_dir=intermediate_output_dir,
            is_nrt=False,
            ancillary_source=ancillary_source,
        )

        cde_ds.to_netcdf(
            cde_filepath,
        )
        logger.success(f"Wrote complete daily ncfile: {cde_filepath}")
    except Exception as e:
        logger.exception(
            "Failed to create complete daily NetCDF for"
            f" {hemisphere=}, {date=}, {resolution=}."
        )
        raise e

    return cde_filepath


def read_cdecdr_ds(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    intermediate_output_dir: Path,
    platform_id: SUPPORTED_PLATFORM_ID,
    is_nrt: bool,
) -> xr.Dataset:
    cde_filepath = get_ecdr_filepath(
        date,
        hemisphere,
        resolution,
        intermediate_output_dir=intermediate_output_dir,
        platform_id=platform_id,
        is_nrt=is_nrt,
    )
    logger.debug(f"Reading cdeCDR file from: {cde_filepath}")
    cde_ds = xr.load_dataset(cde_filepath)

    return cde_ds


def create_standard_ecdr_for_dates(
    dates: Iterable[dt.date],
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    base_output_dir: Path,
    land_spillover_alg: LAND_SPILL_ALGS,
    ancillary_source: ANCILLARY_SOURCES,
    overwrite_cde: bool = False,
) -> list[dt.date]:
    """Create "standard" (non-NRT) daily ECDR NC files for the provided dates.

    This function will try to create a daily NC file for each day. If any errors
    are encountered, the dates for which errors occurred are returned as a
    list. It's the responsiblity of calling-code to ensure that those dates are
    handled (e.g., raise an error alerting the user to the issue).
    """
    error_dates = []
    for date in dates:
        try:
            make_standard_cdecdr_netcdf(
                date=date,
                hemisphere=hemisphere,
                resolution=resolution,
                base_output_dir=base_output_dir,
                overwrite_cde=overwrite_cde,
                land_spillover_alg=land_spillover_alg,
                ancillary_source=ancillary_source,
            )
        except Exception:
            logger.exception(f"Failed to create standard ECDR for {date=}")
            error_dates.append(date)

    return error_dates


@click.command(name="intermediate-daily")
@click.option(
    "-d",
    "--date",
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
@click.option(
    "--land-spillover-alg",
    required=True,
    type=click.Choice(get_args(LAND_SPILL_ALGS)),
)
@click.option(
    "--ancillary-source",
    required=True,
    type=click.Choice(get_args(ANCILLARY_SOURCES)),
)
@click.option(
    "--overwrite",
    is_flag=True,
)
def cli(
    *,
    date: dt.date,
    end_date: dt.date | None,
    hemisphere: Hemisphere,
    base_output_dir: Path,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    land_spillover_alg: LAND_SPILL_ALGS,
    ancillary_source: ANCILLARY_SOURCES,
    overwrite: bool,
) -> None:
    """Run the temporal composite daily ECDR algorithm with AMSR2 data.

    This requires the creation/existence of temporally interpolated eCDR
    (tiecdr) files.

    TODO: eventually we want to be able to specify: date, grid (grid includes
    projection, resolution, and bounds), and TBtype (TB type includes source and
    methodology for getting those TBs onto the grid)
    """
    # raise ValueError('made it to here!')
    if end_date is None:
        end_date = copy.copy(date)

    error_dates = create_standard_ecdr_for_dates(
        dates=date_range(start_date=date, end_date=end_date),
        hemisphere=hemisphere,
        resolution=resolution,
        base_output_dir=base_output_dir,
        overwrite_cde=overwrite,
        land_spillover_alg=land_spillover_alg,
        ancillary_source=ancillary_source,
    )
    raise_error_for_dates(error_dates=error_dates)
