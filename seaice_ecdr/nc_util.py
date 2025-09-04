"""Code related to interacting with NetCDF files.

Note/TODO: some of these functions (e.g., those prefaced with `fix_`) are hacks
to accomplish overriding attributes that are originally set in other parts of
the code. These functions are used to ensure our final outputs have the
attributes we want without needing to track down the source of where all of
these attributes get set in intermediate files. Eventually we should set
attributes as desired at creation time instead of "fixing" them at the end of
processing.
"""

import datetime as dt
import shutil
import subprocess
from pathlib import Path

import numpy as np
import xarray as xr
from dateutil.parser import parse
from loguru import logger


def concatenate_nc_files(
    *,
    input_filepaths: list[Path],
    output_filepath: Path,
) -> None:
    """Concatenate the given list of NetCDF files using `ncrcat`.

    `ncrcat` is considerably faster at concatenating NetCDF files along the
    `time` dimension than using `xr.open_mfdataset` followed by
    `ds.to_netcdf()`. E.g., in simple tests of the two methods for a year's
    worth of data - xarray takes ~20 mintues vs `ncrcat`'s ~20 seconds.
    """
    if not input_filepaths:
        raise RuntimeError("No input files given to concatenate.")

    logger.success(f"ncrcat-ing {len(input_filepaths)} files...")
    result = subprocess.run(
        [
            "ncrcat",
            *input_filepaths,
            output_filepath,
        ],
        capture_output=True,
    )

    result.check_returncode()


def remove_valid_range_from_coordinate_vars(ds: xr.Dataset | xr.DataTree):
    """removes `valid_range` attr from coordinate variables in-place.

    TODO: this function should be unnecessary. We use this to cleanup a ds
    before writing out as a netcdf file because e.g., ancillary files that we
    get x/y from contain `valid_range` attrs, but we don't want those in our
    outputs. Ideally, the ancillary files and the code that creates the `time`
    dim should omit `valid_range`, and this function becomes unnecessary.
    """
    ds.x.attrs = {k: v for k, v in ds.x.attrs.items() if k != "valid_range"}
    ds.y.attrs = {k: v for k, v in ds.y.attrs.items() if k != "valid_range"}
    ds.time.attrs = {k: v for k, v in ds.time.attrs.items() if k != "valid_range"}


def remove_FillValue_from_coordinate_vars(ds: xr.Dataset | xr.DataTree):
    """removes `valid_range` attr from coordinate variables in-place.

    TODO: this function should be unnecessary. We use this to cleanup a ds
    before writing out as a netcdf file because e.g., ancillary files that we
    get x/y from contain `valid_range` attrs, but we don't want those in our
    outputs. Ideally, the ancillary files and the code that creates the `time`
    dim should omit `valid_range`, and this function becomes unnecessary.
    """
    for coord_var in ("x", "y", "time"):
        try:
            ds.variables["coord_var"].encoding["_FillValue"] = None
        except KeyError:
            pass

    return ds


def add_coordinate_coverage_content_type(ds: xr.Dataset | xr.DataTree):
    """Adds `coverage_content_type` to the provided dataset in-place.

    TODO: This function should be unnecessary. We use this to cleanup a ds
    before writing out as a netcdf file because e.g., ancillary files that we
    get x/y from do not contain the `coverage_content_type`, but we don't want
    those in our outputs. Ideally, the ancillary files and the code that creates
    the `time` dim should include the `coverage_content_type` to coordinate
    vars, and this function becomes unnecessary.
    """
    ds.x.attrs["coverage_content_type"] = "coordinate"
    ds.y.attrs["coverage_content_type"] = "coordinate"
    ds.time.attrs["coverage_content_type"] = "coordinate"


def add_coordinates_attr(ds: xr.DataTree):
    """Adds `coordinates` attr to data variables in-place.

    TODO: This function should be unnecessary. We use this to cleanup a ds
    before writing out as a netcdf file because our code does not set
    `coordinates` attr at variable creation time. Ideally, code is updated to
    include the `coordinates` attr to data vars at creation time.
    """
    for group_name in ds.groups:
        group = ds[group_name]
        for var_name in group.variables:
            if var_name in ("crs", "time", "y", "x"):
                continue
            var = group[var_name]  # type: ignore[index]
            if var.dims == ("time", "y", "x"):
                var.attrs["coordinates"] = "time y x"


def get_empty_group(ds, ncgroup_name):
    """Return a version of this group with all fill values"""
    ds_empty_group = ds[ncgroup_name].copy()
    for datavar in ds_empty_group.data_vars:
        data_variable = ds_empty_group.data_vars[datavar]
        # NOTE: Getting the _FillValue via
        #   fillvalue = ds_empty_group.data_vars[datavar].encoding['_FillValue']
        # did not work for packed data.
        # Setting the fillvalue explicitly based on data_variable attrs
        if "number_of_missing_pixels" in data_variable.attrs:
            data_variable.attrs["number_of_missing_pixels"] = data_variable.size
        dtype = data_variable.dtype
        if dtype == "float32" or dtype == "float64":
            fillvalue = np.nan
        elif "flag_masks" in data_variable.attrs:
            fillvalue = 0
        elif dtype == "uint8":
            fillvalue = 255
            logger.warning(f"Setting fillvalue of {ncgroup_name} to {fillvalue}")
        else:
            fillvalue = 0
            logger.warning(f"Setting fillvalue of {ncgroup_name} to {fillvalue}")
        data_variable[:] = fillvalue

    return ds_empty_group


def add_ncgroup(
    tmpdir: Path,
    filepath_list: list[Path],
    ncgroup_name: str,
) -> list[Path]:
    """If the specified group name is in any of the listed filepaths,
    Add an "empty" version to all of them.

    In practice, this is used to create empty `prototype_am2` groups when we are
    aggregating files that do (during the AMSR2 period) and do not contain
    (pre-AMSR2) the `prototype_am2` group. This occurs when e.g., aggregating
    daily data over a period that intersects the start date for AMSR2.
    """
    add_groupname = False
    for nc_filename in reversed(filepath_list):
        ds = xr.open_datatree(nc_filename)
        if ncgroup_name in ds.groups:
            add_groupname = True
            empty_nc_datatree = get_empty_group(
                ds,
                ncgroup_name,
            )
            break

    # If we don't need to add this nc_group, return the filename list unaltered
    if not add_groupname:
        return filepath_list

    # Want to copy the files to the tempdir, and add the prototype field
    root_datatree = xr.DataTree()
    empty_nc_datatree.parent = root_datatree
    clean_ncgroup_name = ncgroup_name.replace("/", "")
    empty_datatree_fn = f"empty_datatree_{clean_ncgroup_name}.nc"
    empty_datatree_fp = Path(tmpdir, empty_datatree_fn)
    empty_nc_datatree = remove_FillValue_from_coordinate_vars(empty_nc_datatree)

    empty_nc_datatree.to_netcdf(empty_datatree_fp)

    new_file_list = []
    updated_file_count = 0
    logger.info(
        f"Adding {clean_ncgroup_name} nc group to {len(filepath_list)} files as needed..."
    )
    for fp in filepath_list:
        new_fp = Path(tmpdir, fp.name)
        shutil.copyfile(fp, new_fp)
        new_file_list.append(new_fp)
        ds_check = xr.open_datatree(new_fp)
        if ncgroup_name not in ds_check.groups:
            logger.info(f"adding {clean_ncgroup_name} group to {new_fp}...")
            result = subprocess.run(
                [
                    "ncks",
                    "-A",
                    "-h",
                    "-g",
                    clean_ncgroup_name,
                    empty_datatree_fp,
                    new_fp,
                ],
                capture_output=True,
            )
            result.check_returncode()
            updated_file_count += 1
        else:
            logger.info(f"{new_fp} already has nc group {clean_ncgroup_name}")

    logger.success(f"Added {clean_ncgroup_name} to {updated_file_count} files.")

    return new_file_list


def adjust_monthly_aggregate_ncattrs(fp):
    """Clean up source, platform, and sensor ncattrs"""
    print("SKIPPING adjust_monthly_aggregate_ncattrs")
    print("  pending final decisions on monagg_attrs")
    pass


def fix_monthly_ncattrs(
    ds: xr.Dataset | xr.DataTree,
) -> xr.Dataset | xr.DataTree:
    """Hack to fix the attributes of the monthly CDR files

    Attrs get set in other parts of the code, but we made a decision to change
    these in the final published output. This should be considered a temporary
    hack until we can track down and make updates to the code where these
    attributes are originally set.
    """
    # TODO: Have these fixes pull information from proper config files
    start_of_month_str = ds.attrs["time_coverage_start"]
    start_of_month = parse(start_of_month_str).date()

    # Fixes for "normal product"
    platform_string = None
    sensor_string = None
    if start_of_month < dt.date(1987, 7, 1):
        source_string = "Generated from https://nsidc.org/data/nsidc-0007"
    elif start_of_month == dt.date(1987, 7, 1):
        source_string = "Generated from https://nsidc.org/data/nsidc-0001 and https://nsidc.org/data/nsidc-0007"
        platform_string = (
            "Nimbus-7; DMSP 5D-2/F8 > Defense Meteorological Satellite Program-F8"
        )
        sensor_string = "SMMR > Scanning Multichannel Microwave Radiometer; SSM/I > Special Sensor Microwave/Imager"
    else:
        source_string = "Generated from https://nsidc.org/data/nsidc-0001"

    ds.attrs["source"] = source_string
    if platform_string is not None:
        ds.attrs["platform"] = platform_string
    if sensor_string is not None:
        ds.attrs["sensor"] = sensor_string

    # AMSR2 (prototype) fix
    try:
        if "/prototype_am2" in ds.groups:
            ds.prototype_am2.attrs["source"] = (
                "Generated from https://nsidc.org/data/au_si25"
            )
    except ValueError:
        # Ignore error if ds is xr.Dataset (with no groups)
        pass

    return ds


def get_unique_comma_separated_items(item_string):
    """Return an ordered list of the unique items in a comma-separated list
    of items"""
    all_items_list = item_string.split(",")
    unique_items_list = list(dict.fromkeys(all_items_list))
    no_space_items = [p if p[0] != " " else p[1:] for p in unique_items_list]
    unique_items = list(dict.fromkeys(no_space_items))
    unique_items_string = ", ".join(unique_items)

    return unique_items_string


def fix_monthly_aggregate_ncattrs(
    ds: xr.Dataset | xr.DataTree,
) -> xr.Dataset | xr.DataTree:
    """Hack to fix the attributes of the monthly-aggregated (all months) CDR files

    Attrs get set in other parts of the code, but we made a decision to change
    these in the final published output. This should be considered a temporary
    hack until we can track down and make updates to the code where these
    attributes are originally set.
    """
    # NOTE: No need to fix prototype_am2 attrs

    # Fixes for "normal" monthly-aggregate product -- i.e. non-prototype
    source_string = "Generated from https://nsidc.org/data/nsidc-0001, https://nsidc.org/data/nsidc-0007"
    ds.attrs["source"] = source_string

    platform_string = get_unique_comma_separated_items(ds.attrs["platform"])
    ds.attrs["platform"] = platform_string

    sensor_string = get_unique_comma_separated_items(ds.attrs["sensor"])
    ds.attrs["sensor"] = sensor_string

    return ds


def fix_daily_aggregate_ncattrs(
    ds: xr.Dataset | xr.DataTree,
) -> xr.Dataset | xr.DataTree:
    """Hack to fix the attributes of the daily-aggregated (annual) CDR files

    Attrs get set in other parts of the code, but we made a decision to change
    these in the final published output. This should be considered a temporary
    hack until we can track down and make updates to the code where these
    attributes are originally set.
    """
    # Note: No fixes are needed for the prototype_am2 group
    # TODO: Have these fixes pull information from proper config files
    start_of_year_str = ds.attrs["time_coverage_start"]
    start_of_year = parse(start_of_year_str).date()

    # Fixes for "normal product"
    if start_of_year < dt.date(1987, 1, 1):
        source_string = "Generated from https://nsidc.org/data/nsidc-0007"
    elif start_of_year == dt.date(1987, 1, 1):
        source_string = "Generated from https://nsidc.org/data/nsidc-0001 and https://nsidc.org/data/nsidc-0007"
    else:
        source_string = "Generated from https://nsidc.org/data/nsidc-0001"
    ds.attrs["source"] = source_string

    platform_string = get_unique_comma_separated_items(ds.attrs["platform"])
    ds.attrs["platform"] = platform_string

    sensor_string = get_unique_comma_separated_items(ds.attrs["sensor"])
    ds.attrs["sensor"] = sensor_string

    return ds
