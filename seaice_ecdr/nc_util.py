"""Code related to interacting with NetCDF files.
"""

import subprocess
from pathlib import Path

import datatree
import xarray as xr


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

    result = subprocess.run(
        [
            "ncrcat",
            *input_filepaths,
            output_filepath,
        ],
        capture_output=True,
    )

    result.check_returncode()


def remove_valid_range_from_coordinate_vars(ds: xr.Dataset | datatree.DataTree):
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


def add_coordinate_coverage_content_type(ds: xr.Dataset | datatree.DataTree):
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


def add_coordinates_attr(ds: datatree.DataTree):
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
            var = group[var_name]  # type:ignore[index]
            if var.dims == ("time", "y", "x"):
                var.attrs["coordinates"] = "time y x"
