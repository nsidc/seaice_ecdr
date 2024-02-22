"""Code related to interacting with NetCDF files.
"""

import subprocess
from pathlib import Path


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
