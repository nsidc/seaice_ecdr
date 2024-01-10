"""One-off script to add x/y variable attrs to SH ancillary file

The x/y variables in the SH files was missing attributes as of Jan 10,
2024. This script fixes that issue. Should only need to be run once!
"""
import xarray as xr
from loguru import logger
from pm_tb_data._types import SOUTH

from seaice_ecdr.ancillary import get_ancillary_filepath


def fix_xy_attrs(*, ancillary_ds: xr.Dataset) -> xr.Dataset:
    ancillary_ds.x.attrs = dict(
        standard_name="projection_x_coordinate",
        long_name="x coordinate of projection",
        units="meters",
        axis="X",
        valid_range=(ancillary_ds.x.min(), ancillary_ds.x.max()),
    )

    ancillary_ds.y.attrs = dict(
        standard_name="projection_y_coordinate",
        long_name="y coordinate of projection",
        units="meters",
        axis="Y",
        valid_range=(ancillary_ds.y.min(), ancillary_ds.y.max()),
    )

    return ancillary_ds


if __name__ == "__main__":
    hemisphere = SOUTH

    anc_fp = get_ancillary_filepath(
        hemisphere=hemisphere,
        resolution="12.5",
    )

    anc_ds = xr.load_dataset(anc_fp)

    fixed = fix_xy_attrs(ancillary_ds=anc_ds)

    backup_fp = anc_fp.parent / (anc_fp.name + ".old")
    anc_fp.rename(backup_fp)
    logger.info(f"Moved existing ancillary file from {anc_fp} to {backup_fp}")

    fixed.to_netcdf(anc_fp)
    logger.info(f"Wrote fixed {anc_fp}. If everything looks OK, remove {backup_fp}")
