"""Adapt the CDR v4 ancillary daily climatology files for CDR v5.
"""
from pathlib import Path
from typing import get_args

import xarray as xr
from pm_tb_data._types import Hemisphere

from seaice_ecdr.constants import CDR_ANCILLARY_DIR
from seaice_ecdr.grid_id import get_grid_id

CDR_V4_ANCILLARY_DIR = Path("/home/trst2284/code/seaice_cdr/source/ancillary/")


def get_v4_climatology(*, hemisphere: Hemisphere) -> xr.Dataset:
    ds = xr.open_dataset(CDR_V4_ANCILLARY_DIR / f"doy-validice-{hemisphere}-smmr.nc")

    return ds


def make_v5_climatology(*, hemisphere: Hemisphere):
    grid_id = get_grid_id(hemisphere=hemisphere, resolution="25")
    output_filepath = (
        CDR_ANCILLARY_DIR / f"ecdr-ancillary-{grid_id}-smmr-invalid-ice.nc"
    )

    v4_ds = get_v4_climatology(hemisphere=hemisphere)

    v4_ds.to_netcdf(output_filepath)


if __name__ == "__main__":
    for hemisphere in get_args(Hemisphere):
        make_v5_climatology(hemisphere=hemisphere)
