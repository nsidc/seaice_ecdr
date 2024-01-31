"""Generate xarray data set from a grid_id.

grid_id_to_xr_dataset

Return an xarray dataset with appropriate geolocation variables for
a given NSIDC grid_id
"""


import datetime as dt

import xarray as xr
from pm_tb_data._types import Hemisphere

from seaice_ecdr._types import ECDR_SUPPORTED_RESOLUTIONS
from seaice_ecdr.ancillary import get_ancillary_ds


def get_empty_ds_with_time(
    *, hemisphere: Hemisphere, resolution: ECDR_SUPPORTED_RESOLUTIONS, date: dt.date
) -> xr.Dataset:
    """Return xarray dataset.

    with appropriate geolocation dataarrays:
        x
        y
        time
        crs

    Because none of these need compression, no 'encoding' dictionary is
    returned
    """
    ancillary_ds = get_ancillary_ds(
        hemisphere=hemisphere,
        resolution=resolution,
    )

    time_as_int = (date - dt.date(1970, 1, 1)).days
    time_da = xr.DataArray(
        name="time",
        data=[int(time_as_int)],
        dims=["time"],
        attrs={
            "standard_name": "time",
            "long_name": "ANSI date",
            "calendar": "standard",
            "axis": "T",
            "units": "days since 1970-01-01",
            "coverage_content_type": "coordinate",
            "valid_range": [int(0), int(30000)],
        },
    )

    return_ds = xr.Dataset(
        data_vars=dict(
            x=ancillary_ds.x,
            y=ancillary_ds.y,
            crs=ancillary_ds.crs,
            time=time_da,
        ),
    )

    return return_ds
