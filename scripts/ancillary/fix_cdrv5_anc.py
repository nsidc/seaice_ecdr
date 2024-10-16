"""Fix the 25km v5 ancillary files.

NOTE: This will calculate the ancillary xr Dataset, but then NOT overwrite
      an existing netCDF file.

NOTE: This assumes that the old files are:
    - misnamed
    - have the proper surface_type field
    - have WRONG adj123 field
    - have WRONG l90c (land-as-90% siconc) field
    - have _FillValue for 

Usage:
    python fix_cdrv5_anc.py psn25  (for Northern Hemisphere)
        input:
            ecdr-ancillary-psn25.nc
        output:
            g02202-ancillary-psn25-v05r00.nc

    python fix_cdrv5_anc.py pss25  (for Southern Hemisphere)
        input:
            ecdr-ancillary-pss25.nc
        output:
            g02202-ancillary-pss25-v05r00.nc

                crs: latlon
              month: <generated here>
                  x: latlon
                  y: latlon
           latitude: latlon
          longitude: latlon
       surface_type: regions
             adj123: <code>
               l90c: <external code>
  min_concentration: ecdr_anc
   invalid_ice_mask: ecdr_anc
   polehole_bitmask: ecdr_anc
"""

import os

import numpy as np
import xarray as xr
from pm_icecon.land_spillover import create_land90
from scipy.signal import convolve2d

old_ecdr_anc_fn = {
    "psn12.5": "/share/apps/G02202_V5/v05r00_ancillary/ecdr-ancillary-psn12.5.nc",
    "pss12.5": "/share/apps/G02202_V5/v05r00_ancillary/ecdr-ancillary-pss12.5.nc",
    "psn25": "./ecdr-ancillary-psn25.nc",
    "pss25": "./ecdr-ancillary-pss25.nc",
    # "psn25": "/share/apps/G02202_V5/v05r00_ancillary/ecdr-ancillary-psn25_new.nc",
    # "pss25": "/share/apps/G02202_V5/v05r00_ancillary/ecdr-ancillary-pss25_new.nc",
}

new_ecdr_anc_fn = {
    "psn12.5": "/share/apps/G02202_V5/v05r00_ancillary/ecdr-ancillary-psn12.5.nc",
    "pss12.5": "/share/apps/G02202_V5/v05r00_ancillary/ecdr-ancillary-pss12.5.nc",
    "psn25": "./g02202-ancillary-psn25-v05r00.nc",
    "pss25": "./g02202-ancillary-pss25-v05r00.nc",
    # "psn25": "/share/apps/G02202_V5/v05r00_ancillary/ecdr-ancillary-psn25_new.nc",
    # "pss25": "/share/apps/G02202_V5/v05r00_ancillary/ecdr-ancillary-pss25_new.nc",
}


def find_coast(mask, ocean_values=(50,), land_values=(75, 250)):
    """Find indices where land values are adjacent to ocean values."""
    is_ocean = np.zeros(mask.shape, dtype=bool)
    for val in ocean_values:
        is_ocean[mask == val] = True

    is_land = np.zeros(mask.shape, dtype=bool)
    for val in land_values:
        is_land[mask == val] = True

    kernel = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    convolved = convolve2d(
        is_ocean.astype(np.uint8), kernel, mode="same", boundary="fill", fillvalue=0
    )
    is_coast = is_land & (convolved > 0)

    return is_coast


def calc_adj123_np(surftype_da, ocean_val=50, coast_val=200):
    """Compute the land-adjacency field for this surfacetype mask.
    Input:
        DataArray with values:
            ocean: ocean_val (default 50)
            coast: coast_val (default 200)
    Output:
        Numpy array with adjacency values of 1, 2, 3
    """
    surftype = surftype_da.to_numpy()
    is_ocean = surftype == ocean_val
    is_land = ~is_ocean

    kernel = [
        [
            1,
            1,
            1,
        ],
        [1, 1, 1],
        [
            1,
            1,
            1,
        ],
    ]
    adj123_arr = np.zeros(surftype.shape, dtype=np.uint8)
    adj123_arr[~is_ocean] = 255
    for adj_val in range(1, 4):
        is_unlabeled = adj123_arr == 0
        is_labeled = (adj123_arr == 255) | ((~is_unlabeled) & (adj123_arr < adj_val))
        convolved = convolve2d(
            is_labeled.astype(np.uint8),
            kernel,
            mode="same",
            boundary="fill",
            fillvalue=0,
        )
        is_newly_labeled = is_unlabeled & (convolved > 0)
        adj123_arr[is_newly_labeled] = adj_val

    # Swap land and unassigned
    # so that land has value of 0 and unlabeled ocean has value of 255
    is_unassigned_ocean = (adj123_arr == 0) & is_ocean

    adj123_arr[is_land] = 0
    adj123_arr[is_unassigned_ocean] = 255

    return adj123_arr


def fix_ecdr_anc_file(gridid):
    """Create the cdrv5 ancillary ile for this GridId.
    Note that this may use pre-existing cdrv5 ancillary fields
    as will as other primary sources."""
    if gridid == "psn25":
        old_anc_fn = old_ecdr_anc_fn["psn25"]
        newres_anc_fn = new_ecdr_anc_fn["psn25"]
    elif gridid == "pss25":
        old_anc_fn = old_ecdr_anc_fn["pss25"]
        newres_anc_fn = new_ecdr_anc_fn["pss25"]
    else:
        raise RuntimeError(f"GridID not implemented: {gridid}")

    if os.path.isfile(newres_anc_fn):
        raise RuntimeError(f"Output file exists, aborting\n{newres_anc_fn}")

    ds_old = xr.open_dataset(old_anc_fn)
    ds_new = ds_old.copy()

    # Find the land-adjacency matrix which indicates whether grid cells are
    #  1, 2, or 3 grid cells away from land (coast).
    adj123_np = calc_adj123_np(ds_old["surface_type"])
    ds_new["adj123"].data[:] = adj123_np[:]

    # Calculate the land90conc field
    l90c_np = create_land90(adj123=ds_new["adj123"].data)
    ds_new["l90c"].data[:] = l90c_np[:]

    # Ensure that crs var has units of meters
    ds_new["crs"].attrs["units"] = "meters"

    # Write out ancillary file
    ds_new.to_netcdf(newres_anc_fn)
    print(f"Wrote new {gridid} ancillary file to: {newres_anc_fn}")


if __name__ == "__main__":
    import sys

    supported_gridids = ("psn25", "pss25")

    try:
        gridid = sys.argv[1]
        assert gridid in supported_gridids
    except IndexError:
        raise RuntimeError(
            "No GridID provided." f"\n  Valid values: {supported_gridids}"
        )
    except AssertionError:
        raise RuntimeError(
            "GridID {gridid} not recognized." f"\n  Valid values: {supported_gridids}"
        )

    fix_ecdr_anc_file(gridid)
