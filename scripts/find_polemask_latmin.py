"""Find the min latitude for each value in the pole hole mask.

Usage:
    python ./find_polemask_latmin.py <ancillary_file>
  eg:
    python scripts/find_polemask_latmin.py /share/apps/G02202_V5/v05r00_ancillary/ecdr-ancillary-psn12.5.nc
"""

import numpy as np
import pyproj
import xarray as xr

DEFAULT_ANC_FN = "/share/apps/G02202_V5/v05r00_ancillary/ecdr-ancillary-psn12.5.nc"


def get_area_grid():
    """This assumes NH psn12.5"""
    area_ds_fn = "/projects/DATASETS/nsidc0771_polarstereo_anc_grid_info/NSIDC0771_CellArea_PS_N12.5km_v1.0.nc"
    ds = xr.load_dataset(area_ds_fn)

    area_grid = ds.cell_area.to_numpy()
    area_grid = area_grid / 1000000.0

    return area_grid


def find_min_latitudes(
    ds,
    varname="polehole_bitmask",
):
    """Find the minimum latitude of the bit mask."""

    is_sh = "_SH_" in ds.crs.long_name

    if is_sh:
        print("No pole mask in Southern Hemisphere")
        return

    try:
        field = ds.data_vars[varname]
    except KeyError:
        print(f"No var {varname} in dataset")
        return

    res = 12500
    print(f"Note: Assuming resolution is {res} min")

    area_grid = get_area_grid()

    psn_crs = pyproj.CRS("epsg:3411")
    ll_crs = pyproj.CRS("epsg:4326")
    transformer = pyproj.Transformer.from_crs(psn_crs, ll_crs)

    for idx, meaning in enumerate(field.attrs["flag_meanings"].split(" ")):
        data = field.to_numpy()
        sat = meaning.replace("_polemask", "")
        bit_val = field.attrs["flag_masks"][idx]

        is_bit = (data & bit_val) > 0
        is_bit.tofile(f"is_bit_{bit_val:02d}.dat")

        area_sum = np.sum(area_grid[is_bit])

        where_mask = np.where(is_bit)
        ivals = where_mask[1]
        jvals = where_mask[0]

        # Use these values to test that min_lat comes to 30.890564...
        # ivals = [0, 607, 607, 0]
        # jvals = [0, 0, 895, 895]

        x_psn = []
        y_psn = []

        for ival, jval in zip(ivals, jvals):
            xval = ds.x[ival].to_numpy()
            yval = ds.y[jval].to_numpy()

            x_psn.append(xval - res / 2)
            y_psn.append(yval - res / 2)

            x_psn.append(xval + res / 2)
            y_psn.append(yval - res / 2)

            x_psn.append(xval + res / 2)
            y_psn.append(yval + res / 2)

            x_psn.append(xval - res / 2)
            y_psn.append(yval + res / 2)

        lat, lon = transformer.transform(x_psn, y_psn)
        latarr = np.array(lat)
        min_lat = latarr.min()
        n_pole_cells = np.sum(np.where(is_bit, 1, 0))
        print(
            f"  sat: {sat}  bit: {bit_val:2d}  label: {meaning}    min_lat: {min_lat:7.4f}   n_12.5km_pole_cells: {n_pole_cells:5,d}  area: {area_sum:12,.2f} km^2"
        )


if __name__ == "__main__":
    import sys

    try:
        ifn = sys.argv[1]
    except IndexError:
        ifn = DEFAULT_ANC_FN
        print(f"No ancillary file given\nAssuming default: {ifn}")

    try:
        ds = xr.load_dataset(ifn)
    except ValueError or FileNotFoundError:
        raise FileNotFoundError(f"\n  Could not open as dataset: {ifn}")

    find_min_latitudes(ds)
