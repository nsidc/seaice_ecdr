"""Find the min latitude for each value in the pole hole mask.

Usage:
    python ./find_polemask_latmin.py <ancillary_file>
  eg:
    python scripts/find_polemask_latmin.py /share/apps/G02202_V5/v05r00_ancillary/ecdr-ancillary-psn12.5.nc
"""

import numpy as np
import pyproj
import xarray as xr


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

    psn_crs = pyproj.CRS("epsg:3411")
    ll_crs = pyproj.CRS("epsg:4326")
    transformer = pyproj.Transformer.from_crs(psn_crs, ll_crs)

    for idx, meaning in enumerate(field.attrs["flag_meanings"].split(" ")):
        data = field.to_numpy()
        sat = meaning.replace("_polemask", "")
        bit_val = field.attrs["flag_masks"][idx]

        is_bit = (data & bit_val) > 0
        is_bit.tofile(f"is_bit_{bit_val:02d}.dat")

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
        print(
            f"  {idx}:  bit: {bit_val:2d}  meaning:  {meaning}  sat: {sat}  min_lat: {min_lat}"
        )


if __name__ == "__main__":
    import sys

    try:
        ifn = sys.argv[1]
        ds = xr.load_dataset(ifn)
    except IndexError:
        raise ValueError("No ancillary file given")
    except ValueError or FileNotFoundError:
        raise FileNotFoundError(f"\n  Could not open as dataset: {ifn}")

    find_min_latitudes(ds)
