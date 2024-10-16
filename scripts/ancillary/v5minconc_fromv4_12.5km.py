import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import maximum_filter, median_filter

"""Create a v5 minconc array from the v4 minconc array

This is the 12.5km version.  Start by filling in with overlying 25km v4cell, then max-fill the remainder
"""


def get_hem(gridid):
    if gridid == "psn12.5":
        return "n"
    elif gridid == "psn25":
        return "n"
    elif gridid == "pss12.5":
        return "s"
    elif gridid == "pss25":
        return "s"
    else:
        raise ValueError(f"Could not determine hem of {gridid}")


def get_res(gridid):
    if gridid == "psn25":
        return "25"
    elif gridid == "pss25":
        return "25"
    elif gridid == "psn12.5":
        return "12.5"
    elif gridid == "pss12.5":
        return "12.5"
    else:
        raise ValueError(f"Could not determine res of {gridid}")


def gen_v5minconc_12p5(gridid):
    """Generate the extended v5 min_conc field from the v4 version"""
    hem = get_hem(gridid)
    # res = get_res(gridid)

    # This will be the 25km v4 anc file (for minconc and v4land)
    ds4 = Dataset(f"./G02202-cdr-ancillary-{hem}h.nc")
    ds4.set_auto_maskandscale(False)
    ds4land = np.array(ds4.variables["landmask"])
    nonoceanv4 = ds4land != 0
    v4minconc_i2 = np.array(ds4.variables["min_concentration"])

    # This will be the 12.5km v5 anc file (for surface type)
    ds5 = Dataset(f"./ecdr-ancillary-{gridid}.nc")
    ds5.set_auto_maskandscale(False)
    ds5land = np.array(ds5.variables["surface_type"])
    nonoceanv5 = ds5land != 50

    ydim12, xdim12 = ds5land.shape
    v5minconc_i2 = np.zeros((ydim12, xdim12), dtype=np.int16)
    for joff in range(2):
        for ioff in range(2):
            v5_slice = v5minconc_i2[joff::2, ioff::2]
            v5_slice[:, :] = v4minconc_i2[:, :]
            v5_slice[nonoceanv4] = 1200  # 1200 for v4land/v5ocean

    v5minconc_i2[nonoceanv5] = 1500  #    1500 for v5land

    # Check the initial fill
    # breakpoint()
    # print('check v5minconc_i2 init fill')

    # Do an initial pass, replacing values with the max of the '+' locality
    kernel = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    minconc = v5minconc_i2.copy()
    minconc[minconc >= 1200] = -100
    expanded = median_filter(minconc, footprint=kernel)
    has_new_values = (v5minconc_i2 > 0) & (v5minconc_i2 < 1200) & (expanded >= 0)
    v5minconc_i2[has_new_values] = expanded[has_new_values]

    # Check the initial smooth
    # breakpoint()
    # print('check v5minconc_i2 init fill')

    # Now, update v5minconc_i2 until all nonocean cells have a value
    n_unassigned = np.sum(np.where(v5minconc_i2 == 1200, 1, 0))

    # First, attempt '+' filter
    kernel = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    last_n_unassigned = 0
    n_iterations = 0
    while (n_unassigned > 0) and (n_unassigned != last_n_unassigned):
        minconc = v5minconc_i2.copy()
        minconc[minconc >= 1200] = -100
        expanded = maximum_filter(minconc, footprint=kernel)

        has_new_values = (v5minconc_i2 == 1200) & (expanded >= 0)
        v5minconc_i2[has_new_values] = expanded[has_new_values]

        try:
            assert np.all(v5minconc_i2 >= 0)
        except AssertionError:
            breakpoint()

        last_n_unassigned = n_unassigned
        n_unassigned = np.sum(np.where(v5minconc_i2 == 1200, 1, 0))
        n_iterations += 1

    print(f"n still unassigned after {n_iterations}: {n_unassigned}")

    footprint_radius = 3
    while (n_unassigned > 0) and (footprint_radius < 21):
        last_n_unassigned = 0
        n_iterations = 0
        while (n_unassigned > 0) and (n_unassigned != last_n_unassigned):
            print(f"{footprint_radius}: {n_unassigned}", flush=True)
            minconc = v5minconc_i2.copy()
            minconc[minconc >= 1200] = -100
            expanded = maximum_filter(minconc, size=footprint_radius)

            assert minconc.shape == expanded.shape

            has_new_values = (v5minconc_i2 == 1200) & (expanded >= 0)
            v5minconc_i2[has_new_values] = expanded[has_new_values]

            assert np.all(v5minconc_i2 >= 0)

            last_n_unassigned = n_unassigned
            n_unassigned = np.sum(np.where(v5minconc_i2 == 1200, 1, 0))
            n_iterations += 1

        footprint_radius += 2

    ofn = f"v5minconc_i2_{gridid}.dat"
    v5minconc_i2.tofile(ofn)
    print(f"Wrote: {ofn}")

    print(f"n still unassigned after {n_iterations}: {n_unassigned}")
    print(f"  footprint_radius: {footprint_radius}")


if __name__ == "__main__":
    for gridid in ("psn12.5", "pss12.5"):
        gen_v5minconc_12p5(gridid)
