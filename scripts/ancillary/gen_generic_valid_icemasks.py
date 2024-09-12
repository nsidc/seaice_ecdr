"""Create land-agnostic versions of the CDRv4 valid ice masks

gen_generic_valid_icemasks.py

The issue with the original valid ice masks is that they include the land
and lakes.  This version extends oceans and lakes separately. 
"""

import numpy as np
from netCDF4 import Dataset
from scipy.ndimage import generate_binary_structure
from scipy.signal import convolve2d

OCEAN = 0
LAKE = 252
COAST = 253
LAND = 254

INVALOCEAN = 100
INVALLAKE = 125
VALOCEAN = 200
VALLAKE = 225


def get_grid_dims(gridid):
    if gridid == "psn25":
        xdim = 304
        ydim = 448
    elif gridid == "pss25":
        xdim = 316
        ydim = 332
    else:
        raise ValueError(f"gridid not implemented: {gridid}")

    return xdim, ydim


def get_grid_hem(gridid):
    if gridid == "psn25":
        hem = "n"
    elif gridid == "pss25":
        hem = "s"
    else:
        raise ValueError(f"gridid not implemented: {gridid}")

    return hem


def is_adjacent(boolarr):
    """Calc pixels adjacent to boolean array"""
    kernel = generate_binary_structure(2, 1)
    convolved = convolve2d(
        boolarr.astype(np.uint8),
        kernel,
        mode="same",
        boundary="fill",
        fillvalue=0,
    )

    return convolved > 0


def fix_valid(validarr, gridid, month_index):
    """Implement known fixes for CD$v4 valid ice mask"""
    if gridid == "psn25":
        # There is a small pocket of ocean that is missed in some of the
        # valid ice arrays
        psn25_errors = [
            # check_index then nearby_index
            ((236, 180), (234, 182)),
            ((216, 0), (216, 1)),
            ((217, 0), (217, 1)),
            ((218, 0), (218, 1)),
            ((219, 0), (219, 1)),
            ((220, 0), (220, 2)),
            ((220, 1), (220, 2)),
            ((77, 405), (78, 405)),
            ((76, 405), (78, 405)),
            ((76, 406), (78, 405)),
            ((75, 406), (78, 405)),
            ((74, 406), (78, 405)),
            ((74, 407), (78, 405)),
            ((73, 407), (78, 405)),
            ((72, 408), (78, 405)),
            ((71, 408), (78, 405)),
            ((70, 409), (78, 405)),
            ((85, 411), (86, 411)),
            ((53, 348), (53, 347)),
            ((87, 404), (98, 404)),
            ((82, 407), (83, 407)),
            ((106, 407), (107, 407)),
            ((267, 316), (267, 315)),
            ((271, 312), (270, 312)),
            ((25, 181), (25, 180)),
            ((25, 190), (25, 189)),
            ((25, 179), (25, 178)),
            ((135, 69), (135, 70)),
            ((139, 42), (139, 43)),
            ((285, 308), (285, 309)),
            ((286, 313), (286, 314)),
            ((260, 286), (260, 285)),
            ((267, 316), (267, 317)),
            ((267, 315), (267, 317)),
            ((114, 422), (115, 422)),
            ((113, 422), (115, 422)),
            ((112, 422), (111, 421)),
            ((111, 422), (111, 421)),
            ((25, 190), (25, 191)),
            ((27, 176), (27, 175)),
        ]

        for check_index, nearby_index in psn25_errors:
            check_value = validarr[check_index[1], check_index[0]]
            nearby_value = validarr[nearby_index[1], nearby_index[0]]
            if check_value != nearby_value:
                validarr[check_index[1], check_index[0]] = nearby_value
                print(
                    f"Mismatch between {check_index} and {nearby_index} (month={month_index + 1}): fixing"
                )

    return validarr


def gen_generic_valid_icemasks(gridid):
    """Create valid ice masks that are landmask-independent."""
    xdim, ydim = get_grid_dims(gridid)
    hem = get_grid_hem(gridid)
    nmonths = 12

    incfn = f"./G02202-cdr-ancillary-{hem}h.nc"
    ds_in = Dataset(incfn)
    validice_arr_v4 = np.array(ds_in.variables["valid_ice_mask"])
    assert validice_arr_v4.shape == (12, ydim, xdim)

    # CDRv4 land mask is ocean=0, lake=252, coast=253, land=254
    landmask = np.array(ds_in.variables["landmask"])
    is_ocean = landmask == OCEAN
    is_lake = landmask == LAKE
    is_coast = landmask == COAST

    # Loop through all months, creating no-land versions of the mask
    newvalid_12months = np.zeros_like(validice_arr_v4, dtype=np.uint8)
    for m in range(nmonths):
        valid = validice_arr_v4[m, :, :]

        # newvalid will have values:
        #  0: not yet set
        #  100: not-valid-seaice-ocean (INVALOCEAN)
        #  125: not-valid-seaice-lake (INVALLAKE)
        #  200: valid-seaice-ocean (VALOCEAN)
        #  225: valid-seaice-lake (VALLAKE)

        valid = fix_valid(valid, gridid, m)

        # initialize with valid/invalid ocean/lake
        newvalid = np.zeros_like(valid, dtype=np.uint8)
        newvalid[is_ocean & (valid == 0)] = INVALOCEAN  # invalid_ocean
        newvalid[is_lake & (valid == 0)] = INVALLAKE  # invalid_lake
        newvalid[is_ocean & (valid == 1)] = VALOCEAN  # valid_ocean
        newvalid[is_lake & (valid == 1)] = VALLAKE  # valid_lake

        # Loop until all pixels are explicitly assigned a value
        n_unset = np.sum(np.where(newvalid == 0, 1, 0))
        prior_n_unset = n_unset

        while n_unset > 0:
            # Cyclicly dilate: validocean, validlake, invalidocean, invalid lake

            # validocean
            is_adj_validocean = is_adjacent(newvalid == VALOCEAN)
            newvalid[is_adj_validocean & (newvalid == 0)] = VALOCEAN

            # validlake
            is_adj_validlake = is_adjacent(newvalid == VALLAKE)
            newvalid[is_adj_validlake & (newvalid == 0)] = VALLAKE

            # invalidocean
            is_adj_invalidocean = is_adjacent(newvalid == INVALOCEAN)
            newvalid[is_adj_invalidocean & (newvalid == 0)] = INVALOCEAN

            # invalidlake
            # Note: By inspection, there were no lake grid cells that were
            #       ever marked "invalid" ice
            is_adj_invalidlake = is_adjacent(newvalid == INVALLAKE)
            newvalid[is_adj_invalidlake & (newvalid == 0)] = INVALLAKE

            n_invalid_lake = np.sum(np.where(is_adj_invalidlake, 1, 0))
            if n_invalid_lake > 0:
                print("***********")
                print(f"num invalid lake: {n_invalid_lake}")
                print("***********")

            n_unset = np.sum(np.where(newvalid == 0, 1, 0))
            if n_unset == prior_n_unset:
                print("n_unset did not change!")
                breakpoint()

            prior_n_unset = n_unset

        # Add the coast back in
        is_valid = (newvalid == VALLAKE) | (newvalid == VALOCEAN)
        newvalid[is_coast] = 75
        newvalid[is_coast & is_valid] = 250

        newvalid_12months[m, :, :] = newvalid[:, :]

    ofn = f"cdrv4_{gridid}_validice_12m_corrected.dat"
    newvalid_12months.tofile(ofn)
    print(f"Wrote: {ofn}")


if __name__ == "__main__":
    for gridid in ("psn25", "pss25"):
        gen_generic_valid_icemasks(gridid)
