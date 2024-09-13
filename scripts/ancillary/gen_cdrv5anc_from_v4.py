"""Create a CDRv5-compatible ancillary file from CDRv4 ancillary files

Note: this assumes you've run the code ./gen_generic_valid_icemasks.py
      in order to produce local files:
         ./cdrv4_psn25_validice_12m_corrected.dat
         ./cdrv4_pss25_validice_12m_corrected.dat

The v5 files need:
    latitude:  not used for calculations, so can come from nsidc0771 or v5 file
    longitude: not used for calculations, so can come from nsidc0771 or v5 file
    surface_type: should come from nsidc0780, but is adapted here for compat
    adj123:  calculated from surface_type
    l90c:    calcluated from surface_type
    min_concentration: adapted from CDRv4 file
    invalid_ice_mask:  adapted from CDRv4 file
    polehole_bitmask [NH only]:  adapted from CDRv4 file


"""

import numpy as np
from land_spillover import create_land90
from netCDF4 import Dataset
from scipy.ndimage import binary_dilation, generate_binary_structure

fnmap = {
    "v5_orig_psn25": "ecdr-ancillary-psn25.nc",
    "v5_orig_pss25": "ecdr-ancillary-pss25.nc",
    "v5_from_v4_psn25": "ecdr-ancillary-psn25-v04r00.nc",
    "v5_from_v4_pss25": "ecdr-ancillary-pss25-v04r00.nc",
    "v4_psn25": "G02202-cdr-ancillary-nh.nc",
    "v4_pss25": "G02202-cdr-ancillary-sh.nc",
}


def gen_adj123(is_ocean, nlevels=3):
    """Gen diagonal-orthogonality-distance map"""
    # Initialize so that non-ocean is zero and ocean is 255
    is_ocean.tofile("is_ocean.dat")
    dists = np.zeros_like(is_ocean, dtype=np.uint8)
    dists[is_ocean] = 255
    kernel = generate_binary_structure(2, 2)
    for dist in range(nlevels):
        is_labeled = dists <= dist
        dilated = binary_dilation(is_labeled, structure=kernel)
        is_newly_labeled = (dilated > 0) & (dists == 255)
        dists[is_newly_labeled] = dist + 1
        """ Debug code:
        ofn = f'added_dist_{dist}.dat'
        dists.tofile(ofn)
        print(f'Wrote: {ofn} ({np.sum(np.where(is_newly_labeled, 1, 0))} vals: {np.unique(dists)})')
        """

    return dists


def gen_v4_anc_file(hem):
    """Generate the given hem's anc file
    Expects 'hem' of 'n' or 's'
    Currently set up for 25km only
    """
    gridid = f"ps{hem}25"
    print(f"Setting gridid to: {gridid}")

    ds4 = Dataset(fnmap[f"v4_ps{hem}25"])

    v5fromv4 = {}
    v5fromv4["latitude"] = np.array(ds4.variables["latitude"]).astype(np.float64)
    v5fromv4["longitude"] = np.array(ds4.variables["longitude"]).astype(np.float64)

    ydim, xdim = v5fromv4["latitude"].shape

    v4land = np.array(ds4.variables["landmask"])
    v5land = np.zeros_like(v4land, dtype=np.uint8)
    v5land[v4land == 0] = 50  # Ocean
    v5land[v4land == 252] = 75  # Lake
    v5land[v4land == 253] = 200  # Coast: Note v4 is ortho, v5 is 8-conn
    v5land[v4land == 254] = 250  # Land: Note v4 is ortho, v5 is 8-conn
    v5fromv4["surface_type"] = v5land

    v4minconc = np.array(ds4.variables["min_concentration"])
    # v5 conc is float and ranges from 0.0 to 1.0
    v5minconc = v4minconc.astype(np.float32) / 100.0
    v5fromv4["min_concentration"] = v5minconc

    # Note: these input files come from running:
    #  python ./gen_generic_valid_icemasks.py
    #  They are 12 months x ydim x xdim
    #  and are coded:
    #  Valid sea ice (if land mask allows)
    #    200: ocean, valid
    #    225: lake, valid
    #    250: coast, valid
    #  Not-valid sea ice
    #    100: ocean, invalid
    #    125: lake, invalid  # Note: the original data never had invalid lake
    #     75: coast, invalid
    v4valid_corrected = np.fromfile(
        f"./cdrv4_{gridid}_validice_12m_corrected.dat", dtype=np.uint8
    ).reshape(12, ydim, xdim)

    # The invalid ice mask should be:
    #  - everywhere the valid ice mask is 0 (False)
    #  - all non-ocean pixels
    # Defaults to invalid.  Add in valid locations
    v5validice = np.zeros_like(v4valid_corrected, dtype=np.uint8)
    v5invalidice = np.zeros_like(v4valid_corrected, dtype=np.uint8)
    is_v5_ocean = v5fromv4["surface_type"] == 50
    for month in range(12):
        # This should be a slice?
        thismonth_v5 = v5validice[month, ::]
        thismonth_v4 = v4valid_corrected[month, ::]
        thismonth_v5[is_v5_ocean & (thismonth_v4 == 200)] = 1  # valid expanded-ocean
        thismonth_v5[is_v5_ocean & (thismonth_v4 == 225)] = 1  # valid expanded-lake
        thismonth_v5[is_v5_ocean & (thismonth_v4 == 250)] = 1  # valid coast
    # Now flip those valids to invalids (and vice-versa)
    v5invalidice[v5validice == 0] = 1
    v5invalidice[v5validice == 1] = 0
    v5fromv4["invalid_ice_mask"] = v5invalidice
    v5invalidice.tofile("v5invalidice.dat")

    if "polehole" in ds4.variables.keys():
        # CDRv4 mask is: 1-SMMR, 2-SSMI, 4-SSMIS
        v4_polemask = np.array(ds4.variables["polehole"])
        v5_polemask = np.zeros_like(v4_polemask, dtype=np.uint8)

        """
        # Test of the bit-checking technique
        # First, reproducing to verify technique
        for bitnum in range(2 + 1):
            is_bit = np.bitwise_and(v4_polemask, np.left_shift(1, bitnum))
            v5_polemask[is_bit > 0] += 2 ** bitnum

        if np.all(v4_polemask == v5_polemask):
            print('bitmask technique works')
        else:
            print('bitmask technique FAILS')
            v4_polemask.tofile('v4_polemask.dat')
            v5_polemask.tofile('v5_polemask.dat')
            breakpoint()
        """

        # CDRv5 mask is: 1-SMMR, 2-F8, 4-F11, 8-F13, 16-F17, 32-AME, 64-AM2
        # So, will convert by mapping bits:
        #  1 -> 1 (total: 1)
        #  2 -> 2, 4, 8 (total: 14)
        #  4 -> 16, 32, 64 (total: 112)
        is_bit0 = np.bitwise_and(v4_polemask, np.left_shift(1, 0))
        is_bit1 = np.bitwise_and(v4_polemask, np.left_shift(1, 1))
        is_bit2 = np.bitwise_and(v4_polemask, np.left_shift(1, 2))

        is_bit0.tofile("is_bit0.dat")
        is_bit1.tofile("is_bit1.dat")
        is_bit2.tofile("is_bit2.dat")

        v5_polemask[is_bit0 > 0] += 1  # SMMR
        v5_polemask[is_bit1 > 0] += 14  # SSMI => F8, F11, F13
        v5_polemask[is_bit2 > 0] += 112  # SSMIS => F17, AME, AM2

        v5fromv4["polehole_bitmask"] = v5_polemask
        v4_polemask.tofile("v4_polemask.dat")
        v5_polemask.tofile("v5_polemask.dat")

    v4_adj123 = gen_adj123(is_v5_ocean)
    v4_adj123.tofile("v4_adj123.dat")

    v5fromv4["adj123"] = v4_adj123

    v4land90 = create_land90(adj123=v4_adj123)
    v5fromv4["l90c"] = v4land90

    print("v5fromv4 keys:")
    for key in v5fromv4.keys():
        print(f"  {key}")

    # Now, let's create a new file by copying the old one, and then
    # updating its values
    orig_v5_fn = fnmap[f"v5_orig_{gridid}"]
    new_v5_fn = fnmap[f"v5_from_v4_{gridid}"]

    import sh

    # print(f"shell cp options: {f'-vn {orig_v5_fn} {new_v5_fn}'}")
    print(sh.cp("-vn", orig_v5_fn, new_v5_fn))

    # Open the new file...and then edit it
    newds = Dataset(new_v5_fn, "r+")
    print(f"newds vars: {newds.variables.keys()}")
    possible_var_list = [
        "surface_type",
        "adj123",
        "l90c",
        "min_concentration",
        "invalid_ice_mask",
        "polehole_bitmask",
    ]
    var_list = [var for var in possible_var_list if var in newds.variables.keys()]
    for var in var_list:
        print(f"same dtypes {var}: {newds.variables[var].dtype == v5fromv4[var].dtype}")
        print(f"same shapes {var}: {newds.variables[var].shape == v5fromv4[var].shape}")
        newds.variables[var][:] = v5fromv4[var][:]

    if "polehole_bitmask" in newds.variables.keys():
        newds.variables["polehole_bitmask"].valid_range = np.array(
            (0, 127), dtype=np.uint8
        )

    newds.close()
    print(f"Wrote: {new_v5_fn}")


if __name__ == "__main__":
    import sys

    valid_hems = ("n", "s")
    try:
        hem = sys.argv[1][0]
        assert hem in valid_hems

    except IndexError:
        hem = "n"
        print(f"No hem given:  assuming {hem}")
    except AssertionError:
        raise ValueError(f"Given hem ({hem}) not in {valid_hems}")

    gen_v4_anc_file(hem)
