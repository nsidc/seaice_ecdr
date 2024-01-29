"""Code to make CDRv4 valid ice masks consistent with NT2 spillovers.

The gist of the approach is to fix some erroneously non-filled areas
and then to ensure that there are not near-coast valid sea ice locations
that could not be anchored to an off-shore sea ice pixel"""

import numpy as np
import xarray as xr
from scipy.signal import convolve2d


def fix_valid_ice_masks(vimda, surfmaskda):
    vim = vimda.to_numpy()
    old_vim = vim.copy()
    n_months, ydim, xdim = vim.shape
    print(f"vim shape: {vim.shape}")
    print(f"surfmaskda shape: {surfmaskda.shape}")

    is_notocean = surfmaskda.data > 0
    is_lake = surfmaskda.data == 252

    # Calculate adj123
    kernel3x3 = np.ones((3, 3), dtype=np.uint8)
    adj123 = np.zeros(surfmaskda.data.shape, dtype=np.uint8)
    adj123[is_notocean] = 255
    n_adj_values = 3
    for adj_val in range(1, n_adj_values + 1):
        is_unlabeled = adj123 == 0
        is_labeled = (adj123 == 255) | ((~is_unlabeled) & (adj123 < adj_val))
        convolved = convolve2d(
            is_labeled.astype(np.uint8),
            kernel3x3,
            mode="same",
            boundary="fill",
            fillvalue=0,
        )
        is_newly_labeled = is_unlabeled & (convolved > 0)
        adj123[is_newly_labeled] = adj_val

    # Set land to 0 and open ocean to 255
    adj123[adj123 == 255] = 200
    adj123[adj123 == 0] = 255
    adj123[adj123 == 200] = 0

    is_dist3 = adj123 == 3
    is_dist12 = (adj123 == 1) | (adj123 == 2)

    # adj123 now has values:
    #   0: land
    #   1: one away from coast
    #   2: two away from coast
    #   3: three away from coast
    # 255: distant from coast (ie, > 3)

    new_vim = np.zeros(vim.shape, dtype=np.uint8)
    for m in range(12):
        # set up one-month-at-a-time views into 3d array
        vim_slice = vim[m, :, :]
        new_vim_slice = new_vim[m, :, :]
        new_vim_slice[:] = vim_slice[:].copy()

        # Find where the near-coast values are anchored by a nearby dist3 cell
        nearest_dist3 = np.zeros(vim_slice.shape, dtype=np.uint8)
        nearest_dist3[:] = 255
        nearest_dist3[is_dist3] = vim_slice[is_dist3]

        # Now, allow a 7x7 "stamp" of dist3 to permit nearby valid ice concs
        kernel7x7 = np.ones((7, 7), dtype=np.uint8)
        is_valid = nearest_dist3 == 1
        convolved7 = convolve2d(
            is_valid.astype(np.uint8),
            kernel7x7,
            mode="same",
            boundary="fill",
            fillvalue=0,
        )
        is_allowed_by_7x7stamp = (convolved7 > 0) & (is_dist12) & (vim_slice == 1)

        n_dist12_unassigned = np.sum(
            np.where((nearest_dist3 == 255) & (is_dist12), 1, 0)
        )
        n_unassigned_last = n_dist12_unassigned

        while n_dist12_unassigned > 0:
            # First, expand the valid dist3 cells
            is_valid = nearest_dist3 == 1
            n_near_valid = convolve2d(
                is_valid.astype(np.uint8),
                kernel3x3,
                mode="same",
                boundary="fill",
                fillvalue=0,
            )
            is_newly_valid = (n_near_valid > 0) & (is_dist12)
            nearest_dist3[is_newly_valid] = 1

            # Now, expand the invalid dist3 cells
            is_invalid = nearest_dist3 == 0
            n_near_invalid = convolve2d(
                is_invalid.astype(np.uint8),
                kernel3x3,
                mode="same",
                boundary="fill",
                fillvalue=0,
            )
            is_newly_invalid = (
                (n_near_invalid > 0) & (is_dist12) & ~is_allowed_by_7x7stamp
            )
            nearest_dist3[is_newly_invalid] = 0

            n_dist12_unassigned = np.sum(
                np.where((nearest_dist3 == 255) & (is_dist12), 1, 0)
            )
            if n_dist12_unassigned == n_unassigned_last:
                # Unfortuntely, this can happen if the mask has
                # ocean grid cells that are disconnected from the world ocean
                print(
                    f"Stopping because no progress with {n_unassigned_last} remaining"
                )
                is_unreachable_or_assignable = (
                    (nearest_dist3 == 255) & is_dist12 & is_allowed_by_7x7stamp
                )
                nearest_dist3[is_unreachable_or_assignable] = vim_slice[
                    is_unreachable_or_assignable
                ]
                break
            n_unassigned_last = n_dist12_unassigned

        # Started by copying the old vim slice (above)

        # Set lakes to not valid
        new_vim_slice[is_lake] = 0

        # Remove valid ice if no nearby dist3 cell
        cant_be_seaice = is_dist12 & (nearest_dist3 == 0)
        new_vim_slice[cant_be_seaice] = 0

        # Add in the values allowed by 7x7 stamp
        overuled_by_stamp = is_allowed_by_7x7stamp & (vim_slice == 1)
        new_vim_slice[overuled_by_stamp] = 1

        # Set no-access points to invalid (2)
        is_unassignable = is_dist12 & (nearest_dist3 == 255)
        new_vim_slice[is_unassignable] = 2

        # Make corrections from bad fill in original
        # Note: land mask MUST be used to mask out valid after this
        if xdim == 304:
            new_vim_slice[411, 85] = new_vim_slice[411, 86]
            new_vim_slice[405:410, 70:78] = new_vim_slice[405, 78]
            new_vim_slice[0:2, 216:221] = new_vim_slice[1, 217]

        # Add in the land values
        new_vim_slice[is_notocean] = 3
        old_vim_slice = old_vim[m, :, :]
        old_vim_slice[is_notocean] = 3

    new_vim.tofile(f"new_vim_{xdim}x{ydim}x{n_months}.dat")
    old_vim.tofile(f"old_vim_{xdim}x{ydim}x{n_months}.dat")

    return xr.DataArray(new_vim)


if __name__ == "__main__":
    # Run NH
    ds = xr.load_dataset(
        "/projects/DATASETS/NOAA/G02202_V4/ancillary/G02202-cdr-ancillary-nh.nc"
    )
    valid_ice_masks = ds.valid_ice_mask
    surfmask = ds.landmask
    new_valid_ice_masks = fix_valid_ice_masks(valid_ice_masks, surfmask)
    new_valid_ice_masks.to_netcdf("fixed_cdrv4_masks.nc")

    print("Now, run for SH...")
