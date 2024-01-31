"""Verify that smmr day-of-year invalid ice masks are consistent with CDRv5.

morph_smmr_masks.py

Usage:
    python morph_smmr_masks.py

This will run both NH and SH for 25km and output a file for each:
    morphed_smmr_doy_masks_psn25.nc
    morphed_smmr_doy_masks_pss25.nc

NOTE: These output files will be barebones, ie no CRS, no attrs, etc.

The purpose of this script is to demonstrate regridding of 
"""

import numpy as np
import xarray as xr
from scipy.signal import convolve2d


def morph_smmr_masks(gridid):
    anc_fn = f"/share/apps/G02202_V5/v05r00_ancillary/ecdr-ancillary-{gridid}.nc"
    anc_ds = xr.load_dataset(anc_fn)
    if gridid == "psn25":
        old_mask_fn = "~/seaice_cdr/source/ancillary/doy-validice-north-smmr.nc"
    elif gridid == "pss25":
        old_mask_fn = "~/seaice_cdr/source/ancillary/doy-validice-south-smmr.nc"
    old_ds = xr.load_dataset(old_mask_fn, mask_and_scale=False)

    surftype = anc_ds.surface_type.to_numpy()
    old_vim = old_ds.daily_icemask.to_numpy()

    # Investigate arrays
    is_new_ocean = surftype == 50
    is_new_land = ~is_new_ocean

    n_days, ydim, xdim = old_vim.shape
    new_vim = np.zeros((n_days, ydim, xdim), dtype=np.uint8)
    new_vim[:] = 255
    for d in range(n_days):
        if (d % 20) == 0:
            print(f"{d} of {n_days}", flush=True)

        old_slice = old_vim[d, :, :]
        new_slice = new_vim[d, :, :]
        new_slice[is_new_land] = 3

        # Valid ice in CDRv4 file has bit 1 set, so values of 1 and 3 match
        # Invalid (daily!) ice in CDRv4 does not have bit 0 set, so ... is 0 or 2
        #   because there are a few places where the daily SMMR mask is valid but where
        #   the CDRv4 monthly mask was not.  So, invalid is 0 or 2
        # Values above 250 encoded lake, coast, land and are ignored here
        is_valid = is_new_ocean & ((old_slice == 1) | (old_slice == 3))
        is_invalid = is_new_ocean & ((old_slice == 0) | (old_slice == 2))

        new_slice[is_valid] = 1
        new_slice[is_invalid] = 0

        n_unassigned = np.sum(np.where(is_new_ocean & (new_slice == 255), 1, 0))
        n_unassigned_prior = n_unassigned
        kernel = np.ones((3, 3), dtype=np.uint8)
        while n_unassigned > 0:
            # Expand valid
            convolved = convolve2d(
                (new_slice == 1).astype(np.uint8),
                kernel,
                mode="same",
                boundary="fill",
                fillvalue=0,
            )
            is_newly_valid = is_new_ocean & (convolved > 0) & (new_slice == 255)
            new_slice[is_newly_valid] = 1

            # Expand invalid
            convolved = convolve2d(
                (new_slice == 0).astype(np.uint8),
                kernel,
                mode="same",
                boundary="fill",
                fillvalue=0,
            )
            is_newly_invalid = is_new_ocean & (convolved > 0) & (new_slice == 255)
            new_slice[is_newly_invalid] = 0

            # Check to see if we've assigned all ocean grid cells to valid or invalid
            n_unassigned = np.sum(np.where(is_new_ocean & (new_slice == 255), 1, 0))
            if n_unassigned == n_unassigned_prior:
                print(f"breaking with {n_unassigned} still unassigned")
                break

            n_unassigned_prior = n_unassigned

        if not np.all(new_slice <= 3):
            print(f"uh oh...not all points were assigned! d={d}")
            breakpoint()

    # Here, new_vim is the (366, ydim, xdim) VALID ice_mask on the new "land" mask
    assert np.all(new_vim <= 3)

    # Convert to INVALID ice mask by swapping 0s and 1s and by replacing the land values
    # with 0s
    vim_is_valid = new_vim == 1
    vim_is_invalid = new_vim == 0
    vim_is_land = new_vim == 3  # this was set above

    # Reassign values, and make sure we got them all
    new_vim[:] = 255
    new_vim[vim_is_invalid] = 1
    new_vim[vim_is_valid] = 0
    new_vim[vim_is_land] = 1

    assert np.all(new_vim <= 1)

    # Write the new fields
    new_da = xr.DataArray(
        data=new_vim,
        name="new_valid_icemask",
        dims=("doy", "y", "x"),
    )

    output_fn = f"morphed_smmr_doy_masks_{gridid}.nc"
    new_da.to_netcdf(output_fn, encoding={"new_valid_icemask": dict(zlib=True)})
    print(f"Wrote: {output_fn}")


if __name__ == "__main__":
    for gridid in ("psn25", "pss25"):
        morph_smmr_masks(gridid)
