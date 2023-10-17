import os

import numpy as np
from loguru import logger

from pm_icecon.land_spillover import create_land90

# TODO: The various directory vars, eg anc_dir, should be either abstracted
#   as a constant or passed as a configuration parameter.
# TODO: So too, the filename template strings should be authoritatively
#   set in a central location.


def read_adj123_file(
    gridid: str = "psn12.5",
    xdim: int = 608,
    ydim: int = 896,
    anc_dir: str = "/share/apps/amsr2-cdr/nasateam2_ancillary",
    adj123_fn_template: str = "{anc_dir}/coastal_adj_diag123_{gridid}.dat",
):
    """Read the diagonal adjacency 123 file."""
    coast_adj_fn = adj123_fn_template.format(anc_dir=anc_dir, gridid=gridid)
    assert os.path.isfile(coast_adj_fn)
    adj123 = np.fromfile(coast_adj_fn, dtype=np.uint8).reshape(ydim, xdim)

    return adj123


def create_land90_conc_file(
    gridid: str = "psn12.5",
    xdim: int = 608,
    ydim: int = 896,
    anc_dir: str = "/share/apps/amsr2-cdr/nasateam2_ancillary",
    adj123_fn_template: str = "{anc_dir}/coastal_adj_diag123_{gridid}.dat",
    write_l90c_file: bool = True,
    l90c_fn_template: str = "{anc_dir}/land90_conc_{gridid}.dat",
):
    """Create the land90-conc file.

    The 'land90' array is a mock sea ice concentration array that is calculated
    from the land mask.  It assumes that the mock concentration value will be
    the average of a 7x7 array of local surface mask values centered on the
    center pixel.  Water grid cells are considered to have a sea ice
    concentration of zero.  Land grid cells are considered to have a sea ice
    concentration of 90%.  The average of the 49 grid cells in the 7x7 array
    yields the `land90` concentration value.
    """
    adj123 = read_adj123_file(gridid, xdim, ydim, anc_dir, adj123_fn_template)
    land90 = create_land90(ajd123=adj123)

    if write_l90c_file:
        l90c_fn = l90c_fn_template.format(anc_dir=anc_dir, gridid=gridid)
        land90.tofile(l90c_fn)
        print(f"Wrote: {l90c_fn}\n  {land90.dtype}  {land90.shape}")

    return land90


def load_or_create_land90_conc(
    gridid: str = "psn12.5",
    xdim: int = 608,
    ydim: int = 896,
    anc_dir: str = "/share/apps/amsr2-cdr/nasateam2_ancillary",
    l90c_fn_template: str = "{anc_dir}/land90_conc_{gridid}.dat",
    overwrite: bool = False,
):
    # Attempt to load the land90_conc field, and if fail, create it
    l90c_fn = l90c_fn_template.format(anc_dir=anc_dir, gridid=gridid)
    if overwrite or not os.path.isfile(l90c_fn):
        data = create_land90_conc_file(
            gridid, xdim, ydim, anc_dir=anc_dir, l90c_fn_template=l90c_fn_template
        )
    else:
        data = np.fromfile(l90c_fn, dtype=np.float32).reshape(ydim, xdim)
        logger.info(f"Read NT2 land90 mask from:\n  {l90c_fn}")

    return data
