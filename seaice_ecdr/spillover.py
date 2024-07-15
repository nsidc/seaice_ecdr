from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
from pm_icecon.bt.compute_bt_ic import coastal_fix
from pm_icecon.land_spillover import apply_nt2_land_spillover
from pm_icecon.nt.compute_nt_ic import apply_nt_spillover
from pm_tb_data._types import Hemisphere
from scipy.ndimage import binary_dilation, generate_binary_structure

from seaice_ecdr._types import SUPPORTED_SAT
from seaice_ecdr.ancillary import (
    ANCILLARY_SOURCES,
    get_adj123_field,
    get_land90_conc_field,
    get_non_ocean_mask,
    get_nt_minic,
    get_nt_shoremap,
)
from seaice_ecdr.tb_data import (
    EcdrTbData,
)
from seaice_ecdr.util import get_ecdr_grid_shape

NT_MAPS_DIR = Path("/share/apps/G02202_V5/cdr_testdata/nt_datafiles/data36/maps")
LAND_SPILL_ALGS = Literal["BT_NT", "NT2"]


def convert_nonocean_to_shoremap(*, is_nonocean: npt.NDArray):
    """The shoremap is the an array that the original NT spillover algorithm
    used to determine distance-from-shore.  Then encoding is:
      0: Ocean far from coast
      1: land: land not adjacent to ocean
      2: shore: land 4-connected to ocean
      3: coast: ocean 8-connected to land
      4: near-coast:  ocean 4-connected to "coast"
      5: far-coast:  ocean 4-connected to "near-coast"

    All of this can be computed from the non-ocean mask

    Maddenly, there are logical inconsistencies in the CDRv4 fields that
    make the calculations of coast slightly off
    """
    conn4 = generate_binary_structure(2, 1)
    conn8 = generate_binary_structure(2, 2)

    # Initialize to zero
    shoremap = np.zeros(is_nonocean.shape, dtype=np.uint8)

    # Add the land

    # Add the shore (land adjacent to ocean)
    is_ocean = ~is_nonocean
    is_dilated_ocean = binary_dilation(is_ocean, structure=conn4)
    is_shore = is_nonocean & is_dilated_ocean
    is_land = is_nonocean & ~is_shore
    shoremap[is_land] = 1
    shoremap[is_shore] = 2

    # Add the coast
    is_dilated_nonocean = binary_dilation(is_nonocean, structure=conn8)
    is_coast = is_dilated_nonocean & is_ocean
    shoremap[is_coast] = 3

    # Add the nearcoast
    is_dilated_coast = binary_dilation(is_coast, structure=conn4)
    is_nearcoast = is_dilated_coast & is_ocean & ~is_coast
    shoremap[is_nearcoast] = 4

    # Add the farcoast
    is_dilated_nearcoast = binary_dilation(is_nearcoast, structure=conn4)
    is_farcoast = is_dilated_nearcoast & is_ocean & ~is_coast & ~is_nearcoast
    shoremap[is_farcoast] = 5


def _get_25km_shoremap(*, hemisphere: Hemisphere):
    shoremap_fn = NT_MAPS_DIR / f"shoremap_{hemisphere}_25"
    shoremap = np.fromfile(shoremap_fn, dtype=">i2")[150:].reshape(
        get_ecdr_grid_shape(hemisphere=hemisphere, resolution="25")
    )

    return shoremap


def _get_25km_minic(*, hemisphere: Hemisphere):
    if hemisphere == "north":
        minic_fn = "SSMI8_monavg_min_con"
    else:
        minic_fn = "SSMI_monavg_min_con_s"

    minic_path = NT_MAPS_DIR / minic_fn
    minic = np.fromfile(minic_path, dtype=">i2")[150:].reshape(
        get_ecdr_grid_shape(hemisphere=hemisphere, resolution="25")
    )

    # Scale down by 10. The original alg. dealt w/ concentrations scaled by 10.
    minic = minic / 10

    return minic


def land_spillover(
    *,
    cdr_conc: npt.NDArray,
    hemisphere: Hemisphere,
    tb_data: EcdrTbData,
    algorithm: LAND_SPILL_ALGS,
    land_mask: npt.NDArray,
    platform: SUPPORTED_SAT,
    ancillary_source: ANCILLARY_SOURCES,
    bt_conc=None,  # only needed if the BT or NT spillover are used
    nt_conc=None,  # only needed if the BT or NT spillover are used
) -> npt.NDArray:
    """Apply the land spillover technique to the CDR concentration field."""

    # SS: Writing out the spillover anc fields...
    if algorithm == "NT2":
        l90c = get_land90_conc_field(
            hemisphere=hemisphere,
            resolution=tb_data.resolution,
            ancillary_source=ancillary_source,
        )
        adj123 = get_adj123_field(
            hemisphere=hemisphere,
            resolution=tb_data.resolution,
            ancillary_source=ancillary_source,
        )
        spillover_applied_nt2 = apply_nt2_land_spillover(
            conc=cdr_conc,
            adj123=adj123.data,
            l90c=l90c.data,
            anchoring_siconc=50.0,
            affect_dist3=True,
        )
        spillover_applied = spillover_applied_nt2
    elif algorithm == "BT_NT":
        non_ocean_mask = get_non_ocean_mask(
            hemisphere=hemisphere,
            resolution=tb_data.resolution,
            ancillary_source=ancillary_source,
        )

        # Bootstrap alg
        spillover_applied_bt = coastal_fix(
            conc=cdr_conc,
            missing_flag_value=np.nan,
            land_mask=land_mask,
            minic=10,
        )
        is_bt_spillover = (spillover_applied_bt != cdr_conc) & (~non_ocean_mask)

        # NT alg
        # Apply the NT to the nt_conc field
        # Only apply that to the cdr_conc field if nt_spilled > bt_conc
        # shoremap = _get_25km_shoremap(hemisphere=hemisphere)
        shoremap = get_nt_shoremap(
            hemisphere=hemisphere,
            resolution=tb_data.resolution,
            ancillary_source=ancillary_source,
        )

        # minic = _get_25km_minic(hemisphere=hemisphere)
        minic = get_nt_minic(
            hemisphere=hemisphere,
            resolution=tb_data.resolution,
            ancillary_source=ancillary_source,
        )
        spillover_applied_nt = apply_nt_spillover(
            conc=nt_conc,
            shoremap=shoremap,
            minic=minic,
        )
        is_nt_spillover = (spillover_applied_nt != nt_conc) & (~non_ocean_mask)
        use_nt_spillover = (
            is_nt_spillover & (spillover_applied_nt > bt_conc) & (~is_bt_spillover)
        )

        spillover_applied = spillover_applied_bt
        spillover_applied[use_nt_spillover] = spillover_applied_nt[use_nt_spillover]

    else:
        raise RuntimeError(
            f"The requested land spillover algorithm ({algorithm=}) is not implemented."
        )

    return spillover_applied
