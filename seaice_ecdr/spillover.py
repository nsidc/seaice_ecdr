from pathlib import Path
from typing import Literal

import numpy as np
import numpy.typing as npt
from pm_icecon.bt.compute_bt_ic import coastal_fix
from pm_icecon.land_spillover import apply_nt2_land_spillover
from pm_icecon.nt.compute_nt_ic import apply_nt_spillover
from pm_tb_data._types import Hemisphere
from scipy.ndimage import binary_dilation, generate_binary_structure, shift

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
LAND_SPILL_ALGS = Literal["BT_NT", "NT2", "ILS", "ILSb", "BT_NT2", "NT2_BT"]


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


def improved_land_spillover(
    *,
    ils_arr: npt.NDArray,
    init_conc: npt.NDArray,
    sie_min: float = 0.15,
) -> npt.NDArray:
    """Improved land spillover algorithm dilates "anchor"ing siconc far
    far-from-coast to coast and sets siconc values to zero if the dilated
    siconc is less than a threshold [sie_min]

    The ils_arr is encoded as:
        0: missing information: siconc cannot be used to anchor/disanchore
        1: non-ocean (ie land or other grid cells without siconc (~lake))
        2: ocean that may be removed by land spillover, if dilated sie low
        3: ocean whose concentration may anchor/disanchor coastal siconc
        4: ocean whose concentration is ignored for the ILS calcs

    Note: Because the algorithm removes near-coast siconc with nearby
          low siconc, grid cells which could be used to disanchor siconc
          but whose siconc is unknown are treated as "100% siconc" for
          purposes of this calculation.  In other words, missing (ils_arr==0)
          values will be set to max siconc (1.0) before calculations are made

    The input conc and output conc arrays are expected to have:
        min value of 0.0
        sie_min between 0.0 and 1.0
        max_considered siconc of 1.0
    """
    # Sanity check: ils_arr only contains the expected values
    ils_set = {val for val in np.unique(ils_arr)}
    assert ils_set.issubset(set(np.arange(5, dtype=np.uint8)))

    # Actual ILS algorithm begins here.
    conc = init_conc.copy()
    conc[ils_arr == 2] = -1.0  # potential ice removal grid cells to -1
    conc[conc > 1.0] = 1.0  # Clamp siconc to 1.0 max

    # Set any missing values to 100% siconc
    conc[ils_arr == 0] = 1.0

    # Remove vals from the fill_arr
    fill_arr = conc.copy()
    fill_arr[ils_arr == 1] = np.nan  # Init to nan where land
    fill_arr[ils_arr == 2] = -1  # Init to -1 where we need to fill
    # fill_arr at ils_arr will be conc value
    fill_arr[ils_arr == 4] = np.nan  # Init to where won't use value

    # Start by dilating the far-from coast ice into coastal areas
    is_fillable = (ils_arr == 2) | (ils_arr == 3)

    def _dilate_towards_shore(fill_arr, is_fillable):
        sum_arr = np.zeros(fill_arr.shape, dtype=np.float32)
        num_arr = np.zeros(fill_arr.shape, dtype=np.uint8)
        ones_arr = np.ones(fill_arr.shape, dtype=np.uint8)

        needs_filling = (ils_arr == 2) & (fill_arr == -1.0)
        n_needs_filling = np.sum(np.where(needs_filling, 1, 0))
        n_needs_filling_prior = n_needs_filling + 1  # Just needs to be different

        n_missing_values = np.sum(np.where((ils_arr == 2) & (fill_arr == -1.0), 1, 0))

        while (
            (n_needs_filling != n_needs_filling_prior)
            and (n_needs_filling > 0)
            and (n_missing_values > 0)
        ):
            needs_val = is_fillable & (fill_arr < 0)

            # Note: convolved2d can't handle the missing values,
            # so we need a shift-based approach
            left = shift(fill_arr, (0, -1), order=0, mode="constant", cval=np.nan)
            right = shift(fill_arr, (0, 1), order=0, mode="constant", cval=np.nan)
            upward = shift(fill_arr, (-1, 0), order=0, mode="constant", cval=np.nan)
            downward = shift(fill_arr, (1, 0), order=0, mode="constant", cval=np.nan)

            sum_arr[:] = 0.0
            num_arr[:] = 0

            addable = needs_val & (left >= 0)
            sum_arr[addable] += left[addable]
            num_arr[addable] += ones_arr[addable]

            addable = needs_val & (right >= 0)
            sum_arr[addable] += right[addable]
            num_arr[addable] += ones_arr[addable]

            addable = needs_val & (upward >= 0)
            sum_arr[addable] += upward[addable]
            num_arr[addable] += ones_arr[addable]

            addable = needs_val & (downward >= 0)
            sum_arr[addable] += downward[addable]
            num_arr[addable] += ones_arr[addable]

            has_new_values = num_arr > 0

            num_arr[num_arr == 0] = 1  # Prevent div by zero
            new_values = np.divide(sum_arr, num_arr).astype(np.float32)

            fill_arr[has_new_values] = new_values[has_new_values]

            n_needs_filling_prior = n_needs_filling

            n_missing_values = np.sum(
                np.where((ils_arr == 2) & (fill_arr == -1.0), 1, 0)
            )
            needs_filling = is_fillable & (fill_arr == -1.0)
            n_needs_filling = np.sum(np.where(needs_filling, 1, 0))

        return fill_arr

    fill_arr = _dilate_towards_shore(fill_arr, is_fillable)

    # If there are still spots that need filling, then the process is repeated,
    # but this time we allow dilation of conc values over land.
    needs_filling = (ils_arr == 2) & (fill_arr == -1.0)
    n_needs_filling = np.sum(np.where(needs_filling, 1, 0))
    if n_needs_filling > 0:
        print("LOG: Grid has isolated pockets of seaice; dilating over land")
        fill_arr[ils_arr == 1] = -1
        is_fillable[ils_arr == 1] = True
        fill_arr = _dilate_towards_shore(fill_arr, is_fillable)

    # Remove siconc where fill_arr is lower than min_sie
    filtered_conc = init_conc.copy()
    filtered_conc[(ils_arr == 2) & (fill_arr < sie_min)] = 0

    return filtered_conc


def land_spillover(
    *,
    cdr_conc: npt.NDArray,
    hemisphere: Hemisphere,
    tb_data: EcdrTbData,
    algorithm: LAND_SPILL_ALGS,
    land_mask: npt.NDArray,
    ancillary_source: ANCILLARY_SOURCES,
    bt_conc=None,  # only needed if the BT or NT spillover are used
    nt_conc=None,  # only needed if the BT or NT spillover are used
    bt_wx=None,  # only needed if the BT or NT spillover are used
    fix_goddard_bt_error: bool = False,  # By default, don't fix Goddard bug
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
            # anchoring_siconc=50.0,
            # affect_dist3=True,
            anchoring_siconc=0.0,
            affect_dist3=False,
        )
        spillover_applied = spillover_applied_nt2
    elif algorithm == "NT2_BT":
        # Apply the NT2 land spillover_algorithm to the cdr_conc_field
        #   and then apply the BT land spillover algorithm

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
            # anchoring_siconc=50.0,
            # affect_dist3=True,
            anchoring_siconc=0.0,
            affect_dist3=False,
        )

        # Apply the BT land spillover algorithm to the cdr_conc field
        non_ocean_mask = get_non_ocean_mask(
            hemisphere=hemisphere,
            resolution=tb_data.resolution,
            ancillary_source=ancillary_source,
        )

        spillover_applied_nt2_bt = coastal_fix(
            conc=spillover_applied_nt2,
            missing_flag_value=np.nan,
            land_mask=land_mask,
            minic=10,
            fix_goddard_bt_error=fix_goddard_bt_error,
        )

        spillover_applied = spillover_applied_nt2_bt

    elif algorithm == "BT_NT2":
        # Apply the BT land spillover algorithm to the cdr_conc field
        non_ocean_mask = get_non_ocean_mask(
            hemisphere=hemisphere,
            resolution=tb_data.resolution,
            ancillary_source=ancillary_source,
        )

        spillover_applied_bt = coastal_fix(
            # conc=bt_conc,
            conc=cdr_conc,
            missing_flag_value=np.nan,
            land_mask=land_mask,
            minic=10,
            fix_goddard_bt_error=fix_goddard_bt_error,
        )

        # Apply the NT2 land spillover_algorithm to the cdr_conc_field
        #   after the BT algorithm has been applied to it
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
        spillover_applied_bt_nt2 = apply_nt2_land_spillover(
            # conc=cdr_conc,
            conc=spillover_applied_bt,
            adj123=adj123.data,
            l90c=l90c.data,
            # anchoring_siconc=50.0,
            # affect_dist3=True,
            anchoring_siconc=0.0,
            affect_dist3=False,
        )
        spillover_applied = spillover_applied_bt_nt2

    elif algorithm == "BT_NT":
        non_ocean_mask = get_non_ocean_mask(
            hemisphere=hemisphere,
            resolution=tb_data.resolution,
            ancillary_source=ancillary_source,
        )

        # Bootstrap alg
        bt_conc[bt_wx] = 0
        bt_conc[bt_conc == 110] = np.nan
        spillover_applied_bt = coastal_fix(
            conc=bt_conc,
            missing_flag_value=np.nan,
            land_mask=land_mask,
            minic=10,
            fix_goddard_bt_error=fix_goddard_bt_error,
        )

        # NT alg
        # Apply the NT to the nt_conc field
        # Only apply that to the cdr_conc field if nt_spilled > bt_conc
        shoremap = get_nt_shoremap(
            hemisphere=hemisphere,
            resolution=tb_data.resolution,
            ancillary_source=ancillary_source,
        )

        minic = get_nt_minic(
            hemisphere=hemisphere,
            resolution=tb_data.resolution,
            ancillary_source=ancillary_source,
        )
        nt_conc_for_ntspillover = nt_conc.copy()
        is_negative_ntconc = nt_conc < 0
        nt_conc_for_ntspillover[is_negative_ntconc] = np.nan

        spillover_applied_nt = apply_nt_spillover(
            conc=nt_conc_for_ntspillover,
            shoremap=shoremap,
            minic=minic,
            match_CDRv4_cdralgos=True,
        )
        nt_asCDRv4 = (10.0 * spillover_applied_nt).astype(np.int16)
        nt_asCDRv4[is_negative_ntconc] = -10
        nt_asCDRv4[land_mask] = -100

        # This section is for debugging; outputs bt and bt after spillover
        # bt_asCDRv4 = spillover_applied_bt.copy()
        # bt_asCDRv4[np.isnan(bt_asCDRv4)] = 110
        # bt_asCDRv4.tofile('bt_afterspill_v5.dat')
        # nt_asCDRv4.tofile('nt_afterspill_v5.dat')

        # Note: be careful of ndarray vs xarray here!
        is_nt_spillover = (
            (spillover_applied_nt != nt_conc) & (~non_ocean_mask.data) & (nt_conc > 0)
        )
        use_nt_spillover = (spillover_applied_nt > bt_conc) & (spillover_applied_bt > 0)

        spillover_applied = spillover_applied_bt.copy()
        spillover_applied[use_nt_spillover] = spillover_applied_nt[use_nt_spillover]

        # Here, we remove ice that the NT algorithm removed -- with conc
        #   < 15% -- regardless of the BT conc value
        is_ntspill_removed = (
            is_nt_spillover
            & (spillover_applied_nt >= 0)
            & (spillover_applied_nt < 14.5)
        )
        spillover_applied[is_ntspill_removed.data] = 0

    elif algorithm == "ILS":
        # Prepare the ILS array using the adj123 field to init ils_arr
        ils_arr = np.zeros(cdr_conc.shape, dtype=np.uint8)
        adj123 = get_adj123_field(
            hemisphere=hemisphere,
            resolution=tb_data.resolution,
            ancillary_source=ancillary_source,
        )
        ils_arr[adj123 == 0] = 1  # land -> land
        ils_arr[adj123 == 1] = 2  # dist1 -> removable
        ils_arr[adj123 == 2] = 2  # dist2 -> removable
        ils_arr[adj123 == 3] = 3  # dist3 -> anchoring
        ils_arr[adj123 > 3] = 4  # >dist3 -> ignored

        ils_arr[np.isnan(cdr_conc)] = 0  # NaN conc -> "missing

        # Rescale conc from 0-100 to 0-1
        ils_conc = cdr_conc / 100.0
        ils_conc[ils_conc > 1.0] = 1.0

        spillover_applied_ils = improved_land_spillover(
            ils_arr=ils_arr,
            init_conc=ils_conc,
        )

        is_different = spillover_applied_ils != ils_conc
        spillover_applied = cdr_conc.copy()
        spillover_applied[is_different & ((adj123 == 1) | (adj123 == 2))] = 0

    elif algorithm == "ILSb":
        # Prepare the ILS array using shoremap to prep ils_arr
        ils_arr = np.zeros(cdr_conc.shape, dtype=np.uint8)
        shoremap = get_nt_shoremap(
            hemisphere=hemisphere,
            resolution=tb_data.resolution,
            ancillary_source=ancillary_source,
        )

        # Need to use adj123 to get rid of shoremap's lakes
        adj123 = get_adj123_field(
            hemisphere=hemisphere,
            resolution=tb_data.resolution,
            ancillary_source=ancillary_source,
        )
        shoremap[(shoremap == 2) & (adj123 == 0)] = 1  # lakeshore -> land
        shoremap[(shoremap == 3) & (adj123 == 0)] = 1  # lake -> land
        shoremap[(shoremap == 3) & (adj123 == 0)] = 1  # lake -> land
        shoremap[(shoremap == 4) & (adj123 == 0)] = 1  # laked2 -> land
        shoremap[(shoremap == 5) & (adj123 == 0)] = 1  # laked3 -> land
        shoremap[(shoremap == 0) & (adj123 == 0)] = 1  # laked+ -> land

        ils_arr[shoremap == 1] = 1  # land -> land
        ils_arr[shoremap == 2] = 1  # coast -> land
        ils_arr[shoremap == 3] = 2  # dist1 -> removable
        ils_arr[shoremap == 4] = 2  # dist2 -> removable
        ils_arr[shoremap == 5] = 3  # dist3 -> anchoring
        ils_arr[shoremap == 0] = 4  # >dist3 -> ignored

        ils_arr[np.isnan(cdr_conc)] = 0  # NaN conc -> "missing

        # Rescale conc from 0-100 to 0-1
        ils_conc = cdr_conc / 100.0
        ils_conc[ils_conc > 1.0] = 1.0

        spillover_applied_ils = improved_land_spillover(
            ils_arr=ils_arr,
            init_conc=ils_conc,
        )

        is_different = spillover_applied_ils != ils_conc
        spillover_applied = cdr_conc.copy()
        spillover_applied[is_different & ((shoremap == 3) | (shoremap == 4))] = 0

    else:
        raise RuntimeError(
            f"The requested land spillover algorithm ({algorithm=}) is not implemented."
        )

    return spillover_applied
