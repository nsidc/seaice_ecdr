"""Routines for regridding 25km polar stereo files to 12.5km

"""

import datetime as dt

import numpy as np
import numpy.typing as npt
import xarray as xr

# NOTE: Use of opencv -- in python "cv2" -- requires a binary library
# on the VM:
#   sudo apt install libgl1
# Also, the opencv-pytypes package is only available from pip3, not from mamba
#   hence the type-ignore here
from cv2 import INTER_LINEAR, resize  # type: ignore[import-not-found]
from loguru import logger
from pm_tb_data._types import Hemisphere
from scipy.interpolate import griddata
from scipy.signal import convolve2d

from seaice_ecdr._types import ECDR_SUPPORTED_RESOLUTIONS
from seaice_ecdr.ancillary import (
    get_adj123_field,
    get_land_mask,
    get_ocean_mask,
)
from seaice_ecdr.grid_id import get_grid_id
from seaice_ecdr.gridid_to_xr_dataarray import get_dataset_for_grid_id

EXPECTED_TB_NAMES = ("h18", "v18", "v23", "h36", "v36")


# TODO: this is very similar to the `_setup_ecdr_ds`. DRY out? is any of this
# necessary, or can we just copy the attrs from the `initial_ecdr_ds`?
def _setup_ecdr_ds_replacement(
    *,
    date: dt.date,
    xr_tbs: xr.Dataset,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
) -> xr.Dataset:
    # Initialize geo-referenced xarray Dataset
    grid_id = get_grid_id(
        hemisphere=hemisphere,
        resolution=resolution,
    )

    # TODO: These fields should derive from the ancillary file,
    #       not get_dataset_for_grid_id()
    ecdr_ide_ds = get_dataset_for_grid_id(grid_id, date)

    # Set initial global attributes

    # Note: these attributes should probably go with
    #       a variable named "CDR_parameters" or similar
    ecdr_ide_ds.attrs["grid_id"] = grid_id

    file_date = dt.date(1970, 1, 1) + dt.timedelta(
        days=int(ecdr_ide_ds.variables["time"].data)
    )
    ecdr_ide_ds.attrs["time_coverage_start"] = str(
        dt.datetime(file_date.year, file_date.month, file_date.day, 0, 0, 0)
    )
    ecdr_ide_ds.attrs["time_coverage_end"] = str(
        dt.datetime(file_date.year, file_date.month, file_date.day, 23, 59, 59)
    )

    # Move variables to new ecdr_ds
    for key in xr_tbs.data_vars.keys():
        ecdr_ide_ds[key] = xr_tbs[key]

    return ecdr_ide_ds


def get_reprojection_da_psn25to12(
    hemisphere: Hemisphere,
) -> xr.DataArray:
    """Returns a mask with:
    0: No values will be interpolated
    1: bilinear interpolation via resize() should be applied
    2: block interpolation should be applied
    3: nearest neighbor interpolation should be applied
    """
    is_ocean_25 = get_ocean_mask(hemisphere=hemisphere, resolution="25")
    is_ocean_12 = get_ocean_mask(hemisphere=hemisphere, resolution="12.5")

    # Calculate where bilinear interpolation will compute properly
    resize25 = np.zeros(is_ocean_25.shape, dtype=np.float32)
    resize25[:] = 100
    resize25[~is_ocean_25] = np.nan
    ydim12, xdim12 = is_ocean_12.shape
    resize12 = resize(resize25, (xdim12, ydim12), interpolation=INTER_LINEAR)
    can_use_resize = ~np.isnan(resize12) & is_ocean_12

    # Calculate blocked interpolation locations
    blocked12 = np.zeros(is_ocean_12.shape, dtype=is_ocean_12.dtype)
    for joff in range(2):
        for ioff in range(2):
            blocked12[joff::2, ioff::2] = is_ocean_25[:]

    can_use_blocked = blocked12 & is_ocean_12 & ~can_use_resize

    # Determine remaining grid cells that will need to be nearest-neighbor'ed
    needs_nearneighbor = is_ocean_12 & ~can_use_resize & ~can_use_blocked

    reprojection_field = np.zeros(is_ocean_12.shape, dtype=np.uint8)
    reprojection_field[needs_nearneighbor] = 3
    reprojection_field[can_use_blocked] = 2
    reprojection_field[can_use_resize] = 1

    reprojection_da = xr.DataArray(
        reprojection_field,
        dims=("y", "x"),
        attrs=dict(
            flag_values=np.array((0, 1, 2, 3), dtype=np.uint8),
            flag_meanings="no_interp resize_interp block_interp nearest_interp",
        ),
    )

    return reprojection_da


def regrid_da_25to12_bilinear(
    da25: xr.DataArray,
) -> xr.DataArray:
    """Regrid this DataArray to twice the resolution.

    This will work with 2D or 3D arrays
    """
    # Work with 2D numpy fields
    data25 = da25.to_numpy()
    orig_shape = data25.shape

    is_3d = len(orig_shape) == 3
    if is_3d:
        data25 = np.squeeze(data25)
    ydim25, xdim25 = data25.shape

    # Calculate linearly interpolated field
    xdim12 = xdim25 * 2
    ydim12 = ydim25 * 2
    resize12 = resize(
        data25,
        (xdim12, ydim12),
        interpolation=INTER_LINEAR,
    )

    data12 = resize12

    # Make 3D if original array was 3D
    if len(orig_shape) == 3:
        data12 = np.expand_dims(data12, axis=0)

    # Re-assign as DataArray
    da12 = xr.DataArray(
        data=data12,
        dims=da25.dims,
        attrs=da25.attrs,
    )

    return da12


def regrid_da_25to12(
    da25: xr.DataArray,
    hemisphere: Hemisphere,
    reprojection_da: xr.DataArray,
    default_value: None | int | float = None,
    prefer_block: bool = False,
) -> xr.DataArray:
    """Regrid this DataArray to twice the resolution.

    This will work with 2D or 3D arrays
    """
    # Work with 2D numpy fields
    data25 = da25.to_numpy()
    orig_shape = data25.shape

    is_3d = len(orig_shape) == 3
    if is_3d:
        data25 = np.squeeze(data25)
    ydim25, xdim25 = data25.shape

    # Calculate block-interpolated field
    xdim12 = xdim25 * 2
    ydim12 = ydim25 * 2
    blocked12 = np.zeros((ydim12, xdim12), dtype=data25.dtype)
    for joff in range(2):
        for ioff in range(2):
            blocked12[joff::2, ioff::2] = data25[:]

    # Calculate linearly interpolated field
    if prefer_block:
        resize12 = blocked12.copy()
    else:
        resize12 = resize(
            data25,
            (xdim12, ydim12),
            interpolation=INTER_LINEAR,
        )

    # Begin building up the 12.5km resolution grid
    data12 = resize12.copy()
    if default_value is not None:
        data12[:] = default_value

    reprojection_mask = reprojection_da.data
    if prefer_block:
        data12[reprojection_mask == 1] = blocked12[reprojection_mask == 1]
    else:
        data12[reprojection_mask == 1] = resize12[reprojection_mask == 1]

    data12[reprojection_mask == 2] = blocked12[reprojection_mask == 2]

    # Calculate where linear extrapolation is needed
    yi, xi = np.where(reprojection_mask == 3)
    valid_indexes = np.where((reprojection_mask == 1) | (reprojection_mask == 2))
    valid_values = data12[valid_indexes]
    replacement12 = griddata(
        valid_indexes,
        valid_values,
        (yi, xi),
        # Note: any method other than 'nearest' here takes too long
        method="nearest",
    )
    data12[yi, xi] = replacement12

    # Make 3D if original array was 3D
    if len(orig_shape) == 3:
        data12 = np.expand_dims(data12, axis=0)

    # Re-assign as DataArray
    da12 = xr.DataArray(
        data=data12,
        dims=da25.dims,
        attrs=da25.attrs,
    )

    return da12


def adjust_reprojected_siconc_field(
    siconc_da: xr.DataArray,
    hemisphere: Hemisphere,
    adjustment_array: npt.NDArray,
    min_siext: float = 0.10,
    n_adj: int = 3,
) -> xr.DataArray:
    """Adjust this 12.5km concentration field that was reprojected from 25km."""
    # Work with 2D numpy fields
    siconc = np.squeeze(siconc_da.to_numpy())
    coast_adj = get_adj123_field(
        hemisphere=hemisphere,
        resolution="12.5",
    ).to_numpy()

    # Initialize ice_edge_adjacency to 255
    ice_edge_adjacency = np.zeros(siconc.shape, dtype=np.uint8)
    ice_edge_adjacency[:] = 255

    is_land = coast_adj == 0
    is_ocean = ~is_land

    # Set open ocean to 0
    is_open_ocean = is_ocean & (siconc < min_siext)
    ice_edge_adjacency[is_open_ocean] = 0

    kernel = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    for dist in range(1, n_adj + 1):
        is_labeled = (ice_edge_adjacency < dist) & ~is_land
        convolved = convolve2d(
            is_labeled,
            kernel,
            mode="same",
            boundary="fill",
            fillvalue=0,
        )
        is_newly_labeled = (convolved > 0) & (ice_edge_adjacency == 255) & ~is_land
        ice_edge_adjacency[is_newly_labeled] = dist

    ice_edge_adjacency[is_land] = 200

    for dist in range(1, n_adj + 1):
        idx = dist - 1
        siconc[ice_edge_adjacency == idx] = (
            siconc[ice_edge_adjacency == idx] * adjustment_array[idx]
        )

    siconc = np.expand_dims(siconc, axis=0)
    replacement_da = xr.DataArray(
        siconc,
        dims=siconc_da.dims,
        attrs=siconc_da.attrs,
    )

    return replacement_da


def reproject_ideds_25to12(
    initial_ecdr_ds,
    date,
    hemisphere,
    tb_resolution,
    resolution,
):
    # Determine reprojection_masks
    reprojection_da = get_reprojection_da_psn25to12(hemisphere=hemisphere)

    # Re-project any TBs
    # Note: TBs are reinterpreted as continuous fields
    reprojected_tbs = {}
    for key in initial_ecdr_ds.data_vars.keys():
        for expected_tb_name in EXPECTED_TB_NAMES:
            if expected_tb_name in key:
                tb_dataarray = initial_ecdr_ds.variables[key]
                tbda_12 = regrid_da_25to12_bilinear(
                    da25=tb_dataarray,
                )
                reprojected_tbs[key] = tbda_12

    logger.info(f"Regridded TB fields: {reprojected_tbs.keys()}")

    reprojected_tbs_ds = xr.Dataset(reprojected_tbs)

    # Initialize a new geoaware Dataset with the TBs
    reprojected_ideds = _setup_ecdr_ds_replacement(
        date=date,
        xr_tbs=reprojected_tbs_ds,
        resolution=resolution,
        hemisphere=hemisphere,
    )
    # add data_source and platform to the dataset attrs.
    reprojected_ideds.attrs["data_source"] = initial_ecdr_ds.data_source
    reprojected_ideds.attrs["platform"] = initial_ecdr_ds.platform

    # Pull from ancillary file
    reprojected_ideds["land_mask"] = get_land_mask(
        hemisphere=hemisphere,
        resolution=resolution,
    )

    # Block-replace (=nearest-neighbor interp)
    #   then extrapolate to no-value ocean grid cells
    block_regrid_vars = (
        "spatial_interpolation_flag",
        "invalid_ice_mask",
        "pole_mask",
        "invalid_tb_mask",
        "bt_weather_mask",
        "nt_weather_mask",
    )
    for var_name in block_regrid_vars:
        reprojected_ideds[var_name] = regrid_da_25to12(
            da25=initial_ecdr_ds[var_name],
            hemisphere=hemisphere,
            reprojection_da=reprojection_da,
            prefer_block=True,
        )

    # Bilinearly interpolate
    #   then extrapolate to no-value ocean grid cells
    # ecdr_ide_ds["raw_bt_seaice_conc"]
    bilinear_regrid_vars = (
        "raw_bt_seaice_conc",
        "raw_nt_seaice_conc",
        "conc",
        "qa_of_cdr_seaice_conc",
    )

    for var_name in bilinear_regrid_vars:
        reprojected_ideds[var_name] = regrid_da_25to12(
            da25=initial_ecdr_ds[var_name],
            hemisphere=hemisphere,
            reprojection_da=reprojection_da,
        )

    # Adjust the 12.5km cdr_seaice_conc field because it was derived from 25km
    reprojected_ideds["conc"] = adjust_reprojected_siconc_field(
        reprojected_ideds["conc"],
        hemisphere=hemisphere,
        adjustment_array=np.array((0.70, 0.89, 0.97)),
    )

    return reprojected_ideds
