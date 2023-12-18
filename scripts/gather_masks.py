"""Gather/compile various masks/ancillary fields for ECDR into ancillary NetCDF files.

CDR v4 has ancillary files with the following variables:
* `crs`
* `landmask`
* `latitude`
* `longitude`
* `min_concentration`
* `month`
* `polehole` (NH only)
* `valid_ice_mask`
* `x`
* `y`
"""

import datetime as dt
from pathlib import Path

import numpy as np
import numpy.typing as npt
import xarray as xr
from loguru import logger
from pm_icecon.nt.params.amsr2 import get_amsr2_params
from pm_tb_data._types import NORTH, SOUTH, Hemisphere

from seaice_ecdr.constants import NSIDC_NFS_SHARE_DIR
from seaice_ecdr.grid_id import get_grid_id
from seaice_ecdr.masks import get_ancillary_filepath

# originally from `pm_icecon`
BOOTSTRAP_MASKS_DIR = NSIDC_NFS_SHARE_DIR / "bootstrap_masks"
CDR_TESTDATA_DIR = NSIDC_NFS_SHARE_DIR / "cdr_testdata"
BT_GODDARD_ANCILLARY_DIR = CDR_TESTDATA_DIR / "bt_goddard_ANCILLARY"

THIS_DIR = Path(__file__).resolve().parent


def get_surfacegeomask_filepath(grid_id: str) -> Path:
    filepath = THIS_DIR / "surface_geo_masks" / f"cdrv5_surfgeo_{grid_id}.nc"

    return filepath


def get_surfgeo_ds(*, hemisphere, resolution) -> xr.Dataset:
    """Return xr Dataset of ancillary surface/geolocation for this grid."""
    grid_id = get_grid_id(
        hemisphere=hemisphere,
        resolution=resolution,
    )
    return xr.load_dataset(get_surfacegeomask_filepath(grid_id))


def _get_ps25_grid_shape(*, hemisphere: Hemisphere) -> tuple[int, int]:
    """Get the polar stereo 25km resolution grid size."""
    shape = {
        "north": (448, 304),
        "south": (332, 316),
    }[hemisphere]

    return shape


def _get_pss_12_validice_land_coast_array(*, date: dt.date) -> npt.NDArray[np.int16]:
    """Get the polar stereo south 12.5km valid ice/land/coast array.

    4 unique values:
        * 0 == land
        * 4 == valid ice
        * 24 == invalid ice
        * 32 == coast.
    """
    fn = BOOTSTRAP_MASKS_DIR / f"bt_valid_pss12.5_int16_{date:%m}.dat"
    validice_land_coast = np.fromfile(fn, dtype=np.int16).reshape(664, 632)

    return validice_land_coast


def _get_ps_invalid_ice_mask(
    *,
    hemisphere: Hemisphere,
    date: dt.date,
    resolution: str,
) -> npt.NDArray[np.bool_]:
    """Read and return the polar stereo invalid ice mask.

    `True` values indicate areas that are masked as invalid.
    """
    logger.info(
        f"Reading valid ice mask for PS{hemisphere[0].upper()} {resolution}km grid"
    )  # noqa
    if hemisphere == "north":
        if resolution == "25":
            sst_fn = (
                BT_GODDARD_ANCILLARY_DIR / f"np_sect_sst1_sst2_mask_{date:%m}.int"
            ).resolve()
            sst_mask = np.fromfile(sst_fn, dtype=np.int16).reshape(
                _get_ps25_grid_shape(hemisphere=hemisphere)
            )
        elif resolution == "12":
            mask_fn = (
                CDR_TESTDATA_DIR
                / f"btequiv_psn12.5/bt_validmask_psn12.5km_{date:%m}.dat"
            )

            sst_mask = np.fromfile(mask_fn, dtype=np.int16).reshape(896, 608)
    else:
        if resolution == "12":
            # values of 24 indicate invalid ice.
            sst_mask = _get_pss_12_validice_land_coast_array(date=date)
        elif resolution == "25":
            sst_fn = Path(
                BT_GODDARD_ANCILLARY_DIR
                / f"SH_{date:%m}_SST_avhrr_threshold_{date:%m}_fixd.int"
            )
            sst_mask = np.fromfile(sst_fn, dtype=np.int16).reshape(
                _get_ps25_grid_shape(hemisphere=hemisphere)
            )

    is_high_sst = sst_mask == 24

    return is_high_sst


def ecdr_invalid_ice_masks_12km(
    *, hemisphere: Hemisphere, surfgeo_ds: xr.Dataset
) -> xr.DataArray:
    """Gather invalid ice mask fields returned by pm_icecon.bt.fields.get_bootstrap_fields for the 12.5km resolution.

    This function re-implements the logic from pm_icecon's
    `get_ps_invalid_ice_mask` for the 12.5km resolution so that it is more clear
    from this project's perspective the provenance of the ancillary files.

    Currently used by the initial_daily_ecdr module.

    Masks are available for both 12.5 and 25km resolutions. Currently the ECDR
    only supports 12.5km resolution.

    In the Northern Hemisphere, invalid ice masks are based on SST mask files
    originally found in
    `/share/apps/amsr2-cdr/cdr_testdata/bt_goddard_ANCILLARY/`.
    """
    invalid_icemasks = []
    months = range(1, 12 + 1)
    for month in months:
        invalid_icemask = _get_ps_invalid_ice_mask(
            hemisphere=hemisphere,
            # Only the month is used.
            date=dt.date(2023, month, 1),
            resolution="12",
        )
        invalid_icemasks.append(invalid_icemask)

    da = xr.DataArray(
        data=invalid_icemasks,
        coords=dict(
            month=months,
            y=surfgeo_ds.y,
            x=surfgeo_ds.x,
        ),
        attrs=dict(
            grid_mapping="crs",
            flag_meanings="valid_seaice_location invalid_seaice_location",
            flag_values=[np.byte(0), np.byte(1)],
            valid_range=(0, 1),
            units="1",
            comment=(
                "Mask indicating where seaice will not exist on this day"
                " based on climatology"
            ),
            long_name="invalid ice mask",
        ),
    )

    da.encoding = dict(zlib=True)

    return da


def ecdr_min_ice_concentration_12km(
    *, hemisphere: Hemisphere, surfgeo_ds: xr.Dataset
) -> xr.DataArray:
    """Used by the nasateam algorithm."""
    params = get_amsr2_params(
        hemisphere=hemisphere,
        resolution="12",
    )

    da = xr.DataArray(
        data=params.minic,
        coords=dict(
            y=surfgeo_ds.y,
            x=surfgeo_ds.x,
        ),
        attrs=dict(
            grid_mapping="crs",
        ),
    )

    da.encoding = dict(zlib=True)

    return da


if __name__ == "__main__":
    for hemisphere in (NORTH, SOUTH):
        surfgeo_ds = get_surfgeo_ds(
            hemisphere=hemisphere,
            resolution="12.5",
        )
        invalid_ice_masks = ecdr_invalid_ice_masks_12km(
            hemisphere=hemisphere, surfgeo_ds=surfgeo_ds
        )
        minimum_concentration = ecdr_min_ice_concentration_12km(
            hemisphere=hemisphere, surfgeo_ds=surfgeo_ds
        )

        data_vars = dict(
            crs=surfgeo_ds.crs,
            surface_type=surfgeo_ds.surface_type,
            latitude=surfgeo_ds.latitude,
            longitude=surfgeo_ds.longitude,
            min_concentration=minimum_concentration,
            month=invalid_ice_masks.month,
            invalid_ice_mask=invalid_ice_masks,
            x=surfgeo_ds.x,
            y=surfgeo_ds.y,
        )

        ancillary_ds = xr.Dataset(data_vars=data_vars)
        if hemisphere == NORTH:
            # TODO: should this have `flag_masks` instead of `flag_meanings`?
            ancillary_ds["polehole_bitmask"] = surfgeo_ds.polehole_bitmask

        filepath = get_ancillary_filepath(hemisphere=hemisphere, resolution="12.5")
        ancillary_ds.to_netcdf(filepath)

        logger.info(f"wrote {filepath}")
