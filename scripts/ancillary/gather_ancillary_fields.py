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
from pm_icecon.land_spillover import create_land90
from pm_icecon.nt.params.amsr2 import get_amsr2_params
from pm_tb_data._types import NORTH, SOUTH, Hemisphere

from seaice_ecdr.ancillary import get_ancillary_filepath
from seaice_ecdr.constants import NSIDC_NFS_SHARE_DIR
from seaice_ecdr.grid_id import get_grid_id

# originally from `pm_icecon`
BOOTSTRAP_MASKS_DIR = NSIDC_NFS_SHARE_DIR / "bootstrap_masks"
CDR_TESTDATA_DIR = NSIDC_NFS_SHARE_DIR / "cdr_testdata"
BT_GODDARD_ANCILLARY_DIR = CDR_TESTDATA_DIR / "bt_goddard_ANCILLARY"

NASATEAM2_ANCILLARY_DIR = NSIDC_NFS_SHARE_DIR / "nasateam2_ancillary"

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
    `/share/apps/G02202_V5/cdr_testdata/bt_goddard_ANCILLARY/`.
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


def read_adj123_file(
    grid_id: str = "psn12.5",
    xdim: int = 608,
    ydim: int = 896,
    anc_dir: Path = NASATEAM2_ANCILLARY_DIR,
    adj123_fn_template: str = "{anc_dir}/coastal_adj_diag123_{grid_id}.dat",
):
    """Read the diagonal adjacency 123 file."""
    coast_adj_fn = adj123_fn_template.format(anc_dir=anc_dir, grid_id=grid_id)
    assert Path(coast_adj_fn).is_file()
    adj123 = np.fromfile(coast_adj_fn, dtype=np.uint8).reshape(ydim, xdim)

    return adj123


def create_land90_conc_file(
    grid_id: str = "psn12.5",
    xdim: int = 608,
    ydim: int = 896,
    anc_dir: Path = NASATEAM2_ANCILLARY_DIR,
    adj123_fn_template: str = "{anc_dir}/coastal_adj_diag123_{grid_id}.dat",
    write_l90c_file: bool = True,
    l90c_fn_template: str = "{anc_dir}/land90_conc_{grid_id}.dat",
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
    adj123 = read_adj123_file(grid_id, xdim, ydim, anc_dir, adj123_fn_template)
    land90 = create_land90(adj123=adj123)

    if write_l90c_file:
        l90c_fn = l90c_fn_template.format(anc_dir=anc_dir, grid_id=grid_id)
        land90.tofile(l90c_fn)
        print(f"Wrote: {l90c_fn}\n  {land90.dtype}  {land90.shape}")

    return land90


def load_or_create_land90_conc(
    grid_id: str = "psn12.5",
    xdim: int = 608,
    ydim: int = 896,
    anc_dir: Path = NASATEAM2_ANCILLARY_DIR,
    l90c_fn_template: str = "{anc_dir}/land90_conc_{grid_id}.dat",
    overwrite: bool = False,
):
    # Attempt to load the land90_conc field, and if fail, create it
    l90c_fn = l90c_fn_template.format(anc_dir=anc_dir, grid_id=grid_id)
    if overwrite or not Path(l90c_fn).is_file():
        data = create_land90_conc_file(
            grid_id, xdim, ydim, anc_dir=anc_dir, l90c_fn_template=l90c_fn_template
        )
    else:
        data = np.fromfile(l90c_fn, dtype=np.float32).reshape(ydim, xdim)
        logger.info(f"Read NT2 land90 mask from:\n  {l90c_fn}")

    return data


if __name__ == "__main__":
    grid_id = "psn12.5"
    resolution = "12.5"
    for hemisphere in (NORTH, SOUTH):
        x_dim_size, y_dim_size = {
            "north": (608, 896),
            "south": (632, 664),
        }[hemisphere]
        surfgeo_ds = get_surfgeo_ds(
            hemisphere=hemisphere,
            resolution=resolution,
        )
        grid_id = get_grid_id(
            hemisphere=hemisphere,
            resolution=resolution,
        )
        invalid_ice_masks = ecdr_invalid_ice_masks_12km(
            hemisphere=hemisphere, surfgeo_ds=surfgeo_ds
        )
        minimum_concentration = ecdr_min_ice_concentration_12km(
            hemisphere=hemisphere, surfgeo_ds=surfgeo_ds
        )

        l90c = load_or_create_land90_conc(
            grid_id=grid_id,
            xdim=x_dim_size,
            ydim=y_dim_size,
            overwrite=False,
        )
        adj123 = read_adj123_file(
            grid_id=grid_id,
            xdim=x_dim_size,
            ydim=y_dim_size,
        )

        l90c_da = xr.DataArray(
            data=l90c.copy(),
            coords=dict(
                y=surfgeo_ds.y.data,
                x=surfgeo_ds.x.data,
            ),
            attrs=dict(
                grid_mapping="crs",
                comment=(
                    "The 'land90' array is a mock sea ice concentration array that is calculated"
                    "from the land mask.  It assumes that the mock concentration value will be"
                    "the average of a 7x7 array of local surface mask values centered on the"
                    "center pixel.  Water grid cells are considered to have a sea ice"
                    "concentration of zero.  Land grid cells are considered to have a sea ice"
                    "concentration of 90%.  The average of the 49 grid cells in the 7x7 array"
                    "yields the `land90` concentration value."
                ),
            ),
        )

        adj123_da = xr.DataArray(
            data=adj123.copy(),
            coords=dict(
                y=surfgeo_ds.y.data,
                x=surfgeo_ds.x.data,
            ),
            attrs=dict(
                grid_mapping="crs",
                comment="Diagonal adjacency 123 field",
            ),
        )

        data_vars = dict(
            crs=surfgeo_ds.crs,
            l90c=l90c_da,
            adj123=adj123_da,
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
            polehole_bitmask = surfgeo_ds.polehole_bitmask
            bitmask_attrs = polehole_bitmask.attrs
            bitmask_attrs["flag_masks"] = bitmask_attrs.pop("flag_values")
            polehole_bitmask.attrs = bitmask_attrs
            ancillary_ds["polehole_bitmask"] = polehole_bitmask

        filepath = get_ancillary_filepath(hemisphere=hemisphere, resolution="12.5")
        ancillary_ds.compute()
        ### SS: Temporarily change name of filepath
        breakpoint()
        ancillary_ds.to_netcdf(filepath)

        logger.info(f"wrote {filepath}")
