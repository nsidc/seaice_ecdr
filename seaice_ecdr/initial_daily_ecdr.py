"""Create the initial daily ECDR file.

Notes:

* `idecdr` is shorthand for == "Initial Daily ECDR"
"""

import datetime as dt
import sys
import traceback
from functools import cache
from pathlib import Path
from typing import Iterable, TypedDict, cast, get_args

import click
import numpy as np
import numpy.typing as npt
import pm_icecon.bt.compute_bt_ic as bt
import pm_icecon.bt.params.ausi_amsr2 as pmi_bt_params_amsr2
import pm_icecon.bt.params.ausi_amsre as pmi_bt_params_amsre
import pm_icecon.bt.params.nsidc0001 as pmi_bt_params_0001
import pm_icecon.nt.compute_nt_ic as nt
import xarray as xr
from loguru import logger
from pm_icecon.bt.params.nsidc0007 import get_smmr_params
from pm_icecon.bt.xfer_tbs import xfer_rss_tbs
from pm_icecon.errors import UnexpectedSatelliteError
from pm_icecon.fill_polehole import fill_pole_hole
from pm_icecon.interpolation import spatial_interp_tbs
from pm_icecon.land_spillover import apply_nt2_land_spillover
from pm_icecon.nt._types import NasateamGradientRatioThresholds
from pm_icecon.nt.params import get_cdr_nt_params
from pm_icecon.nt.tiepoints import NasateamTiePoints
from pm_tb_data._types import NORTH, Hemisphere

from seaice_ecdr._types import ECDR_SUPPORTED_RESOLUTIONS, SUPPORTED_SAT
from seaice_ecdr.ancillary import (
    get_adj123_field,
    get_empty_ds_with_time,
    get_invalid_ice_mask,
    get_land90_conc_field,
    get_land_mask,
    nh_polehole_mask,
)
from seaice_ecdr.cli.util import datetime_to_date
from seaice_ecdr.constants import STANDARD_BASE_OUTPUT_DIR
from seaice_ecdr.grid_id import get_grid_id
from seaice_ecdr.platforms import get_platform_by_date
from seaice_ecdr.regrid_25to12 import reproject_ideds_25to12
from seaice_ecdr.tb_data import EXPECTED_ECDR_TB_NAMES, EcdrTbData, get_ecdr_tb_data
from seaice_ecdr.util import date_range, standard_daily_filename


def cdr_bootstrap(
    *,
    tb_v37: npt.NDArray,
    tb_h37: npt.NDArray,
    tb_v19: npt.NDArray,
    bt_coefs,
    platform: SUPPORTED_SAT,
):
    """Generate the raw bootstrap concentration field."""
    wtp_37v = bt_coefs["bt_wtp_v37"]
    wtp_37h = bt_coefs["bt_wtp_h37"]
    wtp_19v = bt_coefs["bt_wtp_v19"]

    itp_37v = bt_coefs["bt_itp_v37"]
    itp_37h = bt_coefs["bt_itp_h37"]
    itp_19v = bt_coefs["bt_itp_v19"]

    # Transform TBs for the bootstrap calculation
    try:
        transformed = xfer_rss_tbs(
            tbs=dict(
                v37=tb_v37,
                h37=tb_h37,
                v19=tb_v19,
            ),
            platform=platform.lower(),
        )
        tb_v37 = transformed["v37"]
        tb_h37 = transformed["h37"]
        tb_v19 = transformed["v19"]
    except UnexpectedSatelliteError:
        logger.warning(
            f"No BT Tb transformation to F13 available for {platform=}. Using untransformed Tbs instead."
        )

    bt_conc = bt.calc_bootstrap_conc(
        tb_v37=tb_v37,
        tb_h37=tb_h37,
        tb_v19=tb_v19,
        wtp_37v=wtp_37v,
        wtp_37h=wtp_37h,
        wtp_19v=wtp_19v,
        itp_37v=itp_37v,
        itp_37h=itp_37h,
        itp_19v=itp_19v,
        line_37v37h=bt_coefs["vh37_lnline"],
        line_37v19v=bt_coefs["v1937_lnline"],
        ad_line_offset=bt_coefs["ad_line_offset"],
        maxic_frac=bt_coefs["maxic"],
        # Note: the missing value of 255 ends up getting set to `nan` below.
        missing_flag_value=255,
    )

    # Se any bootstrap concentrations below 10% to 0.
    bt_conc[bt_conc < 10] = 0

    # Remove bt_conc flags (e.g., missing)
    bt_conc[bt_conc > 200] = np.nan

    return bt_conc


def cdr_nasateam(
    *,
    tb_h19: npt.NDArray,
    tb_v37: npt.NDArray,
    tb_v19: npt.NDArray,
    nt_tiepoints: NasateamTiePoints,
) -> npt.NDArray:
    """Generate the raw NASA Team concentration field.

    Note that concentrations from nasateam may be >100%
    """
    nt_pr_1919 = nt.compute_ratio(tb_v19, tb_h19)
    nt_gr_3719 = nt.compute_ratio(tb_v37, tb_v19)
    nt_conc = nt.calc_nasateam_conc(
        pr_1919=nt_pr_1919,
        gr_3719=nt_gr_3719,
        tiepoints=nt_tiepoints,
    )

    return nt_conc


def get_bt_tb_mask(
    *,
    tb_v37,
    tb_h37,
    tb_v19,
    tb_v22,
    mintb,
    maxtb,
    tb_data_mask_function,
):
    """Determine TB mask per Bootstrap algorithm's criteria."""
    bt_tb_mask = tb_data_mask_function(
        tbs=(
            tb_v37,
            tb_h37,
            tb_v19,
            tb_v22,
        ),
        min_tb=mintb,
        max_tb=maxtb,
    )

    try:
        assert tb_v37.shape == tb_h37.shape
        assert tb_v37.shape == tb_v22.shape
        assert tb_v37.shape == tb_v19.shape
        assert tb_v37.shape == bt_tb_mask.shape
    except AssertionError as e:
        raise ValueError(f"Mismatched shape error in get_bt_tb_mask\n{e}")

    return bt_tb_mask


class NtCoefs(TypedDict):
    nt_tiepoints: NasateamTiePoints
    nt_gradient_thresholds: NasateamGradientRatioThresholds


def calculate_bt_nt_cdr_raw_conc(
    *,
    tb_h19: npt.NDArray,
    tb_v37: npt.NDArray,
    tb_h37: npt.NDArray,
    tb_v19: npt.NDArray,
    bt_coefs: dict,
    nt_coefs: NtCoefs,
    platform: SUPPORTED_SAT,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Run the CDR algorithm."""
    # First, get bootstrap conc.
    bt_conc = cdr_bootstrap(
        tb_v37=tb_v37,
        tb_h37=tb_h37,
        tb_v19=tb_v19,
        bt_coefs=bt_coefs,
        platform=platform,
    )

    # Get nasateam conc. Note that concentrations from nasateam may be >100%.
    nt_conc = cdr_nasateam(
        tb_h19=tb_h19,
        tb_v37=tb_v37,
        tb_v19=tb_v19,
        nt_tiepoints=nt_coefs["nt_tiepoints"],
    )

    # Now calculate CDR SIC
    is_bt_seaice = (bt_conc > 0) & (bt_conc <= 100)
    use_nt_values = (nt_conc > bt_conc) & is_bt_seaice
    # Note: Here, values without sea ice (because no TBs) have val np.nan
    cdr_conc = bt_conc.copy()
    cdr_conc[use_nt_values] = nt_conc[use_nt_values]

    return bt_conc, nt_conc, cdr_conc


def _setup_ecdr_ds(
    *,
    date: dt.date,
    tb_data: EcdrTbData,
    hemisphere: Hemisphere,
) -> xr.Dataset:
    # Initialize geo-referenced xarray Dataset
    grid_id = get_grid_id(
        hemisphere=hemisphere,
        resolution=tb_data.resolution,
    )

    ecdr_ide_ds = get_empty_ds_with_time(
        hemisphere=hemisphere, resolution=tb_data.resolution, date=date
    )

    # Set initial global attributes

    # Note: these attributes should probably go with
    #       a variable named "CDR_parameters" or similar
    ecdr_ide_ds.attrs["grid_id"] = grid_id

    # Set data_source attribute
    ecdr_ide_ds.attrs["data_source"] = tb_data.data_source

    # Set the platform
    ecdr_ide_ds.attrs["platform"] = tb_data.platform

    file_date = dt.date(1970, 1, 1) + dt.timedelta(
        days=int(ecdr_ide_ds.variables["time"].data)
    )
    ecdr_ide_ds.attrs["time_coverage_start"] = str(
        dt.datetime(file_date.year, file_date.month, file_date.day, 0, 0, 0)
    )
    ecdr_ide_ds.attrs["time_coverage_end"] = str(
        dt.datetime(file_date.year, file_date.month, file_date.day, 23, 59, 59)
    )

    # Move TBs to ecdr_ds
    for tbname in EXPECTED_ECDR_TB_NAMES:
        tb_varname = f"{tbname}_day"
        tbdata = getattr(tb_data.tbs, tbname)
        freq = tbname[1:]
        pol = tbname[:1]
        tb_longname = f"Daily TB {freq}{pol} from {tb_data.data_source}"
        tb_units = "K"
        ecdr_ide_ds[tb_varname] = (
            ("time", "y", "x"),
            np.expand_dims(tbdata, axis=0),
            {
                "grid_mapping": "crs",
                "standard_name": "brightness_temperature",
                "long_name": tb_longname,
                "units": tb_units,
                "valid_range": [np.float64(10.0), np.float64(350.0)],
            },
            {
                "zlib": True,
            },
        )

    return ecdr_ide_ds


def compute_initial_daily_ecdr_dataset(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    tb_data: EcdrTbData,
    fill_the_pole_hole: bool = False,
) -> xr.Dataset:
    """Create intermediate daily ECDR xarray dataset.

    Returns an xarray Dataset with CDR concentration and other fields required
    to produce a final, temporally interpolated CDR NetCDF file.

    TODO: more details/type definition/model for the various output data
    fields. Its difficult to understand what's in the resulting dataset without
    manually inspecting the result of running this code.
    """
    ecdr_ide_ds = _setup_ecdr_ds(
        date=date,
        tb_data=tb_data,
        hemisphere=hemisphere,
    )

    # Spatially interpolate the brightness temperatures
    for tbname in EXPECTED_ECDR_TB_NAMES:
        tb_day_name = f"{tbname}_day"
        tb_si_varname = f"{tb_day_name}_si"
        tb_si_data = spatial_interp_tbs(ecdr_ide_ds[tb_day_name].data[0, :, :])
        tb_si_longname = f"Spatially interpolated {ecdr_ide_ds[tb_day_name].long_name}"
        tb_units = "K"
        ecdr_ide_ds[tb_si_varname] = (
            ("time", "y", "x"),
            np.expand_dims(tb_si_data, axis=0),
            {
                "grid_mapping": "crs",
                "standard_name": "brightness_temperature",
                "long_name": tb_si_longname,
                "units": tb_units,
                "valid_range": [np.float64(10.0), np.float64(3500.0)],
            },
            {
                "zlib": True,
            },
        )

    # Set up the spatial interpolation bitmask field where the various
    # TB fields were interpolated
    _, ydim, xdim = ecdr_ide_ds["h19_day_si"].data.shape
    spatint_bitmask_arr = np.zeros((ydim, xdim), dtype=np.uint8)
    # TODO: Long term reminder: we want to use "band" labels rather than
    #       exact GHz frequencies -- which vary by satellite -- to ID channels
    TB_SPATINT_BITMASK_MAP = {
        "v19": 1,
        "h19": 2,
        "v22": 4,
        "v37": 8,
        "h37": 16,
        "pole_filled": 32,
    }
    for tbname in EXPECTED_ECDR_TB_NAMES:
        tb_varname = f"{tbname}_day"
        si_varname = f"{tbname}_day_si"
        is_tb_si_diff = (
            ecdr_ide_ds[tb_varname].data[0, :, :]
            != ecdr_ide_ds[si_varname].data[0, :, :]
        ) & (~np.isnan(ecdr_ide_ds[si_varname].data[0, :, :]))
        spatint_bitmask_arr[is_tb_si_diff] += TB_SPATINT_BITMASK_MAP[tbname]
        land_mask = get_land_mask(
            hemisphere=hemisphere,
            resolution=tb_data.resolution,
        )
        spatint_bitmask_arr[land_mask.data] = 0

    ecdr_ide_ds["spatial_interpolation_flag"] = (
        ("time", "y", "x"),
        np.expand_dims(spatint_bitmask_arr, axis=0),
        {
            "grid_mapping": "crs",
            "standard_name": "status_flag",
            "long_name": "spatial_interpolation_flag",
            "units": 1,
            "valid_range": [np.uint8(0), np.uint8(63)],
            "flag_masks": np.array([1, 2, 4, 8, 16, 32], dtype=np.uint8),
            "flag_meanings": (
                "19v_tb_value_interpolated"
                " 19h_tb_value_interpolated"
                " 22v_tb_value_interpolated"
                " 37v_tb_value_interpolated"
                " 37h_tb_value_interpolated"
                " Pole_hole_spatially_interpolated_(Arctic_only)"
            ),
        },
        {
            "zlib": True,
        },
    )
    logger.info("Initialized spatial_interpolation_flag with TB fill locations")

    platform = get_platform_by_date(date)
    if platform == "am2":
        bt_coefs_init = pmi_bt_params_amsr2.get_ausi_amsr2_bootstrap_params(
            date=date,
            satellite="amsr2",
            gridid=ecdr_ide_ds.grid_id,
        )
    elif platform == "ame":
        bt_coefs_init = pmi_bt_params_amsre.get_ausi_amsre_bootstrap_params(
            date=date,
            satellite="amsre",
            gridid=ecdr_ide_ds.grid_id,
        )
    elif platform == "F17":
        bt_coefs_init = pmi_bt_params_0001.get_F17_bootstrap_params(
            date=date,
            satellite=platform,
            gridid=ecdr_ide_ds.grid_id,
        )
    elif platform == "n07":
        bt_coefs_init = get_smmr_params(hemisphere=hemisphere, date=date)
    else:
        raise RuntimeError(f"platform bootstrap params not implemented: {platform}")

    # Add the function that generates the bt_tb_mask to bt_coefs
    # TODO: there's no reason to assign the function to this dictionary. It
    # makes tracking down how the mask is created difficult!
    bt_coefs_init["bt_tb_data_mask_function"] = bt.tb_data_mask

    # Get the bootstrap fields and assign them to ide_ds DataArrays
    invalid_ice_mask = get_invalid_ice_mask(
        hemisphere=hemisphere,
        date=date,
        resolution=tb_data.resolution,
        platform=platform,
    )

    land_mask = get_land_mask(
        hemisphere=hemisphere,
        resolution=tb_data.resolution,
    )

    ecdr_ide_ds["invalid_ice_mask"] = invalid_ice_mask.expand_dims(dim="time")

    # Encode land_mask
    ecdr_ide_ds["land_mask"] = land_mask

    # Encode pole_mask
    # TODO: I think this is currently unused
    # ...but it should be coordinated with pole hole filling routines below
    # ...and the pole filling should occur after temporal interpolation
    if hemisphere == NORTH:
        pole_mask = nh_polehole_mask(
            date=date,
            resolution=tb_data.resolution,
            sat=platform,
        )
        ecdr_ide_ds["pole_mask"] = pole_mask

    nt_params = get_cdr_nt_params(
        hemisphere=hemisphere,
        platform=platform,
    )

    nt_coefs = NtCoefs(
        nt_tiepoints=nt_params.tiepoints,
        nt_gradient_thresholds=nt_params.gradient_thresholds,
    )

    # Compute the invalid TB mask
    invalid_tb_mask = get_bt_tb_mask(
        tb_v37=ecdr_ide_ds["v37_day_si"].data[0, :, :],
        tb_h37=ecdr_ide_ds["h37_day_si"].data[0, :, :],
        tb_v19=ecdr_ide_ds["v19_day_si"].data[0, :, :],
        tb_v22=ecdr_ide_ds["v22_day_si"].data[0, :, :],
        mintb=bt_coefs_init["mintb"],
        maxtb=bt_coefs_init["maxtb"],
        tb_data_mask_function=bt_coefs_init["bt_tb_data_mask_function"],
    )

    ecdr_ide_ds["invalid_tb_mask"] = (
        ("time", "y", "x"),
        np.expand_dims(invalid_tb_mask, axis=0),
        {
            "grid_mapping": "crs",
            "standard_name": "invalid_tb_binary_mask",
            "long_name": "Map of Invalid TBs",
            "comment": "Mask indicating pixels with invalid TBs",
            "units": 1,
        },
        {
            "zlib": True,
        },
    )

    bt_v37 = ecdr_ide_ds["v37_day_si"].data[0, :, :]
    bt_h37 = ecdr_ide_ds["h37_day_si"].data[0, :, :]
    bt_v22 = ecdr_ide_ds["v22_day_si"].data[0, :, :]
    bt_v19 = ecdr_ide_ds["v19_day_si"].data[0, :, :]

    # Transform TBs for the bootstrap calculation
    # TODO: DRY out this code some...These transformations also occur for the
    # bootstrap SIC conc. We only want to transform Tbs for the bootstrap SIC
    # calc and the weather filtering.
    try:
        transformed = xfer_rss_tbs(
            tbs=dict(
                v37=bt_v37,
                h37=bt_h37,
                v19=bt_v19,
                v22=bt_v22,
            ),
            platform=platform.lower(),
        )
        bt_v37 = transformed["v37"]
        bt_h37 = transformed["h37"]
        bt_v19 = transformed["v19"]
        bt_v22 = transformed["v22"]
    except UnexpectedSatelliteError:
        logger.warning(
            f"No BT Tb transformation to F13 available for {platform=}. Using untransformed Tbs instead."
        )

    # Compute the BT weather mask
    bt_weather_mask = bt.get_weather_mask(
        v37=bt_v37,
        h37=bt_h37,
        v22=bt_v22,
        v19=bt_v19,
        land_mask=ecdr_ide_ds["land_mask"].data,
        tb_mask=ecdr_ide_ds["invalid_tb_mask"].data[0, :, :],
        ln1=bt_coefs_init["vh37_lnline"],
        date=date,
        wintrc=bt_coefs_init["wintrc"],
        wslope=bt_coefs_init["wslope"],
        wxlimt=bt_coefs_init["wxlimt"],
    )

    ecdr_ide_ds["bt_weather_mask"] = (
        ("time", "y", "x"),
        np.expand_dims(bt_weather_mask.data, axis=0),
        {
            "grid_mapping": "crs",
            "standard_name": "bt_weather_binary_mask",
            "long_name": "Map of weather masquerading as sea ice per BT",
            "comment": (
                "Mask indicating pixels with erroneously detected sea ice"
                " because of weather per BT "
            ),
            "units": 1,
        },
        {
            "zlib": True,
        },
    )

    # Update the Bootstrap coefficients...
    bt_coefs_to_update = (
        "line_37v37h",
        "bt_wtp_v37",
        "bt_wtp_h37",
        "bt_wtp_v19",
        "ad_line_offset",
        "line_37v19v",
    )

    bt_coefs = bt_coefs_init.copy()
    for coef in bt_coefs_to_update:
        bt_coefs.pop(coef, None)

    bt_coefs["vh37_lnline"] = bt.get_linfit(
        land_mask=ecdr_ide_ds["land_mask"].data,
        tb_mask=ecdr_ide_ds["invalid_tb_mask"].data[0, :, :],
        tbx=ecdr_ide_ds["v37_day_si"].data[0, :, :],
        tby=ecdr_ide_ds["h37_day_si"].data[0, :, :],
        lnline=bt_coefs_init["vh37_lnline"],
        add=bt_coefs["add1"],
        weather_mask=ecdr_ide_ds["bt_weather_mask"].data[0, :, :],
    )

    bt_coefs["bt_wtp_v37"] = bt.calculate_water_tiepoint(
        wtp_init=bt_coefs_init["bt_wtp_v37"],
        weather_mask=ecdr_ide_ds["bt_weather_mask"].data[0, :, :],
        tb=ecdr_ide_ds["v37_day_si"].data[0, :, :],
    )

    bt_coefs["bt_wtp_h37"] = bt.calculate_water_tiepoint(
        wtp_init=bt_coefs_init["bt_wtp_h37"],
        weather_mask=ecdr_ide_ds["bt_weather_mask"].data[0, :, :],
        tb=ecdr_ide_ds["h37_day_si"].data[0, :, :],
    )

    bt_coefs["bt_wtp_v19"] = bt.calculate_water_tiepoint(
        wtp_init=bt_coefs_init["bt_wtp_v19"],
        weather_mask=ecdr_ide_ds["bt_weather_mask"].data[0, :, :],
        tb=ecdr_ide_ds["v19_day_si"].data[0, :, :],
    )

    bt_coefs["ad_line_offset"] = bt.get_adj_ad_line_offset(
        wtp_x=bt_coefs["bt_wtp_v37"],
        wtp_y=bt_coefs["bt_wtp_h37"],
        line_37v37h=bt_coefs["vh37_lnline"],
    )

    bt_coefs["v1937_lnline"] = bt.get_linfit(
        land_mask=ecdr_ide_ds["land_mask"].data,
        tb_mask=ecdr_ide_ds["invalid_tb_mask"].data[0, :, :],
        tbx=ecdr_ide_ds["v37_day_si"].data[0, :, :],
        tby=ecdr_ide_ds["v19_day_si"].data[0, :, :],
        lnline=bt_coefs_init["v1937_lnline"],
        add=bt_coefs["add2"],
        weather_mask=ecdr_ide_ds["bt_weather_mask"].data[0, :, :],
        tba=ecdr_ide_ds["h37_day_si"].data[0, :, :],
        iceline=bt_coefs["vh37_lnline"],
        ad_line_offset=bt_coefs["ad_line_offset"],
    )

    # finally, compute the CDR.
    bt_conc, nt_conc, cdr_conc_raw = calculate_bt_nt_cdr_raw_conc(
        tb_h19=ecdr_ide_ds["h19_day_si"].data[0, :, :],
        tb_v37=ecdr_ide_ds["v37_day_si"].data[0, :, :],
        tb_h37=ecdr_ide_ds["h37_day_si"].data[0, :, :],
        tb_v19=ecdr_ide_ds["v19_day_si"].data[0, :, :],
        bt_coefs=bt_coefs,
        nt_coefs=nt_coefs,
        platform=platform,
    )

    # Apply masks
    # Get Nasateam weather filter
    nt_gr_2219 = nt.compute_ratio(
        ecdr_ide_ds["v22_day_si"].data[0, :, :],
        ecdr_ide_ds["v19_day_si"].data[0, :, :],
    )
    nt_gr_3719 = nt.compute_ratio(
        ecdr_ide_ds["v37_day_si"].data[0, :, :],
        ecdr_ide_ds["v19_day_si"].data[0, :, :],
    )
    nt_weather_mask = nt.get_weather_filter_mask(
        gr_2219=nt_gr_2219,
        gr_3719=nt_gr_3719,
        gr_2219_threshold=nt_coefs["nt_gradient_thresholds"]["2219"],
        gr_3719_threshold=nt_coefs["nt_gradient_thresholds"]["3719"],
    )

    ecdr_ide_ds["nt_weather_mask"] = (
        ("time", "y", "x"),
        np.expand_dims(nt_weather_mask.data, axis=0),
        {
            "grid_mapping": "crs",
            "standard_name": "weather_binary_mask",
            "long_name": "Map of weather masquerading as sea ice per NT",
            "comment": (
                "Mask indicating pixels with erroneously detected sea ice"
                " because of weather per NT "
            ),
            "units": 1,
        },
        {
            "zlib": True,
        },
    )

    set_to_zero_sic = (
        ecdr_ide_ds["nt_weather_mask"].data[0, :, :]
        | ecdr_ide_ds["bt_weather_mask"].data[0, :, :]
        | ecdr_ide_ds["invalid_ice_mask"].data[0, :, :]
    )

    cdr_conc = cdr_conc_raw.copy()
    cdr_conc[set_to_zero_sic] = 0

    # Will use spillover_applied with values:
    #  1: NT2
    #  2: BT (not yet added)
    #  4: NT (not yet added)
    spillover_applied = np.zeros((ydim, xdim), dtype=np.uint8)
    cdr_conc_pre_spillover = cdr_conc.copy()
    logger.info("Applying NT2 land spillover technique...")
    l90c = get_land90_conc_field(
        hemisphere=hemisphere,
        resolution=tb_data.resolution,
    )
    adj123 = get_adj123_field(
        hemisphere=hemisphere,
        resolution=tb_data.resolution,
    )
    cdr_conc = apply_nt2_land_spillover(
        conc=cdr_conc,
        adj123=adj123.data,
        l90c=l90c.data,
    )

    spillover_applied[cdr_conc_pre_spillover != cdr_conc.data] = 1

    # Fill the NH pole hole
    # TODO: Should check for NH and have grid-dependent filling scheme
    # NOTE: Usually, the pole hole will be filled in pass 3, along with melt onset calc.
    if fill_the_pole_hole and hemisphere == NORTH:
        cdr_conc_pre_polefill = cdr_conc.copy()
        platform = get_platform_by_date(date)
        near_pole_hole_mask = nh_polehole_mask(
            date=date,
            resolution=tb_data.resolution,
            sat=platform,
        )
        cdr_conc = fill_pole_hole(
            conc=cdr_conc,
            near_pole_hole_mask=near_pole_hole_mask.data,
        )
        logger.info("Filled pole hole")
        is_pole_filled = (cdr_conc != cdr_conc_pre_polefill) & (~np.isnan(cdr_conc))
        if "spatial_interpolation_bitmask" in ecdr_ide_ds.variables.keys():
            ecdr_ide_ds["spatial_interpolation_flag"] = ecdr_ide_ds[
                "spatial_interpolation_flag"
            ].where(
                ~is_pole_filled,
                other=TB_SPATINT_BITMASK_MAP["pole_filled"],
            )
            logger.info("Updated spatial_interpolation with pole hole value")

    # Mask out land and clamp conc to between 10-100%.
    # TODO: using the land mask here is not enough! We need to mask out all
    # "non-ocean" pixels, which includes coast & lakes.
    cdr_conc[land_mask.data] = np.nan
    cdr_conc[cdr_conc < 10] = 0
    cdr_conc[cdr_conc > 100] = 100

    # Add the BT raw field to the dataset
    bt_conc = bt_conc / 100.0  # re-set range from 0 to 1
    ecdr_ide_ds["raw_bt_seaice_conc"] = (
        ("time", "y", "x"),
        np.expand_dims(bt_conc, axis=0),
        {
            "grid_mapping": "crs",
            "standard_name": "sea_ice_area_fraction",
            "long_name": ("Bootstrap sea ice concentration, raw field with no masking"),
        },
        {
            "zlib": True,
            "dtype": "uint8",
            "scale_factor": 0.01,
            "_FillValue": 255,
        },
    )

    # Add the BT coefficients to the raw_bt_seaice_conc DataArray
    for attr in sorted(bt_coefs.keys()):
        if type(bt_coefs[attr]) in (float, int):
            ecdr_ide_ds.variables["raw_bt_seaice_conc"].attrs[attr] = bt_coefs[attr]
        else:
            ecdr_ide_ds.variables["raw_bt_seaice_conc"].attrs[attr] = str(
                bt_coefs[attr]
            )

    # Add the NT raw field to the dataset
    if (nt_conc > 200).any():
        logger.warning(
            "Raw nasateam concentrations above 200 were found."
            " This is unexpected may need to be investigated."
            f" Max nt value: {np.nanmax(nt_conc)}"
        )

    nt_conc = nt_conc / 100.0
    ecdr_ide_ds["raw_nt_seaice_conc"] = (
        ("time", "y", "x"),
        np.expand_dims(nt_conc, axis=0),
        {
            "grid_mapping": "crs",
            "standard_name": "sea_ice_area_fraction",
            "long_name": ("NASA Team sea ice concentration, raw field with no masking"),
        },
        {
            "zlib": True,
            "dtype": "uint8",
            "scale_factor": 0.01,
            "_FillValue": 255,
        },
    )

    # Add the NT coefficients to the raw_nt_seaice_conc DataArray
    for attr in sorted(nt_coefs.keys()):
        if type(nt_coefs[attr]) in (float, int):  # type: ignore[literal-required]
            ecdr_ide_ds.variables["raw_nt_seaice_conc"].attrs[attr] = nt_coefs[attr]  # type: ignore[literal-required]  # noqa
        else:
            ecdr_ide_ds.variables["raw_nt_seaice_conc"].attrs[attr] = str(nt_coefs[attr])  # type: ignore[literal-required]  # noqa

    # Add the final cdr_conc value to the xarray dataset
    # Rescale conc values to 0-1
    cdr_conc = cdr_conc / 100.0
    ecdr_ide_ds["conc"] = (
        ("time", "y", "x"),
        np.expand_dims(cdr_conc, axis=0),
        {
            "grid_mapping": "crs",
            "standard_name": "sea_ice_area_fraction",
            "long_name": "Sea ice concentration",
        },
        {
            "zlib": True,
            "dtype": "uint8",
            "scale_factor": 0.01,
            "_FillValue": 255,
        },
    )

    # Add the QA bitmask field to the initial daily xarray dataset
    #   1: BT weather
    #   2: NT weather
    #   4: NT2 spillover (or in general...any/all spillover corrections)
    #   8: Missing TBs (exclusive of valid_ice mask)
    #  16: Invalid ice mask
    #  32: Spatial interpolation applied
    #  64: *applied later* Temporal interpolation applied
    # 128: *applied later* Melt onset detected
    # TODO: dynamically read the bitmask values from the source dataset
    # (`flag_masks` & `flag_meanings`)
    qa_bitmask = np.zeros((ydim, xdim), dtype=np.uint8)
    qa_bitmask[ecdr_ide_ds["bt_weather_mask"].data[0, :, :]] += 1
    qa_bitmask[ecdr_ide_ds["nt_weather_mask"].data[0, :, :]] += 2
    qa_bitmask[spillover_applied == 1] += 4
    qa_bitmask[invalid_tb_mask & ~ecdr_ide_ds["invalid_ice_mask"].data[0, :, :]] += 8
    qa_bitmask[ecdr_ide_ds["invalid_ice_mask"].data[0, :, :]] += 16
    qa_bitmask[ecdr_ide_ds["spatial_interpolation_flag"].data[0, :, :] != 0] += 32
    qa_bitmask[land_mask] = 0

    ecdr_ide_ds["qa_of_cdr_seaice_conc"] = (
        ("time", "y", "x"),
        np.expand_dims(qa_bitmask, axis=0),
        {
            "grid_mapping": "crs",
            "standard_name": "status_flag",
            "long_name": "Sea Ice Concentration QC flags",
            "units": 1,
            "valid_range": [np.uint8(), np.uint8(255)],
        },
        {
            "zlib": True,
        },
    )

    return ecdr_ide_ds


def reproject_ideds(
    initial_ecdr_ds: xr.Dataset,
    date: dt.date,
    hemisphere: Hemisphere,
    tb_resolution: ECDR_SUPPORTED_RESOLUTIONS,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
) -> xr.Dataset:
    """Re-project the initial daily eCDR data set to a different grid.
    Currently, this is set up only to reproject 25km polar stereo to 12.5km
    """
    if tb_resolution == "25" and resolution == "12.5":
        reprojected_ideds = reproject_ideds_25to12(
            initial_ecdr_ds=initial_ecdr_ds,
            date=date,
            hemisphere=hemisphere,
            resolution=resolution,
        )
    else:
        raise RuntimeError(
            f"reproject_ideds() not defined for {tb_resolution} to {resolution}"
        )

    return reprojected_ideds


def initial_daily_ecdr_dataset(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
) -> xr.Dataset:
    """Create xr dataset containing the first pass of daily enhanced CDR."""
    tb_data = get_ecdr_tb_data(
        date=date,
        hemisphere=hemisphere,
    )
    initial_ecdr_ds = compute_initial_daily_ecdr_dataset(
        date=date,
        hemisphere=hemisphere,
        tb_data=tb_data,
    )

    # If the computed ide_ds is not on the desired grid (ie resolution),
    # then it needs to be projected to the new grid
    # In general, this should be a comparison of grid_id's, but for
    # eCDR, a comparison of resolutions will suffice
    # Finished!
    if resolution != tb_data.resolution:
        initial_ecdr_ds = reproject_ideds(
            initial_ecdr_ds=initial_ecdr_ds,
            date=date,
            hemisphere=hemisphere,
            tb_resolution=tb_data.resolution,
            resolution=resolution,
        )
        logger.info(f"Reprojected ide_ds to {resolution}km")

    return initial_ecdr_ds


def write_ide_netcdf(
    *,
    ide_ds: xr.Dataset,
    output_filepath: Path,
    uncompressed_fields: Iterable[str] = ("crs", "time", "y", "x"),
    excluded_fields: Iterable[str] = [],
    conc_fields: Iterable[str] = (
        "conc",
        "raw_nt_seaice_conc",
        "raw_bt_seaice_conc",
    ),
    tb_fields: Iterable[str] = ("h19_day_si", "h37_day_si"),
) -> Path:
    """Write the initial_ecdr_ds to a netCDF file and return the path."""
    logger.info(f"Writing netCDF of initial_daily eCDR file to: {output_filepath}")

    for excluded_field in excluded_fields:
        if excluded_field in ide_ds.variables.keys():
            ide_ds = ide_ds.drop_vars(excluded_field)

    nc_encoding = {}
    for varname in ide_ds.variables.keys():
        varname = cast(str, varname)
        if varname not in uncompressed_fields and varname in conc_fields:
            # Skip conc_fields here because the encoding is set
            #   during DataArray assignment
            pass
        elif varname not in uncompressed_fields and varname in tb_fields:
            # Encode tb vals with int16
            nc_encoding[varname] = {
                "zlib": True,
                "dtype": "int16",
                "scale_factor": 0.1,
                "_FillValue": 0,
            }
        else:
            nc_encoding[varname] = {
                "zlib": True,
            }

    ide_ds.to_netcdf(
        output_filepath,
        encoding=nc_encoding,
        unlimited_dims=[
            "time",
        ],
    )

    # Return the path if it exists
    return output_filepath


@cache
def get_idecdr_dir(*, ecdr_data_dir: Path) -> Path:
    """Daily initial output dir for ECDR processing."""
    idecdr_dir = ecdr_data_dir / "initial_daily"
    idecdr_dir.mkdir(exist_ok=True)

    return idecdr_dir


def get_idecdr_filepath(
    *,
    date: dt.date,
    platform,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ecdr_data_dir: Path,
) -> Path:
    """Yields the filepath of the pass1 -- idecdr -- intermediate file."""

    standard_fn = standard_daily_filename(
        hemisphere=hemisphere,
        date=date,
        # TODO: extract to kwarg!!!
        # sat="am2",
        sat=platform,
        resolution=resolution,
    )
    idecdr_fn = "idecdr_" + standard_fn
    idecdr_dir = get_idecdr_dir(ecdr_data_dir=ecdr_data_dir)
    idecdr_path = idecdr_dir / idecdr_fn

    return idecdr_path


def make_idecdr_netcdf(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ecdr_data_dir: Path,
    excluded_fields: Iterable[str] = [],
) -> None:
    logger.info(f"Creating idecdr for {date=}, {hemisphere=}, {resolution=}")
    ide_ds = initial_daily_ecdr_dataset(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
    )
    platform = get_platform_by_date(date)
    output_path = get_idecdr_filepath(
        date=date,
        platform=platform,
        hemisphere=hemisphere,
        ecdr_data_dir=ecdr_data_dir,
        resolution=resolution,
    )

    written_ide_ncfile = write_ide_netcdf(
        ide_ds=ide_ds,
        output_filepath=output_path,
        excluded_fields=excluded_fields,
    )
    logger.info(f"Wrote intermed daily ncfile: {written_ide_ncfile}")


def create_idecdr_for_date_range(
    *,
    hemisphere: Hemisphere,
    start_date: dt.date,
    end_date: dt.date,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ecdr_data_dir: Path,
    verbose_intermed_ncfile: bool = False,
) -> None:
    """Generate the initial daily ecdr files for a range of dates."""
    for date in date_range(start_date=start_date, end_date=end_date):
        try:
            platform = get_platform_by_date(date)

            if not verbose_intermed_ncfile:
                excluded_fields = [
                    "h19_day",
                    "v19_day",
                    "v22_day",
                    "h37_day",
                    "v37_day",
                    # "h19_day_si",  # include this field for melt onset calculation
                    "v19_day_si",
                    "v22_day_si",
                    # "h37_day_si",  # include this field for melt onset calculation
                    "v37_day_si",
                    "NT_icecon_min",
                    "land_mask",
                    "pole_mask",
                    "invalid_tb_mask",
                ]
            make_idecdr_netcdf(
                date=date,
                hemisphere=hemisphere,
                resolution=resolution,
                ecdr_data_dir=ecdr_data_dir,
                excluded_fields=excluded_fields,
            )

        # TODO: either catch and re-throw this exception or throw an error after
        # attempting to make the netcdf for each date. The exit code should be
        # non-zero in such a case.
        except Exception:
            logger.error(
                "Failed to create NetCDF for " f"{hemisphere=}, {date=}, {resolution=}."
            )
            # TODO: These error logs should be written to e.g.,
            # `/share/apps/logs/seaice_ecdr`. The `logger` module should be able
            # to handle automatically logging error details to such a file.
            err_filepath = get_idecdr_filepath(
                date=date,
                platform=platform,
                hemisphere=hemisphere,
                resolution=resolution,
                ecdr_data_dir=ecdr_data_dir,
            )
            err_filename = err_filepath.name + ".error"
            logger.info(f"Writing error info to {err_filename}")
            with open(err_filepath.parent / err_filename, "w") as f:
                traceback.print_exc(file=f)
                traceback.print_exc(file=sys.stdout)


@click.command(name="idecdr")
@click.option(
    "-d",
    "--date",
    required=True,
    type=click.DateTime(
        formats=(
            "%Y-%m-%d",
            "%Y%m%d",
            "%Y.%m.%d",
        )
    ),
    callback=datetime_to_date,
)
@click.option(
    "-h",
    "--hemisphere",
    required=True,
    type=click.Choice(get_args(Hemisphere)),
)
@click.option(
    "--ecdr-data-dir",
    required=True,
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
    default=STANDARD_BASE_OUTPUT_DIR,
    help=(
        "Base output directory for standard ECDR outputs."
        " Subdirectories are created for outputs of"
        " different stages of processing."
    ),
    show_default=True,
)
@click.option(
    "-r",
    "--resolution",
    required=True,
    type=click.Choice(get_args(ECDR_SUPPORTED_RESOLUTIONS)),
)
@click.option(
    "-v",
    "--verbose_intermed_ncfile",
    help=(
        "Create intermediate daily netcdf file that has"
        " extra fields unnecessary for subsequent CDR processing."
    ),
    required=False,
    default=False,
    type=bool,
)
def cli(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    ecdr_data_dir: Path,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    verbose_intermed_ncfile: bool,
) -> None:
    """Run the initial daily ECDR algorithm with AMSR2 data.

    TODO: eventually we want to be able to specify: date, grid (grid includes
    projection, resolution, and bounds), and TBtype (TB type includes source and
    methodology for getting those TBs onto the grid)
    """

    create_idecdr_for_date_range(
        hemisphere=hemisphere,
        start_date=date,
        end_date=date,
        resolution=resolution,
        ecdr_data_dir=ecdr_data_dir,
        verbose_intermed_ncfile=verbose_intermed_ncfile,
    )
