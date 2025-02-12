"""Create the initial daily ECDR file.

Notes:

* `idecdr` is shorthand for == "Initial Daily ECDR"
"""

import datetime as dt
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
from pm_icecon._types import LandSpilloverMethods
from pm_icecon.bt.params.nsidc0007 import get_smmr_params
from pm_icecon.bt.xfer_tbs import xfer_rss_tbs
from pm_icecon.errors import UnexpectedSatelliteError
from pm_icecon.interpolation import spatial_interp_tbs
from pm_icecon.nt._types import NasateamGradientRatioThresholds
from pm_icecon.nt.params.params import get_cdr_nt_params
from pm_icecon.nt.tiepoints import NasateamTiePoints
from pm_tb_data._types import NORTH, Hemisphere
from pm_tb_data.fetch.nsidc_0001 import NSIDC_0001_SATS

from seaice_ecdr._types import ECDR_SUPPORTED_RESOLUTIONS
from seaice_ecdr.ancillary import (
    ANCILLARY_SOURCES,
    get_empty_ds_with_time,
    get_invalid_ice_mask,
    get_non_ocean_mask,
    nh_polehole_mask,
)
from seaice_ecdr.cli.util import datetime_to_date
from seaice_ecdr.constants import (
    CDR_ANCILLARY_DIR,
    DEFAULT_BASE_OUTPUT_DIR,
    ECDR_PRODUCT_VERSION,
)
from seaice_ecdr.days_treated_differently import (
    day_has_all_bad_tbs,
)
from seaice_ecdr.grid_id import get_grid_id
from seaice_ecdr.nc_util import remove_FillValue_from_coordinate_vars
from seaice_ecdr.platforms import PLATFORM_CONFIG, SUPPORTED_PLATFORM_ID
from seaice_ecdr.regrid_25to12 import reproject_ideds_25to12
from seaice_ecdr.spillover import LAND_SPILL_ALGS, land_spillover
from seaice_ecdr.tb_data import (
    EXPECTED_ECDR_TB_NAMES,
    EcdrTbData,
    get_25km_ecdr_tb_data,
    get_ecdr_tb_data,
    get_null_tb_data,
)
from seaice_ecdr.util import (
    get_intermediate_output_dir,
    standard_daily_filename,
)


def platform_is_smmr(platform_id: SUPPORTED_PLATFORM_ID):
    return platform_id in ("n07", "s36")


def cdr_bootstrap_raw(
    *,
    tb_v37: npt.NDArray,
    tb_h37: npt.NDArray,
    tb_v19: npt.NDArray,
    bt_coefs,
    platform: SUPPORTED_PLATFORM_ID,
):
    """Generate the raw bootstrap concentration field.
    Note: tb fields should already be transformed before
          being passed to this function.
    Note: bt_conc output values will range from 0.0 to maybe 150.0%
          We will default to having the maximum value be 200%, which
          should cover all reasonable cases.
          This is the "maxic" in bt_coefs.
    """
    wtp_37v = bt_coefs["bt_wtp_v37"]
    wtp_37h = bt_coefs["bt_wtp_h37"]
    wtp_19v = bt_coefs["bt_wtp_v19"]

    itp_37v = bt_coefs["bt_itp_v37"]
    itp_37h = bt_coefs["bt_itp_h37"]
    itp_19v = bt_coefs["bt_itp_v19"]

    # Preserve bt_conc values up to 254% siconc.
    # We need a value here because the pm_icecon function requires it.
    # But we are only setting this particular value so that any calculated
    # conc value can be represented by an unsigned byte (non-flag 255) value
    bt_coefs["maxic"] = 2.54

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
        line_37v37h=bt_coefs["vh37_iceline"],
        line_37v19v=bt_coefs["v1937_iceline"],
        ad_line_offset=bt_coefs["ad_line_offset"],
        maxic_frac=bt_coefs["maxic"],
        missing_flag_value=255,
    )

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


def calc_cdr_conc(
    *,
    bt_conc: npt.NDArray,
    nt_conc: npt.NDArray,
    cdr_conc_min_fraction: float,
    cdr_conc_max_fraction: float,
) -> npt.NDArray:
    """
    Run the CDR algorithm
    Here, bt_conc and nt_conc are expected to have values in percent,
          though the upper range of values might be great than 100%.
          This is permitted for the bt_conc and nt_conc fields, but
          not for the cdr_conc field.
    Note: bt_conc and nt_conc values are expected to be >= 0.0 or np.nan
    Note: range of output values will be 0 to 100.0 and np.nan (=missing)
    """
    cdr_conc_min_percent = cdr_conc_min_fraction * 100.0
    cdr_conc_max_percent = cdr_conc_max_fraction * 100.0

    is_bt_seaice = (bt_conc > cdr_conc_min_percent) & np.isfinite(bt_conc)
    is_cdr_seaice = is_bt_seaice
    is_nt_seaice = (nt_conc > cdr_conc_min_percent) & np.isfinite(nt_conc)

    use_nt_values = is_nt_seaice & is_bt_seaice & (nt_conc > bt_conc)

    # Note: Here, values without sea ice (because no TBs) have val np.nan
    cdr_conc = bt_conc.copy()
    cdr_conc[use_nt_values] = nt_conc[use_nt_values]

    # Clamp cdr_conc to min/max
    is_low_siconc = is_cdr_seaice & (cdr_conc < cdr_conc_min_percent)
    cdr_conc[is_low_siconc] = 0

    is_high_siconc = is_cdr_seaice & (cdr_conc > cdr_conc_max_percent)
    cdr_conc[is_high_siconc] = cdr_conc_max_percent

    return cdr_conc


def _setup_ecdr_ds(
    *,
    date: dt.date,
    tb_data: EcdrTbData,
    hemisphere: Hemisphere,
    ancillary_source: ANCILLARY_SOURCES,
) -> xr.Dataset:
    # Initialize geo-referenced xarray Dataset
    grid_id = get_grid_id(
        hemisphere=hemisphere,
        resolution=tb_data.resolution,
    )

    ecdr_ide_ds = get_empty_ds_with_time(
        hemisphere=hemisphere,
        resolution=tb_data.resolution,
        date=date,
        ancillary_source=ancillary_source,
    )

    # Set initial global attributes

    # TODO: these attributes should probably go with
    #       a variable named "CDR_parameters" or similar
    ecdr_ide_ds.attrs["grid_id"] = grid_id

    # Set data_source attribute
    ecdr_ide_ds.attrs["data_source"] = tb_data.data_source

    # Set the platform
    ecdr_ide_ds.attrs["platform"] = tb_data.platform_id

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
                # TODO: These TB fields can be expressly packed as uint16
                #       because that is how they are initially provided.
            },
        )

    return ecdr_ide_ds


def get_nasateam_weather_mask(
    *,
    ecdr_ide_ds: xr.Dataset,
    nt_coefs: NtCoefs,
) -> npt.NDArray[np.bool_]:
    """Return the "nasateam" weather mask.

    TODO: the nasateam weather mask is the combination of two seaparate masks:
    one generated from the ratio between v37 and v19 and the other generated
    from a ratio between the v22 and v19 channels. When data are available to
    compute masks for both of these ratios, one single mask is returned, which
    is a combination of the aforementioned masks. Ideally, we separate these out
    and keep track of the masks separately so that we can evaluate the effect
    each has on the final output.
    """
    # Get Nasateam weather filter
    nt_gr_3719 = nt.compute_ratio(
        ecdr_ide_ds["v37_day_si"].data[0, :, :],
        ecdr_ide_ds["v19_day_si"].data[0, :, :],
    )
    nt_3719_weather_mask = nt_gr_3719 > nt_coefs["nt_gradient_thresholds"]["3719"]

    # SMMR does not have a 22v channel that's suitable for the nt weather
    # filter. Instead, we just re-use the 3719 gradient ratios for 2219.
    if platform_is_smmr(ecdr_ide_ds.platform):
        return nt_3719_weather_mask

    nt_gr_2219 = nt.compute_ratio(
        ecdr_ide_ds["v22_day_si"].data[0, :, :],
        ecdr_ide_ds["v19_day_si"].data[0, :, :],
    )
    nt_2219_weather_mask = nt_gr_2219 > nt_coefs["nt_gradient_thresholds"]["2219"]
    weather_mask = nt_3719_weather_mask | nt_2219_weather_mask

    is_zero_tb = (
        (ecdr_ide_ds["v22_day_si"].data[0, :, :] == 0)
        | (ecdr_ide_ds["v19_day_si"].data[0, :, :] == 0)
        | (ecdr_ide_ds["v37_day_si"].data[0, :, :] == 0)
    )
    weather_mask[is_zero_tb] = 0

    return weather_mask


def get_flagmask(
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    ancillary_source: ANCILLARY_SOURCES,
) -> None | npt.NDArray:
    """
    Return a set of flags (uint8s of value 251-255)
    that correspond to non-ocean features of this grid/landmask
    """

    # NOTE: This could be better organized and name-conventioned?

    # TODO: Put these flagmasks in the ancillary files because they
    #       are a convenience to the user for creating plots of the
    #       siconc fields with sentinel values defined.  But, for
    #       stricter compliance with CF-standards, we do not want
    #       these sentinel values in the actual data arrays.

    flagmask = None

    if resolution == "25" and hemisphere == "north":
        gridid = "psn25"
        xdim = 304
        ydim = 448
    elif resolution == "25" and hemisphere == "south":
        gridid = "pss25"
        xdim = 316
        ydim = 332
    elif resolution == "12.5" and hemisphere == "north":
        gridid = "psn12.5"
        xdim = 608
        ydim = 896
    elif resolution == "12.5" and hemisphere == "south":
        gridid = "pss12.5"
        xdim = 632
        ydim = 664

    if ancillary_source == "CDRv4":
        version_string = "v04r00"
    elif ancillary_source == "CDRv5":
        version_string = ECDR_PRODUCT_VERSION.version_str

    flagmask_fn = CDR_ANCILLARY_DIR / f"flagmask_{gridid}_{version_string}.dat"
    try:
        flagmask_fn.is_file()
    except AssertionError as e:
        print(f"No such flagmask_fn: {flagmask_fn}")
        raise e

    try:
        flagmask = np.fromfile(flagmask_fn, dtype=np.uint8).reshape(ydim, xdim)
    except ValueError as e:
        print(f"Could not reshape to: {xdim}, {ydim}")
        raise e

    return flagmask


def is_pre_AMSR_platform(platform_id: SUPPORTED_PLATFORM_ID):
    """Returns True for SMMR, SSMI, SSMIS platforms"""
    pre_AMSR_platforms = ("n07", "F08", "F11", "F13", "F17")
    return platform_id in pre_AMSR_platforms


def compute_bt_asCDRv4_conc(bt_conc, bt_weather_mask, invalid_ice_mask, non_ocean_mask):
    """
    This routine was created to match CDRv4 code.
    The only reason not to delete is is for hints of how to compare
    CDR concentration fields with legacy Goddard results.

    this would have been called before the land spillover algorithm with:

    bt_asCDRv4_conc = compute_bt_asCDRv4_conc(
        bt_conc,
        ecdr_ide_ds["bt_weather_mask"].data[0, :, :],
        ecdr_ide_ds["invalid_ice_mask"].data[0, :, :],
        ecdr_ide_ds["non_ocean_mask"].data,
    )

    # By inspection the difference between these values and CDRv4 values
    # was ~3E-05 (ie 4-byte floating point roundoff).
    """

    bt_asCDRv4_conc = bt_conc.copy()
    bt_asCDRv4_conc[bt_weather_mask] = 0
    bt_asCDRv4_conc[invalid_ice_mask] = 0
    bt_asCDRv4_conc[non_ocean_mask] = 120.0
    bt_asCDRv4_conc[np.isnan(bt_asCDRv4_conc)] = 110.0

    return bt_asCDRv4_conc


def compute_nt_asCDRv4_conc(nt_conc, weather_mask, invalid_ice_mask):
    """
    This routine was created to match CDRv4 code.
    The only reason not to delete is is for hints of how to compare
    CDR concentration fields with legacy Goddard results.

    this would have been called before the land spillover algorithm with:

    nt_asCDRv4_conc = compute_nt_asCDRv4_conc(nt_conc,
        weather_mask=ecdr_ide_ds["nt_weather_mask"].data[0, :, :],
        invalid_ice_mask=ecdr_ide_ds["invalid_ice_mask"].data[0, :, :],
    )

    """
    nt_asCDRv4_conc = nt_conc.copy()

    # Convert to 2-byte int
    is_nt_nan = np.isnan(nt_asCDRv4_conc)
    nt_asCDRv4_conc = (10 * nt_asCDRv4_conc).astype(np.int16)

    # In CDRv4, NT weather is only where weather condition removes NT conc
    is_ntwx_CDRv4 = (nt_conc > 0) & weather_mask
    nt_asCDRv4_conc[is_ntwx_CDRv4] = 0

    nt_asCDRv4_conc[invalid_ice_mask] = -10
    nt_asCDRv4_conc[is_nt_nan] = -10

    # Note: the NT array here is np.int16
    nt_asCDRv4_conc = nt_asCDRv4_conc.copy()
    is_negative_10 = nt_asCDRv4_conc == -10

    nt_asCDRv4_conc = np.divide(nt_asCDRv4_conc, 10.0).astype(np.float32)
    nt_asCDRv4_conc[is_negative_10] = -10

    return nt_asCDRv4_conc


def compute_initial_daily_ecdr_dataset(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    tb_data: EcdrTbData,
    land_spillover_alg: LAND_SPILL_ALGS,
    ancillary_source: ANCILLARY_SOURCES,
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
        ancillary_source=ancillary_source,
    )

    # Spatially interpolate the brightness temperatures
    for tbname in EXPECTED_ECDR_TB_NAMES:
        tb_day_name = f"{tbname}_day"

        tb_si_varname = f"{tb_day_name}_si"

        tb_si_data = spatial_interp_tbs(
            ecdr_ide_ds[tb_day_name].data[0, :, :],
        )

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
                "valid_range": [np.float64(10.0), np.float64(350.0)],
            },
            {
                "zlib": True,
            },
        )

    # Enforce missing TB value consistency across all channels
    _, ydim, xdim = ecdr_ide_ds["v19_day_si"].data.shape
    is_atleastone_zerotb = np.zeros((1, ydim, xdim), dtype="bool")
    is_atleastone_nantb = np.zeros((1, ydim, xdim), dtype="bool")
    tb_field_list = [
        "h19_day_si",
        "v19_day_si",
        "v22_day_si",
        "h37_day_si",
        "v37_day_si",
    ]
    for key in tb_field_list:
        if key in ecdr_ide_ds.variables.keys():
            is_zero_tb = ecdr_ide_ds[key].data == 0
            is_atleastone_zerotb[is_zero_tb] = True

            is_NaN_tb = np.isnan(ecdr_ide_ds[key].data)
            is_atleastone_nantb[is_NaN_tb] = True
    for key in tb_field_list:
        if key in ecdr_ide_ds.variables.keys():
            ecdr_ide_ds[key].data[is_atleastone_zerotb] = 0
            ecdr_ide_ds[key].data[is_atleastone_nantb] = np.nan

    # If there are no valid TBs in the ocean (non-non_ocean), then return empty
    non_ocean_mask = get_non_ocean_mask(
        hemisphere=hemisphere,
        resolution=tb_data.resolution,
        ancillary_source=ancillary_source,
    )
    is_ocean = ~non_ocean_mask.data
    is_valid_tb = ~(is_atleastone_zerotb | is_atleastone_nantb)
    n_valid_tb_grid_cells = np.sum(np.where(is_ocean & is_valid_tb, 1, 0))

    # Set up the spatial interpolation bitmask field where the various
    # TB fields were interpolated
    _, ydim, xdim = ecdr_ide_ds["h19_day_si"].data.shape
    spatint_bitmask_arr = np.zeros((ydim, xdim), dtype=np.uint8)
    # TODO: Long term reminder: we want to use "band" labels rather than
    #       exact GHz frequencies -- which vary by satellite -- to ID channels
    # TODO: This mapping should be found in some configuration elsewhere
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
        spatint_bitmask_arr[non_ocean_mask.data] = 0

        # Drop the un-spatially interpolated TB field to save space and compute
        ecdr_ide_ds = ecdr_ide_ds.drop_vars(tb_varname)

    # Set a missing data mask using 19V TB field
    # Note: this will not include missing TBs in the day's invalid_ice field
    ecdr_ide_ds["missing_tb_mask"] = (
        ("time", "y", "x"),
        np.isnan(ecdr_ide_ds["v19_day_si"].data),
        {
            "grid_mapping": "crs",
            "standard_name": "missing_tb_binary_mask",
            "long_name": "Map of Missing TBs",
            "comment": "Mask indicating pixels with missing TBs",
            "units": 1,
        },
        {
            "zlib": True,
        },
    )
    logger.debug("Initialized missing_tb_mask where TB was NaN")

    spat_int_flag_mask_values = np.array([1, 2, 4, 8, 16, 32], dtype=np.uint8)
    ecdr_ide_ds["cdr_seaice_conc_interp_spatial_flag"] = (
        ("time", "y", "x"),
        np.expand_dims(spatint_bitmask_arr, axis=0),
        {
            "grid_mapping": "crs",
            "standard_name": "status_flag",
            "long_name": "NOAA/NSIDC CDR of Passive Microwave Sea Ice Concentration spatial interpolation flags",
            "units": 1,
            "flag_masks": spat_int_flag_mask_values,
            "flag_meanings": (
                "19v_tb_value_interpolated"
                " 19h_tb_value_interpolated"
                " 22v_tb_value_interpolated"
                " 37v_tb_value_interpolated"
                " 37h_tb_value_interpolated"
                " Pole_hole_spatially_interpolated"
            ),
            "valid_range": [np.uint8(0), np.sum(spat_int_flag_mask_values)],
        },
        {
            "zlib": True,
        },
    )
    logger.debug(
        "Initialized cdr_seaice_conc_interp_spatial_flag with TB fill locations"
    )

    platform = PLATFORM_CONFIG.platform_for_id(tb_data.platform_id)
    if platform.id == "am2":
        bt_coefs_init = pmi_bt_params_amsr2.get_ausi_amsr2_bootstrap_params(
            date=date,
            satellite="amsr2",
            gridid=ecdr_ide_ds.grid_id,
        )
    elif platform.id == "ame":
        bt_coefs_init = pmi_bt_params_amsre.get_ausi_amsre_bootstrap_params(
            date=date,
            satellite="amsre",
            gridid=ecdr_ide_ds.grid_id,
        )
    # NOTE/TODO: we get F17 data from NSIDC-0080 for NRT processing, but we are
    # using the NSIDC-0001 specific BT params for F17 regardless of the source.
    elif platform.id in get_args(NSIDC_0001_SATS):
        bt_coefs_init = pmi_bt_params_0001.get_nsidc0001_bootstrap_params(
            date=date,
            satellite=platform.id,
            gridid=ecdr_ide_ds.grid_id,
        )
    elif platform_is_smmr(platform.id):
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
        ancillary_source=ancillary_source,
    )

    non_ocean_mask = get_non_ocean_mask(
        hemisphere=hemisphere,
        resolution=tb_data.resolution,
        ancillary_source=ancillary_source,
    )

    ecdr_ide_ds["invalid_ice_mask"] = invalid_ice_mask.expand_dims(dim="time")

    # Encode non_ocean_mask
    ecdr_ide_ds["non_ocean_mask"] = non_ocean_mask

    # Encode pole_mask
    # TODO: I think this is currently unused
    # ...but it should be coordinated with pole hole filling routines below
    # ...and the pole filling should occur after temporal interpolation
    if hemisphere == NORTH:
        pole_mask = nh_polehole_mask(
            date=date,
            resolution=tb_data.resolution,
            ancillary_source=ancillary_source,
            platform=platform,
        )
        ecdr_ide_ds["pole_mask"] = pole_mask

    # Even if there are no valid values, allow retrieval
    # of nt_params and nt_coefs
    nt_params = get_cdr_nt_params(
        hemisphere=hemisphere,
        platform=platform.id,
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

    # NOTE: Strictly speaking, perhaps this should happen before computing
    #       the TB mask?  But that does a very broad check (10 < TB < 350K)
    #       and so does not end up mattering
    # TODO: this mapping is in preparation for using separately-transformed
    #       TBs in the Bootstrap algorithm.  The better approach would be to
    #       adjust the parameters for each platform so that this tranformation
    #       would not be needed.  That's...a fair bit of work though.
    # NOTE: Notice that the bootstrap algorithms below will take these
    #       transformed *local* bt_??? vars, not he ecdr_ide_es arrays
    bt_v37 = ecdr_ide_ds["v37_day_si"].data[0, :, :]
    bt_h37 = ecdr_ide_ds["h37_day_si"].data[0, :, :]
    bt_v22 = ecdr_ide_ds["v22_day_si"].data[0, :, :]
    bt_v19 = ecdr_ide_ds["v19_day_si"].data[0, :, :]

    # Transform TBs for the bootstrap calculation
    # TODO: Are there separate NT algorithm TB transformations?
    try:
        transformed = xfer_rss_tbs(
            tbs=dict(
                v37=bt_v37,
                h37=bt_h37,
                v19=bt_v19,
                v22=bt_v22,
            ),
            platform=platform.id.lower(),
        )
        bt_v37 = transformed["v37"]
        bt_h37 = transformed["h37"]
        bt_v19 = transformed["v19"]
        bt_v22 = transformed["v22"]
    except UnexpectedSatelliteError:
        logger.info(f"Using un-transformed TBs for {platform=}.")

    # Compute the BT weather mask

    # Note: the variable returned is water_arr in cdralgos
    # Note: in cdrv4, only part of the water_arr is labeled "BT weather"
    #       but I think that's because there are two parts to the
    #       BT weather filter
    # bt_weather_mask = bt.get_weather_mask(
    if n_valid_tb_grid_cells == 0:
        bt_water_mask = np.zeros(bt_v37.shape, dtype=np.bool_)
    else:
        bt_water_mask = bt.get_water_mask(
            v37=bt_v37,
            h37=bt_h37,
            v22=bt_v22,
            v19=bt_v19,
            land_mask=ecdr_ide_ds["non_ocean_mask"].data,
            tb_mask=ecdr_ide_ds["invalid_tb_mask"].data[0, :, :],
            ln1=bt_coefs_init["vh37_lnline"],
            date=date,
            # TODO: in the future, we will want these to come
            #       from a bt_coefs structure, not bt_coefs_init
            wintrc=bt_coefs_init["wintrc"],
            wslope=bt_coefs_init["wslope"],
            wxlimt=bt_coefs_init["wxlimt"],
            is_smmr=platform_is_smmr(platform.id),
        )

    # Note:
    # Here, the "bt_weather_mask" is the almost "water_arr" variable in cdralgos
    # However, it does not include the invalid_tb_mask (which here also
    # includes the pole hole) nor does it include land, so:
    # cdralgos_water_arr = \
    #    bt_weather_mask \
    #    | (ecdr_ide_ds["invalid_tb_mask"].data \
    #    & ~ecdr_ide_ds["non_ocean_mask"].data)
    # Note that this cdralgos_water_arr will differ because it will have
    # zeros where there is no data because of the pole hole

    ecdr_ide_ds["bt_weather_mask"] = (  # note <-- name is weather_mask
        ("time", "y", "x"),
        np.expand_dims(bt_water_mask.data, axis=0),  # note <-- var water_mask
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
    bt_coefs = bt_coefs_init.copy()

    if n_valid_tb_grid_cells == 0:
        bt_conc = np.zeros(bt_v37.shape, dtype=np.float32)
        bt_conc[:] = np.nan

        nt_conc = np.zeros(bt_v37.shape, dtype=np.float32)
        nt_conc[:] = np.nan

        cdr_conc_raw = np.zeros(bt_v37.shape, dtype=np.float32)
        cdr_conc_raw[:] = np.nan

        nt_weather_mask = np.zeros(bt_v37.shape, dtype=np.bool_)

    else:
        bt_coefs_to_update = (
            "line_37v37h",
            "bt_wtp_v37",
            "bt_wtp_h37",
            "bt_wtp_v19",
            "ad_line_offset",
            "line_37v19v",
        )

        for coef in bt_coefs_to_update:
            bt_coefs.pop(coef, None)

        """
        This is the fortran function:
            call ret_linfit1(
         >    nc, nr, land, gdata, v37, h37, ln, lnchk, add1, water_arr,
         >    vh37)
        """
        bt_coefs["vh37_iceline"] = bt.get_linfit(
            land_mask=ecdr_ide_ds["non_ocean_mask"].data,
            tb_mask=ecdr_ide_ds["invalid_tb_mask"].data[0, :, :],
            tbx=bt_v37,  # Note: <-- not the ecdr_ide_ds key/value
            tby=bt_h37,  # Note: <-- not the ecdr_ide_ds key/value
            lnline=bt_coefs_init["vh37_lnline"],
            add=bt_coefs["add1"],
            water_mask=ecdr_ide_ds["bt_weather_mask"].data[0, :, :],  # note: water
        )

        bt_coefs["bt_wtp_v37"] = bt.calculate_water_tiepoint(
            wtp_init=bt_coefs_init["bt_wtp_v37"],
            water_mask=ecdr_ide_ds["bt_weather_mask"].data[0, :, :],
            tb=bt_v37,
        )

        bt_coefs["bt_wtp_h37"] = bt.calculate_water_tiepoint(
            wtp_init=bt_coefs_init["bt_wtp_h37"],
            water_mask=ecdr_ide_ds["bt_weather_mask"].data[0, :, :],
            tb=bt_h37,
        )

        bt_coefs["bt_wtp_v19"] = bt.calculate_water_tiepoint(
            wtp_init=bt_coefs_init["bt_wtp_v19"],
            water_mask=ecdr_ide_ds["bt_weather_mask"].data[0, :, :],
            tb=bt_v19,
        )

        bt_coefs["ad_line_offset"] = bt.get_adj_ad_line_offset(
            wtp_x=bt_coefs["bt_wtp_v37"],
            wtp_y=bt_coefs["bt_wtp_h37"],
            line_37v37h=bt_coefs["vh37_iceline"],
        )

        # NOTE: note that we are using bt_<vars> not <_day_si> vars...
        #       The bt_vars have been adjusted to match F13 TB distributions

        # NOTE: cdralgos uses ret_linfit1() for NH sets 1,2 and SH set2
        #            and uses ret_linfit2() for NH set 2
        if hemisphere == "south" and is_pre_AMSR_platform(platform.id):
            bt_coefs["v1937_iceline"] = bt.get_linfit(
                land_mask=ecdr_ide_ds["non_ocean_mask"].data,
                tb_mask=ecdr_ide_ds["invalid_tb_mask"].data[0, :, :],
                tbx=bt_v37,
                tby=bt_v19,
                lnline=bt_coefs_init["v1937_lnline"],
                add=bt_coefs["add2"],
                water_mask=ecdr_ide_ds["bt_weather_mask"].data[0, :, :],
                # these default to None in bt.get_linfit()
                #   this is equivalent to using "ret_linfit1(), not ret_linfit2()"
                # tba=bt_h37,
                # iceline=bt_coefs["vh37_iceline"],
                # ad_line_offset=bt_coefs["ad_line_offset"],
            )
        else:
            bt_coefs["v1937_iceline"] = bt.get_linfit(
                land_mask=ecdr_ide_ds["non_ocean_mask"].data,
                tb_mask=ecdr_ide_ds["invalid_tb_mask"].data[0, :, :],
                tbx=bt_v37,
                tby=bt_v19,
                lnline=bt_coefs_init["v1937_lnline"],
                add=bt_coefs["add2"],
                water_mask=ecdr_ide_ds["bt_weather_mask"].data[0, :, :],
                tba=bt_h37,
                iceline=bt_coefs["vh37_iceline"],
                ad_line_offset=bt_coefs["ad_line_offset"],
            )

        bt_conc = cdr_bootstrap_raw(
            tb_v37=bt_v37,
            tb_h37=bt_h37,
            tb_v19=bt_v19,
            bt_coefs=bt_coefs,
            platform=platform.id,
        )

        # Now, compute CDR version of NT estimate
        nt_conc = cdr_nasateam(
            tb_h19=ecdr_ide_ds["h19_day_si"].data[0, :, :],
            tb_v37=ecdr_ide_ds["v37_day_si"].data[0, :, :],
            tb_v19=ecdr_ide_ds["v19_day_si"].data[0, :, :],
            nt_tiepoints=nt_coefs["nt_tiepoints"],
        )

        cdr_conc_raw = calc_cdr_conc(
            bt_conc=bt_conc,
            nt_conc=nt_conc,
            cdr_conc_min_fraction=0.1,
            cdr_conc_max_fraction=1.0,
        )

        nt_weather_mask = get_nasateam_weather_mask(
            ecdr_ide_ds=ecdr_ide_ds, nt_coefs=nt_coefs
        )

    # end of cdr array calculations

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

    # Apply weather and invalid ice masks
    set_to_zero_sic = (
        ecdr_ide_ds["nt_weather_mask"].data[0, :, :]
        | ecdr_ide_ds["bt_weather_mask"].data[0, :, :]
        | ecdr_ide_ds["invalid_ice_mask"].data[0, :, :]
    )

    cdr_conc = cdr_conc_raw.copy()
    cdr_conc[set_to_zero_sic] = 0
    cdr_conc_pre_spillover = cdr_conc.copy()

    if n_valid_tb_grid_cells != 0:
        cdr_conc = land_spillover(
            cdr_conc=cdr_conc,
            hemisphere=hemisphere,
            tb_data=tb_data,
            algorithm=land_spillover_alg,
            land_mask=non_ocean_mask.data,
            ancillary_source=ancillary_source,
            fix_goddard_bt_error=True,
        )

    had_spillover_applied = (cdr_conc_pre_spillover != cdr_conc.data) & np.isfinite(
        cdr_conc.data
    )

    # Set missing TBs to NaN
    cdr_conc[ecdr_ide_ds["missing_tb_mask"].data[0, :, :]] = np.nan
    bt_conc[ecdr_ide_ds["missing_tb_mask"].data[0, :, :]] = np.nan
    nt_conc[ecdr_ide_ds["missing_tb_mask"].data[0, :, :]] = np.nan

    # Add the BT raw field to the dataset
    bt_conc = bt_conc / 100.0  # re-set range from 0 to 1
    ecdr_ide_ds["raw_bt_seaice_conc"] = (
        ("time", "y", "x"),
        np.expand_dims(bt_conc, axis=0),
        {
            "grid_mapping": "crs",
            "standard_name": "sea_ice_area_fraction",
            "long_name": (
                "NOAA/NSIDC CDR of Bootstrap sea ice concentration, raw field with no masking"
            ),
        },
        {
            "zlib": True,
            "dtype": "uint8",
            "scale_factor": 0.01,
            "_FillValue": 255,
        },
    )

    # Add the BT coefficients to the raw_bt_seaice_conc DataArray as attrs
    for attr in sorted(bt_coefs.keys()):
        if type(bt_coefs[attr]) in (float, int):
            ecdr_ide_ds.variables["raw_bt_seaice_conc"].attrs[attr] = bt_coefs[attr]
        else:
            ecdr_ide_ds.variables["raw_bt_seaice_conc"].attrs[attr] = str(
                bt_coefs[attr]
            )

    # Add the NT raw field to the dataset
    if (nt_conc[~non_ocean_mask.data] > 200).any():
        logger.warning(
            "Raw nasateam concentrations above 200% were found in ocean."
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
            "long_name": (
                "NOAA/NSIDC CDR of NASA Team sea ice concentration, raw field with no masking"
            ),
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
    # Rescale conc values fraction (i.e. 0 - 1 instead of 0.0 - 100.0
    cdr_conc = cdr_conc / 100.0
    # TODO: rename this variable from "conc" to "cdr_conc_init" or similar
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
    #   4: Land spillover applied (e.g,. NT2 land spillover)
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
    qa_bitmask[had_spillover_applied] += 4
    qa_bitmask[invalid_tb_mask & ~ecdr_ide_ds["invalid_ice_mask"].data[0, :, :]] += 8
    qa_bitmask[
        ecdr_ide_ds["cdr_seaice_conc_interp_spatial_flag"].data[0, :, :] != 0
    ] += 32

    # Explicitly set climatologically masked ocean cells to 16
    invalid_ice_ocean_mask = invalid_ice_mask.data & ~non_ocean_mask.data
    qa_bitmask[invalid_ice_ocean_mask] = 16

    # Explicitly set land to zero
    qa_bitmask[non_ocean_mask] = 0

    ecdr_ide_ds["cdr_seaice_conc_qa_flag"] = (
        ("time", "y", "x"),
        np.expand_dims(qa_bitmask, axis=0),
        {
            "grid_mapping": "crs",
            "standard_name": "status_flag",
            "long_name": "Sea Ice Concentration QA flags",
            "units": 1,
            "valid_range": [np.uint8(), np.uint8(255)],
        },
        {
            "zlib": True,
        },
    )

    if n_valid_tb_grid_cells == 0:
        logger.info(f"Skipped computations for empty idecdr on {date}")

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
    land_spillover_alg: LAND_SPILL_ALGS,
    ancillary_source: ANCILLARY_SOURCES,
) -> xr.Dataset:
    """Create xr dataset containing the first pass of daily enhanced CDR."""
    platform_id = PLATFORM_CONFIG.get_platform_by_date(date).id
    if day_has_all_bad_tbs(
        platform_id,
        hemisphere,
        date,
    ):
        tb_data = get_null_tb_data(
            hemisphere=hemisphere,
            resolution=resolution,
            date=date,
            platform_id=platform_id,
        )
        logger.warning(f"Used all null TB data for {platform_id} on {date}")
    else:
        # TODO: if/else should be temporary.
        #       It's just a way to clearly divide how a
        #       request for 12.5km or 25km ECDR is handled.
        if resolution == "12.5":
            # In the 12.5km case, we try to get 12.5km Tbs, but sometimes we get
            # 25km (older, non-AMSR platforms)
            tb_data = get_ecdr_tb_data(
                date=date,
                hemisphere=hemisphere,
            )
        else:
            # In the 25km case, we always expect 25km Tbs
            tb_data = get_25km_ecdr_tb_data(
                date=date,
                hemisphere=hemisphere,
            )

    initial_ecdr_ds = compute_initial_daily_ecdr_dataset(
        date=date,
        hemisphere=hemisphere,
        tb_data=tb_data,
        land_spillover_alg=land_spillover_alg,
        ancillary_source=ancillary_source,
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
        logger.debug(f"Reprojected ide_ds to {resolution}km")

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
        if excluded_field in ide_ds.data_vars.keys():
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
            # TODO: With uint16, these values could have scale factor 0.01
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

    ide_ds = remove_FillValue_from_coordinate_vars(ide_ds)
    ide_ds.to_netcdf(
        output_filepath,
        encoding=nc_encoding,
        unlimited_dims=[
            "time",
        ],
    )

    return output_filepath


@cache
def get_idecdr_dir(*, intermediate_output_dir: Path) -> Path:
    """Daily initial output dir for ECDR processing."""
    idecdr_dir = intermediate_output_dir / "initial_daily"
    idecdr_dir.mkdir(parents=True, exist_ok=True)

    return idecdr_dir


def get_idecdr_filepath(
    *,
    date: dt.date,
    platform_id: SUPPORTED_PLATFORM_ID,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    intermediate_output_dir: Path,
) -> Path:
    """Yields the filepath of the pass1 -- idecdr -- intermediate file."""

    standard_fn = standard_daily_filename(
        hemisphere=hemisphere,
        date=date,
        platform_id=platform_id,
        resolution=resolution,
    )
    idecdr_fn = "idecdr_" + standard_fn
    idecdr_dir = get_idecdr_dir(
        intermediate_output_dir=intermediate_output_dir,
    )
    idecdr_path = idecdr_dir / idecdr_fn

    return idecdr_path


def make_idecdr_netcdf(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    intermediate_output_dir: Path,
    excluded_fields: Iterable[str],
    land_spillover_alg: LAND_SPILL_ALGS,
    ancillary_source: ANCILLARY_SOURCES,
    overwrite_ide: bool = False,
    platform_id: SUPPORTED_PLATFORM_ID,
) -> None:
    output_path = get_idecdr_filepath(
        date=date,
        platform_id=platform_id,
        hemisphere=hemisphere,
        intermediate_output_dir=intermediate_output_dir,
        resolution=resolution,
    )

    if overwrite_ide or not output_path.is_file():
        logger.info(f"Creating idecdr for {date=}, {hemisphere=}, {resolution=}")
        ide_ds = initial_daily_ecdr_dataset(
            date=date,
            hemisphere=hemisphere,
            resolution=resolution,
            land_spillover_alg=land_spillover_alg,
            ancillary_source=ancillary_source,
        )

        written_ide_ncfile = write_ide_netcdf(
            ide_ds=ide_ds,
            output_filepath=output_path,
            excluded_fields=excluded_fields,
        )
        logger.info(f"Wrote initial daily ncfile: {written_ide_ncfile}")
    else:
        logger.info(f"idecdr file exists and {overwrite_ide=}: {output_path=}")


def read_or_create_and_read_idecdr_ds(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    intermediate_output_dir: Path,
    land_spillover_alg: LAND_SPILL_ALGS,
    ancillary_source: ANCILLARY_SOURCES,
    overwrite_ide: bool = False,
) -> xr.Dataset:
    """Read an idecdr netCDF file, creating it if it doesn't exist."""
    platform = PLATFORM_CONFIG.get_platform_by_date(
        date,
    )

    ide_filepath = get_idecdr_filepath(
        date=date,
        platform_id=platform.id,
        hemisphere=hemisphere,
        resolution=resolution,
        intermediate_output_dir=intermediate_output_dir,
    )
    if overwrite_ide or not ide_filepath.is_file():
        create_idecdr_for_date(
            date=date,
            hemisphere=hemisphere,
            resolution=resolution,
            intermediate_output_dir=intermediate_output_dir,
            land_spillover_alg=land_spillover_alg,
            ancillary_source=ancillary_source,
        )
    logger.debug(f"Reading ideCDR file from: {ide_filepath}")
    ide_ds = xr.load_dataset(ide_filepath)

    return ide_ds


def create_idecdr_for_date(
    date: dt.date,
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    intermediate_output_dir: Path,
    overwrite_ide: bool = False,
    verbose_intermed_ncfile: bool = False,
    land_spillover_alg: LAND_SPILL_ALGS,
    ancillary_source: ANCILLARY_SOURCES,
) -> None:
    """Create a standard IDECDR file for the given date.

    Uses the default platform start date config to determine the correct
    platform to use.
    """
    excluded_fields = []
    if not verbose_intermed_ncfile:
        excluded_fields = [
            "h19_day",
            "v19_day",
            "v22_day",
            "h37_day",
            "v37_day",
            # "h19_day_si",  # include this field for melt onset calculation
            "v19_day_si",  # comment out this line to get all TB fields
            "v22_day_si",  # comment out this line to get all TB fields
            # "h37_day_si",  # include this field for melt onset calculation
            "v37_day_si",  # comment out this line to get all TB fields
            "NT_icecon_min",
            "non_ocean_mask",
            "pole_mask",
            "invalid_tb_mask",
            "bt_weather_mask",
            "nt_weather_mask",
            "missing_tb_mask",
        ]
    try:
        platform = PLATFORM_CONFIG.get_platform_by_date(date)
        make_idecdr_netcdf(
            date=date,
            hemisphere=hemisphere,
            resolution=resolution,
            intermediate_output_dir=intermediate_output_dir,
            excluded_fields=excluded_fields,
            overwrite_ide=overwrite_ide,
            land_spillover_alg=land_spillover_alg,
            ancillary_source=ancillary_source,
            platform_id=platform.id,
        )

    except Exception as e:
        logger.exception(
            "Failed to create NetCDF for " f"{hemisphere=}, {date=}, {resolution=}."
        )
        raise e


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
    "--land-spillover-alg",
    required=True,
    type=click.Choice(get_args(LandSpilloverMethods)),
)
@click.option(
    "--base-output-dir",
    required=True,
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
    default=DEFAULT_BASE_OUTPUT_DIR,
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
@click.option(
    "--ancillary-source",
    required=True,
    type=click.Choice(get_args(ANCILLARY_SOURCES)),
)
def cli(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    base_output_dir: Path,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    land_spillover_alg: LAND_SPILL_ALGS,
    verbose_intermed_ncfile: bool,
    ancillary_source: ANCILLARY_SOURCES,
) -> None:
    """Run the initial daily ECDR algorithm with AMSR2 data.

    TODO: eventually we want to be able to specify: date, grid (grid includes
    projection, resolution, and bounds), and TBtype (TB type includes source and
    methodology for getting those TBs onto the grid)
    """
    intermediate_output_dir = get_intermediate_output_dir(
        base_output_dir=base_output_dir,
        hemisphere=hemisphere,
    )
    create_idecdr_for_date(
        hemisphere=hemisphere,
        date=date,
        resolution=resolution,
        intermediate_output_dir=intermediate_output_dir,
        verbose_intermed_ncfile=verbose_intermed_ncfile,
        land_spillover_alg=land_spillover_alg,
        ancillary_source=ancillary_source,
    )
