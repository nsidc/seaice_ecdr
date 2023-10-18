"""Create the initial daily ECDR file.

Notes:

* `idecdr` is shorthand for == "Initial Daily ECDR"
"""

import datetime as dt
import sys
import traceback
from pathlib import Path
from typing import TypedDict, get_args

import click
import numpy as np
import numpy.typing as npt
import pm_icecon.bt.compute_bt_ic as bt
import pm_icecon.bt.params.ausi_amsr2 as pmi_bt_params
import pm_icecon.nt.compute_nt_ic as nt
import pm_icecon.nt.params.amsr2 as nt_amsr2_params
import xarray as xr
from loguru import logger
from pm_icecon._types import Hemisphere
from pm_icecon.bt.fields import get_bootstrap_fields
from pm_icecon.cli.util import datetime_to_date
from pm_icecon.constants import DEFAULT_FLAG_VALUES
from pm_icecon.fill_polehole import fill_pole_hole
from pm_icecon.interpolation import spatial_interp_tbs
from pm_icecon.land_spillover import apply_nt2_land_spillover
from pm_icecon.nt._types import NasateamGradientRatioThresholds
from pm_icecon.nt.tiepoints import NasateamTiePoints
from pm_icecon.util import date_range, standard_output_filename
from pm_tb_data.fetch.au_si import AU_SI_RESOLUTIONS, get_au_si_tbs

from seaice_ecdr.gridid_to_xr_dataarray import get_dataset_for_gridid
from seaice_ecdr.land_spillover import load_or_create_land90_conc, read_adj123_file
from seaice_ecdr.masks import psn_125_near_pole_hole_mask


def cdr_bootstrap(
    date: dt.date,
    tb_v37: npt.NDArray,
    tb_h37: npt.NDArray,
    tb_v19: npt.NDArray,
    tb_v22: npt.NDArray,
    bt_coefs,
    missing_flag_value: float,
):
    """Generate the raw bootstrap concentration field."""
    wtp_37v = bt_coefs["bt_wtp_v37"]
    wtp_37h = bt_coefs["bt_wtp_h37"]
    wtp_19v = bt_coefs["bt_wtp_v19"]

    itp_37v = bt_coefs["bt_itp_v37"]
    itp_37h = bt_coefs["bt_itp_h37"]
    itp_19v = bt_coefs["bt_itp_v19"]

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
        missing_flag_value=missing_flag_value,
    )

    return bt_conc


def cdr_nasateam(
    date: dt.date,
    tb_h19: npt.NDArray,
    tb_v37: npt.NDArray,
    tb_v19: npt.NDArray,
    nt_tiepoints: NasateamTiePoints,
) -> npt.NDArray:
    """Generate the raw NASA Team concentration field."""
    # Compute the NASA Team conc field
    # Note that concentrations from nasateam may be >100%
    nt_pr_1919 = nt.compute_ratio(tb_v19, tb_h19)
    nt_gr_3719 = nt.compute_ratio(tb_v37, tb_v19)
    nt_conc = nt.calc_nasateam_conc(
        pr_1919=nt_pr_1919,
        gr_3719=nt_gr_3719,
        tiepoints=nt_tiepoints,
    )

    return nt_conc


def get_bt_tb_mask(
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
    date: dt.date,
    tb_h19: npt.NDArray,
    tb_v37: npt.NDArray,
    tb_h37: npt.NDArray,
    tb_v19: npt.NDArray,
    tb_v22: npt.NDArray,
    bt_coefs: dict,
    nt_coefs: NtCoefs,
    missing_flag_value: float | int,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Run the CDR algorithm."""
    # First, get bootstrap conc.
    bt_conc = cdr_bootstrap(
        date,
        tb_v37,
        tb_h37,
        tb_v19,
        tb_v22,
        bt_coefs,
        missing_flag_value,
    )

    # Next, get nasateam conc. Note that concentrations from nasateam may be
    # >100%.
    nt_conc = cdr_nasateam(
        date,
        tb_h19,
        tb_v37,
        tb_v19,
        nt_coefs["nt_tiepoints"],
    )

    # Now calculate CDR SIC
    is_bt_seaice = (bt_conc > 0) & (bt_conc <= 100)
    use_nt_values = (nt_conc > bt_conc) & is_bt_seaice
    # Note: Here, values without sea ice (because no TBs) have val np.nan
    cdr_conc = bt_conc.copy()
    cdr_conc[use_nt_values] = nt_conc[use_nt_values]

    return bt_conc, nt_conc, cdr_conc


def _setup_ecdr_ds(*, grid_id: str, date: dt.date):
    # Initialize geo-referenced xarray Dataset
    ecdr_ide_ds = get_dataset_for_gridid(grid_id, date)

    # Set initial global attributes
    ecdr_ide_ds.attrs["description"] = "Initial daily cdr conc file"

    # Note: these attributes should probably go with
    #       a variable named "CDR_parameters" or similar
    ecdr_ide_ds.attrs["grid_id"] = grid_id
    ecdr_ide_ds.attrs["date"] = date.strftime("%Y-%m-%d")
    ecdr_ide_ds.attrs["missing_value"] = DEFAULT_FLAG_VALUES.missing

    file_date = dt.date(1970, 1, 1) + dt.timedelta(
        days=int(ecdr_ide_ds.variables["time"].data)
    )
    ecdr_ide_ds.attrs["time_coverage_start"] = str(
        dt.datetime(file_date.year, file_date.month, file_date.day, 0, 0, 0)
    )
    ecdr_ide_ds.attrs["time_coverage_end"] = str(
        dt.datetime(file_date.year, file_date.month, file_date.day, 23, 59, 59)
    )

    return ecdr_ide_ds


def _get_grid_id(*, hemisphere: Hemisphere, resolution: AU_SI_RESOLUTIONS) -> str:
    # Set the gridid
    if hemisphere == "north" and resolution == "12":
        gridid = "psn12.5"
    elif hemisphere == "south" and resolution == "12":
        gridid = "pss12.5"
    else:
        raise RuntimeError(
            f"Could not determine gridid from:\n" f"{hemisphere} and {resolution}"
        )

    return gridid


def compute_initial_daily_ecdr_dataset(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: AU_SI_RESOLUTIONS,
) -> xr.Dataset:
    """Create xr dataset containing the first pass of daily enhanced CDR."""
    gridid = _get_grid_id(
        hemisphere=hemisphere,
        resolution=resolution,
    )
    ecdr_ide_ds = _setup_ecdr_ds(
        grid_id=gridid,
        date=date,
    )

    # Get AU_SI TBs
    xr_tbs = get_au_si_tbs(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
    )

    # Move TBs to ecdr_ds
    for tbname in ("h18", "v18", "v23", "h36", "v36"):
        tb_varname = f"{tbname}_day"
        tbdata = xr_tbs.variables[tbname].data
        freq = tbname[1:]
        pol = tbname[:1]
        tb_longname = f"Daily TB {freq}{pol} from AU_SI{resolution}"
        tb_units = "K"
        ecdr_ide_ds[tb_varname] = (
            ("y", "x"),
            tbdata,
            {
                "_FillValue": 0,
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

    # Spatially interpolate the brightness temperatures
    for tbname in ("h18", "v18", "v23", "h36", "v36"):
        tb_day_name = f"{tbname}_day"
        tb_si_varname = f"{tb_day_name}_si"
        tb_si_data = spatial_interp_tbs(ecdr_ide_ds[tb_day_name].data)
        freq = tbname[1:]
        pol = tbname[:1]
        tb_si_longname = f"Spatially interpolated {ecdr_ide_ds[tb_day_name].long_name}"
        tb_units = "K"
        ecdr_ide_ds[tb_si_varname] = (
            ("y", "x"),
            tb_si_data,
            {
                "_FillValue": 0,
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

    # Set up the spatial interpolation bitmask field where the various
    # TB fields were interpolated
    ydim, xdim = ecdr_ide_ds["h18_day_si"].data.shape
    spatint_bitmask_arr = np.zeros((ydim, xdim), dtype=np.uint8)
    tb_spatint_bitmask_map = {
        "v18": 1,
        "h18": 2,
        "v23": 4,
        "v36": 8,
        "h36": 16,
        "pole_filled": 32,
    }
    for tbname in ("h18", "v18", "v23", "h36", "v36"):
        tb_varname = f"{tbname}_day"
        si_varname = f"{tbname}_day_si"
        is_tb_si_diff = (
            ecdr_ide_ds[tb_varname].data != ecdr_ide_ds[si_varname].data
        ) & (~np.isnan(ecdr_ide_ds[si_varname].data))
        spatint_bitmask_arr[is_tb_si_diff] += tb_spatint_bitmask_map[tbname]

    ecdr_ide_ds["spatint_bitmask"] = (
        ("y", "x"),
        spatint_bitmask_arr,
        {
            "_FillValue": 0,
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

    bt_coefs_init = pmi_bt_params.get_ausi_bootstrap_params(
        date=date,
        satellite="amsr2",
        gridid=gridid,
    )

    # Add the function that generates the bt_tb_mask to bt_coefs
    bt_coefs_init["bt_tb_data_mask_function"] = bt.tb_data_mask

    # Get the bootstrap fields and assign them to ide_ds DataArrays
    bt_fields = get_bootstrap_fields(
        date=date,
        satellite="amsr2",
        gridid=gridid,
    )

    # Encode invalid_ice_mask
    ecdr_ide_ds["invalid_ice_mask"] = (
        ("y", "x"),
        bt_fields["invalid_ice_mask"],
        {
            "_FillValue": 0,
            "grid_mapping": "crs",
            "standard_name": "seaice_binary_mask",
            "long_name": "invalid ice mask",
            "comment": (
                "Mask indicating where seaice will not exist on this day"
                " based on climatology"
            ),
            "units": 1,
            "valid_range": [0, 1],
        },
        {
            "zlib": True,
        },
    )

    # Encode land_mask
    ecdr_ide_ds["land_mask"] = (
        ("y", "x"),
        bt_fields["land_mask"],
        {
            "_FillValue": 0,
            "grid_mapping": "crs",
            "standard_name": "land_binary_mask",
            "long_name": "land mask",
            "comment": "Mask indicating where land is",
            "units": 1,
        },
        {
            "zlib": True,
        },
    )

    # Encode pole_mask
    # TODO: I think this is currently unused
    # ...but it should be coordinated with pole hole filling routines below
    # ...and the pole filling should occur after temporal interpolation
    if bt_fields["pole_mask"] is not None:
        ecdr_ide_ds["pole_mask"] = (
            ("y", "x"),
            bt_fields["pole_mask"],
            {
                "_FillValue": 0,
                "grid_mapping": "crs",
                "standard_name": "pole_binary_mask",
                "long_name": "pole mask",
                "comment": "Mask indicating where pole hole might be",
                "units": 1,
            },
            {
                "zlib": True,
            },
        )

    # Determine the NT fields and coefficients
    nt_params = nt_amsr2_params.get_amsr2_params(
        hemisphere=hemisphere,
        resolution=resolution,
    )
    nt_coefs = NtCoefs(
        nt_tiepoints=nt_params.tiepoints,
        nt_gradient_thresholds=nt_params.gradient_thresholds,
    )

    # Encode NT shoremap field
    ecdr_ide_ds["shoremap"] = (
        ("y", "x"),
        nt_params.shoremap,
        {
            "_FillValue": 0,
            "grid_mapping": "crs",
            "standard_name": "surface mask",
            "long_name": "NT shoremap",
            "comment": "Mask indicating land-adjacency of ocean pixels",
            "units": 1,
        },
        {
            "zlib": True,
        },
    )

    # Encode NT minic field
    ecdr_ide_ds["NT_icecon_min"] = (
        ("y", "x"),
        nt_params.minic,
        {
            "_FillValue": 0,
            "grid_mapping": "crs",
            "standard_name": "sea_ice_area_fraction",
            "long_name": "Minimum ice concentration over observation period",
            "comment": "Map indicating minimum observed ice concentration",
            "units": 1,
        },
        {
            "zlib": True,
        },
    )

    # Compute the invalid TB mask
    invalid_tb_mask = get_bt_tb_mask(
        tb_v37=ecdr_ide_ds["v36_day_si"].data,
        tb_h37=ecdr_ide_ds["h36_day_si"].data,
        tb_v19=ecdr_ide_ds["v18_day_si"].data,
        tb_v22=ecdr_ide_ds["v23_day_si"].data,
        mintb=bt_coefs_init["mintb"],
        maxtb=bt_coefs_init["maxtb"],
        tb_data_mask_function=bt_coefs_init["bt_tb_data_mask_function"],
    )

    ecdr_ide_ds["invalid_tb_mask"] = (
        ("y", "x"),
        invalid_tb_mask,
        {
            "_FillValue": 0,
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

    # Compute the BT weather mask
    bt_weather_mask = bt.get_weather_mask(
        v37=ecdr_ide_ds["v36_day_si"].data,
        h37=ecdr_ide_ds["h36_day_si"].data,
        v22=ecdr_ide_ds["v23_day_si"].data,
        v19=ecdr_ide_ds["v18_day_si"].data,
        land_mask=ecdr_ide_ds["land_mask"].data,
        tb_mask=ecdr_ide_ds["invalid_tb_mask"],
        ln1=bt_coefs_init["vh37_lnline"],
        date=date,
        wintrc=bt_coefs_init["wintrc"],
        wslope=bt_coefs_init["wslope"],
        wxlimt=bt_coefs_init["wxlimt"],
    )

    ecdr_ide_ds["bt_weather_mask"] = (
        ("y", "x"),
        bt_weather_mask.data,
        {
            "_FillValue": 0,
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
        tb_mask=ecdr_ide_ds["invalid_tb_mask"],
        tbx=ecdr_ide_ds["v36_day_si"].data,
        tby=ecdr_ide_ds["h36_day_si"].data,
        lnline=bt_coefs_init["vh37_lnline"],
        add=bt_coefs["add1"],
        weather_mask=ecdr_ide_ds["bt_weather_mask"],
    )

    bt_coefs["bt_wtp_v37"] = bt.calculate_water_tiepoint(
        wtp_init=bt_coefs_init["bt_wtp_v37"],
        weather_mask=ecdr_ide_ds["bt_weather_mask"],
        tb=ecdr_ide_ds["v36_day_si"].data,
    )

    bt_coefs["bt_wtp_h37"] = bt.calculate_water_tiepoint(
        wtp_init=bt_coefs_init["bt_wtp_h37"],
        weather_mask=ecdr_ide_ds["bt_weather_mask"],
        tb=ecdr_ide_ds["h36_day_si"].data,
    )

    bt_coefs["bt_wtp_v19"] = bt.calculate_water_tiepoint(
        wtp_init=bt_coefs_init["bt_wtp_v19"],
        weather_mask=ecdr_ide_ds["bt_weather_mask"],
        tb=ecdr_ide_ds["v18_day_si"].data,
    )

    bt_coefs["ad_line_offset"] = bt.get_adj_ad_line_offset(
        wtp_x=bt_coefs["bt_wtp_v37"],
        wtp_y=bt_coefs["bt_wtp_h37"],
        line_37v37h=bt_coefs["vh37_lnline"],
    )

    bt_coefs["v1937_lnline"] = bt.get_linfit(
        land_mask=ecdr_ide_ds["land_mask"].data,
        tb_mask=ecdr_ide_ds["invalid_tb_mask"],
        tbx=ecdr_ide_ds["v36_day_si"].data,
        tby=ecdr_ide_ds["v18_day_si"].data,
        lnline=bt_coefs_init["v1937_lnline"],
        add=bt_coefs["add2"],
        weather_mask=ecdr_ide_ds["bt_weather_mask"],
        tba=ecdr_ide_ds["h36_day_si"].data,
        iceline=bt_coefs["vh37_lnline"],
        ad_line_offset=bt_coefs["ad_line_offset"],
    )

    # finally, compute the CDR.
    bt_conc, nt_conc, cdr_conc_raw = calculate_bt_nt_cdr_raw_conc(
        date=date,
        tb_h19=ecdr_ide_ds["h18_day_si"].data,
        tb_v37=ecdr_ide_ds["v36_day_si"].data,
        tb_h37=ecdr_ide_ds["h36_day_si"].data,
        tb_v19=ecdr_ide_ds["v18_day_si"].data,
        tb_v22=ecdr_ide_ds["v23_day_si"].data,
        bt_coefs=bt_coefs,
        nt_coefs=nt_coefs,
        missing_flag_value=ecdr_ide_ds.attrs["missing_value"],
    )

    # Apply masks
    # Get Nasateam weather filter
    nt_gr_2219 = nt.compute_ratio(
        ecdr_ide_ds["v23_day_si"].data,
        ecdr_ide_ds["v18_day_si"].data,
    )
    nt_gr_3719 = nt.compute_ratio(
        ecdr_ide_ds["v36_day_si"].data,
        ecdr_ide_ds["v18_day_si"].data,
    )
    nt_weather_mask = nt.get_weather_filter_mask(
        gr_2219=nt_gr_2219,
        gr_3719=nt_gr_3719,
        gr_2219_threshold=nt_coefs["nt_gradient_thresholds"]["2219"],
        gr_3719_threshold=nt_coefs["nt_gradient_thresholds"]["3719"],
    )

    ecdr_ide_ds["nt_weather_mask"] = (
        ("y", "x"),
        nt_weather_mask,
        {
            "_FillValue": 0,
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
        ecdr_ide_ds["nt_weather_mask"].data
        | ecdr_ide_ds["bt_weather_mask"].data
        | ecdr_ide_ds["invalid_ice_mask"].data
    )

    cdr_conc = cdr_conc_raw.copy()
    cdr_conc[set_to_zero_sic] = 0

    # Apply land spillover corrections
    # TODO: eventually, we want each of these routines to return a e.g.,
    #   delta that can be applied to the input concentration
    #   instead of returning a new conc. Then we would have a
    #   seprate algorithm for choosing how to apply
    #   multiple spillover deltas to a given conc field.

    use_only_nt2_spillover = True

    tb_h19 = ecdr_ide_ds["h18_day_si"].data
    # Will use spillover_applied with values:
    #  1: NT2
    #  2: BT (not yet added)
    #  4: NT (not yet added)
    spillover_applied = np.zeros((ydim, xdim), dtype=np.uint8)
    cdr_conc_pre_spillover = cdr_conc.copy()
    if use_only_nt2_spillover:
        logger.info("Applying NT2 land spillover technique...")
        if tb_h19.shape == (896, 608):
            # NH
            l90c = load_or_create_land90_conc(
                gridid="psn12.5",
                xdim=608,
                ydim=896,
                overwrite=False,
            )
            adj123 = read_adj123_file(
                gridid="psn12.5",
                xdim=608,
                ydim=896,
            )
            cdr_conc = apply_nt2_land_spillover(
                conc=cdr_conc,
                adj123=adj123,
                l90c=l90c,
            )
        elif tb_h19.shape == (664, 632):
            # SH
            l90c = load_or_create_land90_conc(
                gridid="pss12.5",
                xdim=632,
                ydim=664,
                overwrite=False,
            )
            adj123 = read_adj123_file(
                gridid="pss12.5",
                xdim=632,
                ydim=664,
            )
            cdr_conc = apply_nt2_land_spillover(
                conc=cdr_conc,
                adj123=adj123,
                l90c=l90c,
            )

        else:
            raise SystemExit(
                "Could not determine hemisphere from tb shape: {tb_h19.shape}"
            )
    else:
        # TODO: Fix minic means field for NT and float for BT (!)
        # nasateam first:
        logger.info("Applying NASA TEAM land spillover technique...")
        cdr_conc = nt.apply_nt_spillover(
            conc=cdr_conc,
            shoremap=ecdr_ide_ds["shoremap"],
            minic=ecdr_ide_ds["NT_icecon_min"],
        )
        # then bootstrap:
        logger.info("Applying Bootstrap land spillover technique...")
        cdr_conc = bt.coastal_fix(
            conc=cdr_conc,
            missing_flag_value=ecdr_ide_ds.attrs["missing_value"],
            land_mask=ecdr_ide_ds["land_mask"].data,
            minic=bt_coefs["minic"],
        )
    spillover_applied[cdr_conc_pre_spillover != cdr_conc.data] = 1

    # Fill the NH pole hole
    # TODO: Should check for NH and have grid-dependent filling scheme
    if cdr_conc.shape == (896, 608):
        cdr_conc_pre_polefill = cdr_conc.copy()
        near_pole_hole_mask = psn_125_near_pole_hole_mask()
        cdr_conc = fill_pole_hole(
            conc=cdr_conc,
            near_pole_hole_mask=near_pole_hole_mask,
        )
        logger.info("Filled pole hole")
        is_pole_filled = (cdr_conc != cdr_conc_pre_polefill) & (~np.isnan(cdr_conc))
        if "spatint_bitmask" in ecdr_ide_ds.variables.keys():
            ecdr_ide_ds["spatint_bitmask"].data[
                is_pole_filled
            ] += tb_spatint_bitmask_map["pole_filled"]
            logger.info("Updated spatial_interpolation with pole hole value")

    # Apply land flag value and clamp max conc to 100.
    # TODO: extract this func from nt and allow override of flag values
    cdr_conc = nt._clamp_conc_and_set_flags(
        shoremap=ecdr_ide_ds["shoremap"],
        conc=cdr_conc,
    )

    # Add the BT raw field to the dataset
    if bt_conc is not None:
        ecdr_ide_ds["bt_conc_raw"] = (
            ("y", "x"),
            bt_conc,
            {
                "_FillValue": 255,
                "grid_mapping": "crs",
                "standard_name": "sea_ice_area_fraction",
                "long_name": (
                    "Bootstrap sea ice concentration, raw field with no masking"
                ),
            },
            {
                "zlib": True,
            },
        )

    # Add the BT coefficients to the bt_conc_raw DataArray
    for attr in sorted(bt_coefs.keys()):
        if type(bt_coefs[attr]) in (float, int):
            ecdr_ide_ds.variables["bt_conc_raw"].attrs[attr] = bt_coefs[attr]
        else:
            ecdr_ide_ds.variables["bt_conc_raw"].attrs[attr] = str(bt_coefs[attr])

    # Add the NT raw field to the dataset
    if nt_conc is not None:
        ecdr_ide_ds["nt_conc_raw"] = (
            ("y", "x"),
            nt_conc,
            {
                "_FillValue": 255,
                "grid_mapping": "crs",
                "standard_name": "sea_ice_area_fraction",
                "long_name": (
                    "NASA Team sea ice concentration, raw field with no masking"
                ),
            },
            {
                "zlib": True,
            },
        )

    # Add the NT coefficients to the nt_conc_raw DataArray
    for attr in sorted(nt_coefs.keys()):
        if type(nt_coefs[attr]) in (float, int):  # type: ignore[literal-required]
            ecdr_ide_ds.variables["nt_conc_raw"].attrs[attr] = nt_coefs[attr]  # type: ignore[literal-required]  # noqa
        else:
            ecdr_ide_ds.variables["nt_conc_raw"].attrs[attr] = str(nt_coefs[attr])  # type: ignore[literal-required]  # noqa

    # Add the final cdr_conc value to the xarray dataset
    ecdr_ide_ds["conc"] = (
        ("time", "y", "x"),
        np.expand_dims(cdr_conc, axis=0),
        {
            "_FillValue": 255,
            "grid_mapping": "crs",
            "standard_name": "sea_ice_area_fraction",
            "long_name": "Sea ice concentration",
        },
        {
            "zlib": True,
        },
    )

    # Add the QA bitmask field to the initial daily xarray dataset
    #   1: BT weather
    #   2: NT weather
    #   4: NT2 spillover (or in general...any/all spillover corrections)
    #   8: Missing TBs (exclusive of valid_ice mask)
    #  16: Invalid ice mask
    #  32: Spatial interpolation applied
    # --
    #  64: Temporal interpolation applied (applied later)
    # 128: Melt onset detected (applied later)
    qa_bitmask = np.zeros((ydim, xdim), dtype=np.uint8)
    qa_bitmask[ecdr_ide_ds["bt_weather_mask"].data] += 1
    qa_bitmask[ecdr_ide_ds["nt_weather_mask"].data] += 2
    qa_bitmask[spillover_applied == 1] += 4
    qa_bitmask[invalid_tb_mask & ~ecdr_ide_ds["invalid_ice_mask"].data] += 8
    qa_bitmask[ecdr_ide_ds["invalid_ice_mask"].data] += 16
    qa_bitmask[ecdr_ide_ds["spatint_bitmask"].data != 0] += 32
    ecdr_ide_ds["qa_of_cdr_seaice_conc"] = (
        ("y", "x"),
        qa_bitmask,
        {
            "_FillValue": 0,
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

    # Finished!
    return ecdr_ide_ds


def make_cdr_netcdf(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    resolution: AU_SI_RESOLUTIONS,
    output_dir: Path,
) -> None:
    """Create the cdr netCDF file."""
    logger.info(f"Creating CDR for {date=}, {hemisphere=}, {resolution=}")
    conc_ds = compute_initial_daily_ecdr_dataset(
        date=date,
        hemisphere=hemisphere,
        resolution=resolution,
    )

    output_fn = standard_output_filename(
        hemisphere=hemisphere,
        date=date,
        sat="u2",
        algorithm="cdr",
        resolution=f"{resolution}km",
    )
    output_path = output_dir / output_fn
    conc_ds.to_netcdf(
        output_path,
        encoding={"conc": {"zlib": True}},
    )
    logger.info(f"Wrote AMSR2 CDR concentration field: {output_path}")


def create_idecdr_for_date_range(
    *,
    hemisphere: Hemisphere,
    start_date: dt.date,
    end_date: dt.date,
    resolution: AU_SI_RESOLUTIONS,
    output_dir: Path,
) -> None:
    """Generate the initial daily ecdr files for a range of dates."""
    for date in date_range(start_date=start_date, end_date=end_date):
        try:
            make_cdr_netcdf(
                date=date,
                hemisphere=hemisphere,
                resolution=resolution,
                output_dir=output_dir,
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
            err_filename = standard_output_filename(
                hemisphere=hemisphere,
                date=date,
                sat="u2",
                algorithm="cdr",
                resolution=f"{resolution}km",
            )
            err_filename += ".error"
            logger.info(f"Writing error info to {err_filename}")
            with open(output_dir / err_filename, "w") as f:
                traceback.print_exc(file=f)
                traceback.print_exc(file=sys.stdout)


@click.command(name="idecdr")
@click.option(
    "-d",
    "--date",
    required=True,
    type=click.DateTime(formats=("%Y-%m-%d",)),
    callback=datetime_to_date,
)
@click.option(
    "-h",
    "--hemisphere",
    required=True,
    type=click.Choice(get_args(Hemisphere)),
)
# TODO: default output-dir for initial daily files.
@click.option(
    "-o",
    "--output-dir",
    required=True,
    type=click.Path(
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
        path_type=Path,
    ),
)
@click.option(
    "-r",
    "--resolution",
    required=True,
    type=click.Choice(get_args(AU_SI_RESOLUTIONS)),
)
def cli(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
    output_dir: Path,
    resolution: AU_SI_RESOLUTIONS,
) -> None:
    """Run the initial daily ECDR algorithm with AMSR2 data."""
    create_idecdr_for_date_range(
        hemisphere=hemisphere,
        start_date=date,
        end_date=date,
        # gridid=gridid,
        # tbsrc=tb_source,
        resolution=resolution,
        output_dir=output_dir,
    )
