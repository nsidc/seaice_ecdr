import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Final, get_args

import numpy as np
import numpy.typing as npt
import xarray as xr
from loguru import logger
from pm_tb_data._types import Hemisphere
from pm_tb_data.fetch.amsr.ae_si import get_ae_si_tbs_from_disk
from pm_tb_data.fetch.amsr.au_si import get_au_si_tbs
from pm_tb_data.fetch.amsr.util import AMSR_RESOLUTIONS
from pm_tb_data.fetch.nsidc_0001 import NSIDC_0001_SATS, get_nsidc_0001_tbs_from_disk
from pm_tb_data.fetch.nsidc_0007 import get_nsidc_0007_tbs_from_disk

from seaice_ecdr._types import ECDR_SUPPORTED_RESOLUTIONS
from seaice_ecdr.platforms import SUPPORTED_SAT, get_platform_by_date

EXPECTED_ECDR_TB_NAMES = ("h19", "v19", "v22", "h37", "v37")


@dataclass
class EcdrTbs:
    v19: npt.NDArray
    h19: npt.NDArray
    v22: npt.NDArray
    v37: npt.NDArray
    h37: npt.NDArray


@dataclass
class EcdrTbData:
    tbs: EcdrTbs
    resolution: ECDR_SUPPORTED_RESOLUTIONS
    data_source: str
    platform: SUPPORTED_SAT


def get_null_grid(
    *, hemisphere: Hemisphere, resolution: ECDR_SUPPORTED_RESOLUTIONS
) -> npt.NDArray:
    grid_shapes = {
        "25": {"north": (448, 304), "south": (332, 316)},
        "12.5": {"north": (896, 608), "south": (664, 632)},
    }

    null_grid = np.full(grid_shapes[resolution][hemisphere], np.nan)

    return null_grid


def get_null_ecdr_tbs(
    *, hemisphere: Hemisphere, resolution: ECDR_SUPPORTED_RESOLUTIONS
) -> EcdrTbs:
    null_grid = get_null_grid(hemisphere=hemisphere, resolution=resolution)

    null_ecdr_tbs = EcdrTbs(
        v19=null_grid.copy(),
        h19=null_grid.copy(),
        v22=null_grid.copy(),
        v37=null_grid.copy(),
        h37=null_grid.copy(),
    )

    return null_ecdr_tbs


def _pm_icecon_amsr_res_str(
    *, resolution: ECDR_SUPPORTED_RESOLUTIONS
) -> AMSR_RESOLUTIONS:
    """Given an AMSR ECDR resolution string, return a compatible `pm_icecon` resolution string."""
    _ecdr_pm_icecon_resolution_mapping: dict[
        ECDR_SUPPORTED_RESOLUTIONS, AMSR_RESOLUTIONS
    ] = {
        "12.5": "12",
        "25": "25",
    }
    au_si_resolution_str = _ecdr_pm_icecon_resolution_mapping[resolution]

    return au_si_resolution_str


# TODO: Eventually, these should be renamed based on microwave
#       band names, not AMSR2 TB channel frequencies
def rename_0001_tbs(
    *,
    input_ds: xr.Dataset,
) -> xr.Dataset:
    """Rename 0001 TB fields for use with siconc code written for AMSR2."""
    nsidc0001_to_amsr_tb_mapping = {
        "h19": "h18",
        "v19": "v18",
        "v22": "v23",
        "h37": "h36",
        "v37": "v36",
    }

    new_tbs = {}
    for key in input_ds.data_vars.keys():
        data_var = input_ds.data_vars[key]
        new_key = nsidc0001_to_amsr_tb_mapping[key]
        new_tbs[new_key] = xr.DataArray(
            data_var.data,
            dims=("fake_y", "fake_x"),
            attrs=data_var.attrs,
        )

    new_ds = xr.Dataset(new_tbs)

    return new_ds


def create_null_au_si_tbs(
    *,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
) -> xr.Dataset:
    """Create xr dataset containing all null-value data for all tbs."""
    chan_desc = {
        "h18": "18.7 GHz horizontal daily average Tbs",
        "v18": "18.7 GHz vertical daily average Tbs",
        "h23": "23.8 GHz horizontal daily average Tbs",
        "v23": "23.8 GHz vertical daily average Tbs",
        "h36": "36.5 GHz horizontal daily average Tbs",
        "v36": "36.5 GHz vertical daily average Tbs",
        "h89": "89.0 GHz horizontal daily average Tbs",
        "v89": "89.0 GHz vertical daily average Tbs",
    }

    if hemisphere == "north":
        if resolution == "12.5":
            xdim = 608
            ydim = 896
        if resolution == "25":
            xdim = 304
            ydim = 448
    elif hemisphere == "south":
        if resolution == "12.5":
            xdim = 632
            ydim = 664
        elif resolution == "25":
            xdim = 316
            ydim = 332
    else:
        value_error_string = f"""
        Could not create null_set of TBs for
          hemisphere: {hemisphere}
          resolution: {resolution}
        """
        raise ValueError(value_error_string)

    null_array = np.zeros((ydim, xdim), dtype=np.float64)
    null_array[:] = np.nan
    common_tb_attrs = {
        "_FillValue": 0,
        "units": "degree_kelvin",
        "standard_name": "brightness_temperature",
        "packing_convention": "netCDF",
        "packing_convention_description": "unpacked = scale_factor x packed + add_offset",
        "scale_factor": 0.1,
        "add_offset": 0.0,
    }
    null_tbs = xr.Dataset(
        data_vars=dict(
            h18=(
                ["YDim", "XDim"],
                null_array,
                common_tb_attrs,
            ),
            v18=(
                ["YDim", "XDim"],
                null_array,
                common_tb_attrs,
            ),
            h23=(
                ["YDim", "XDim"],
                null_array,
                common_tb_attrs,
            ),
            v23=(
                ["YDim", "XDim"],
                null_array,
                common_tb_attrs,
            ),
            h36=(
                ["YDim", "XDim"],
                null_array,
                common_tb_attrs,
            ),
            v36=(
                ["YDim", "XDim"],
                null_array,
                common_tb_attrs,
            ),
            h89=(
                ["YDim", "XDim"],
                null_array,
                common_tb_attrs,
            ),
            v89=(
                ["YDim", "XDim"],
                null_array,
                common_tb_attrs,
            ),
        ),
    )

    for key in chan_desc.keys():
        null_tbs.data_vars[key].attrs.update({"long_name": chan_desc[key]})

    return null_tbs


def ecdr_tbs_from_amsr_channels(*, xr_tbs: xr.Dataset) -> EcdrTbs:
    return EcdrTbs(
        v19=xr_tbs.v18.data,
        h19=xr_tbs.h18.data,
        v22=xr_tbs.v23.data,
        v37=xr_tbs.v36.data,
        h37=xr_tbs.h36.data,
    )


def _get_am2_tbs(*, date: dt.date, hemisphere: Hemisphere) -> EcdrTbData:
    tb_resolution: Final = "12.5"
    try:
        xr_tbs = get_au_si_tbs(
            date=date,
            hemisphere=hemisphere,
            resolution="12",
        )
    except FileNotFoundError:
        xr_tbs = create_null_au_si_tbs(
            hemisphere=hemisphere,
            resolution=tb_resolution,
        )
        logger.warning(
            f"Used all-null TBS for date={date},"
            f" hemisphere={hemisphere},"
            f" resolution={tb_resolution}"
        )

    ecdr_tb_data = EcdrTbData(
        tbs=ecdr_tbs_from_amsr_channels(xr_tbs=xr_tbs),
        resolution=tb_resolution,
        data_source="AU_SI12",
        platform="am2",
    )

    return ecdr_tb_data


def _get_ame_tbs(*, date: dt.date, hemisphere: Hemisphere) -> EcdrTbData:
    tb_resolution: Final = "12.5"
    ame_resolution_str = _pm_icecon_amsr_res_str(resolution=tb_resolution)
    AME_DATA_DIR = Path("/ecs/DP4/AMSA/AE_SI12.003/")
    try:
        xr_tbs = get_ae_si_tbs_from_disk(
            date=date,
            hemisphere=hemisphere,
            data_dir=AME_DATA_DIR,
            resolution=ame_resolution_str,
        )
    except FileNotFoundError:
        logger.warning(f"Using null AU_SI12 for AME on {date}")
        xr_tbs = create_null_au_si_tbs(
            hemisphere=hemisphere,
            resolution=tb_resolution,
        )
        logger.warning(
            f"Used all-null TBS for date={date},"
            f" hemisphere={hemisphere},"
            f" resolution={tb_resolution}"
        )

    ecdr_tb_data = EcdrTbData(
        tbs=ecdr_tbs_from_amsr_channels(xr_tbs=xr_tbs),
        resolution=tb_resolution,
        data_source="AE_SI12",
        platform="ame",
    )

    return ecdr_tb_data


def _get_nsidc_0001_tbs(
    *, date: dt.date, hemisphere: Hemisphere, platform: NSIDC_0001_SATS
) -> EcdrTbData:
    NSIDC0001_DATA_DIR = Path("/ecs/DP4/PM/NSIDC-0001.006/")
    # NSIDC-0001 TBs for siconc are all at 25km
    nsidc0001_resolution: Final = "25"
    tb_resolution: Final = nsidc0001_resolution
    try:
        xr_tbs_0001 = get_nsidc_0001_tbs_from_disk(
            date=date,
            hemisphere=hemisphere,
            data_dir=NSIDC0001_DATA_DIR,
            resolution=nsidc0001_resolution,
            sat=platform,
        )
        # TODO: this is confusing. It seems like this does the same remapping as
        # `ecdr_tbs_from_amsr_channels`. We don't need both, and the name should
        # be generic enough to reflect that. Do all 0001 platforms share the
        # same remapping? IS there any input data we use that doesn't get
        # remapped this way?
        # All platforms in 0001 (F08, F11, F13, and F17 have the same channels:
        # ['h19', 'v19', 'v22', 'h37', 'v37']
        xr_tbs = rename_0001_tbs(input_ds=xr_tbs_0001)
    except FileNotFoundError:
        logger.warning(f"Using null TBs for {platform} on {date}")
        print("this needs to be create_null_0001_tbs()")
        xr_tbs = create_null_au_si_tbs(
            hemisphere=hemisphere,
            resolution=tb_resolution,
        )
        logger.warning(
            f"Used all-null TBS for date={date},"
            f" hemisphere={hemisphere},"
            f" resolution={tb_resolution}"
        )

    ecdr_tb_data = EcdrTbData(
        # TODO: does 0001 need a different mapping?
        tbs=ecdr_tbs_from_amsr_channels(xr_tbs=xr_tbs),
        resolution=tb_resolution,
        data_source="NSIDC-0001",
        platform=platform,  # type: ignore[arg-type]
    )

    return ecdr_tb_data


def _get_nsidc_0007_tbs(*, hemisphere: Hemisphere, date: dt.date) -> EcdrTbData:
    NSIDC0007_DATA_DIR = Path("/projects/DATASETS/nsidc0007_smmr_radiance_seaice_v01/")
    # resolution in km
    SMMR_RESOLUTION: Final = "25"

    try:
        xr_tbs = get_nsidc_0007_tbs_from_disk(
            date=date,
            hemisphere=hemisphere,
            data_dir=NSIDC0007_DATA_DIR,
        )
        # Available channels: h06, h37, v37, h10, v18, v10, h18, v06
        ecdr_tbs = EcdrTbs(
            v19=xr_tbs.v18.data,
            h19=xr_tbs.h18.data,
            v22=xr_tbs.v37.data,
            v37=xr_tbs.v37.data,
            h37=xr_tbs.h37.data,
        )
    except FileNotFoundError:
        logger.warning(f"Using null TBs for SMMR/n07 on {date}")
        ecdr_tbs = get_null_ecdr_tbs(hemisphere=hemisphere, resolution=SMMR_RESOLUTION)

    ecdr_tb_data = EcdrTbData(
        tbs=ecdr_tbs,
        resolution=SMMR_RESOLUTION,
        data_source="NSIDC-0007",
        platform="n07",
    )

    return ecdr_tb_data


def get_ecdr_tb_data(
    *,
    date: dt.date,
    hemisphere: Hemisphere,
) -> EcdrTbData:
    platform = get_platform_by_date(date)
    if platform == "am2":
        return _get_am2_tbs(date=date, hemisphere=hemisphere)
    elif platform == "ame":
        return _get_ame_tbs(date=date, hemisphere=hemisphere)
    elif platform in get_args(NSIDC_0001_SATS):
        return _get_nsidc_0001_tbs(
            platform=platform,  # type: ignore[arg-type]
            date=date,
            hemisphere=hemisphere,
        )
    elif platform == "n07":
        # SMMR
        return _get_nsidc_0007_tbs(date=date, hemisphere=hemisphere)
    else:
        raise RuntimeError(f"Platform not supported: {platform}")
