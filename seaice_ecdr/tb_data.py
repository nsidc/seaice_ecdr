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


# TODO: This is being used by non-amsr channels, eg from 0001
def ecdr_tbs_from_amsr_channels(*, xr_tbs: xr.Dataset) -> EcdrTbs:
    try:
        return EcdrTbs(
            v19=xr_tbs.v18.data,
            h19=xr_tbs.h18.data,
            v22=xr_tbs.v23.data,
            v37=xr_tbs.v36.data,
            h37=xr_tbs.h36.data,
        )
    except AttributeError:
        # This error happens when at least one of the TB fields is missing
        # eg on 10/10/1995 SH for 22v (from F13)
        is_found = []
        is_missing = []
        # TODO: This key list should be an EXPECTED_TB_NAMES list or similar?
        for key in ("v18", "h18", "v23", "v36", "h36"):
            try:
                tbvar = xr_tbs[key]
                assert tbvar.shape is not None
                is_found.append(key)
            except KeyError:
                is_missing.append(key)

        existing_tbvar = is_found[0]
        if len(is_missing) > 0:
            for missing_tbvar in is_missing:
                # TODO: It would be better to create a missing_tb dataarray
                #       from "first principals" rather than copy() and zero.
                #       Even better, there should be a check at the point
                #       of data ingest that ensures that we have all expected
                #       TB fields.
                xr_tbs[missing_tbvar] = xr_tbs[existing_tbvar].copy()
                xr_tbs[missing_tbvar][:] = np.nan
            logger.warning(f"WARNING: created NULL values for tb {missing_tbvar}")

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
        ecdr_tbs = ecdr_tbs_from_amsr_channels(xr_tbs=xr_tbs)
    except FileNotFoundError:
        ecdr_tbs = get_null_ecdr_tbs(hemisphere=hemisphere, resolution=tb_resolution)
        logger.warning(
            f"Used all-null TBS for date={date},"
            f" hemisphere={hemisphere},"
            f" resolution={tb_resolution}"
        )

    ecdr_tb_data = EcdrTbData(
        tbs=ecdr_tbs,
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
        ecdr_tbs = ecdr_tbs_from_amsr_channels(xr_tbs=xr_tbs)
    except FileNotFoundError:
        ecdr_tbs = get_null_ecdr_tbs(hemisphere=hemisphere, resolution=tb_resolution)
        logger.warning(
            f"Used all-null TBS for date={date},"
            f" hemisphere={hemisphere},"
            f" resolution={tb_resolution}"
        )

    ecdr_tb_data = EcdrTbData(
        tbs=ecdr_tbs,
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

        ecdr_tbs = EcdrTbs(
            v19=xr_tbs_0001.v18.data,
            h19=xr_tbs_0001.h18.data,
            v22=xr_tbs_0001.v23.data,
            v37=xr_tbs_0001.v36.data,
            h37=xr_tbs_0001.h36.data,
        )
    except FileNotFoundError:
        ecdr_tbs = get_null_ecdr_tbs(
            hemisphere=hemisphere, resolution=nsidc0001_resolution
        )
        logger.warning(
            f"Used all-null TBS for date={date},"
            f" hemisphere={hemisphere},"
            f" resolution={tb_resolution}"
            f" platform={platform}"
        )

    # TODO: For debugging TBs, consider a print/log statement such as this:
    # print(f'platform: {platform} with tbs: {xr_tbs.data_vars.keys()}')
    ecdr_tb_data = EcdrTbData(
        tbs=ecdr_tbs,
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
