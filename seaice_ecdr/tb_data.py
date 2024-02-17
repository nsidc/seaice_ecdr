import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from typing import Final, cast, get_args

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
from seaice_ecdr.util import get_ecdr_grid_shape

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
    grid_shape = get_ecdr_grid_shape(
        hemisphere=hemisphere,
        resolution=resolution,
    )
    null_grid = np.full(grid_shape, np.nan)

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


def map_tbs_to_ecdr_channels(
    *,
    mapping: dict[str, str],
    xr_tbs: xr.Dataset,
    hemisphere: Hemisphere,
    resolution: ECDR_SUPPORTED_RESOLUTIONS,
    data_source: str,
    date: dt.date,
) -> EcdrTbs:
    """Map tbs from source to ECDR channels.

    `mapping` should be a dictionary with keys representing the expected ECDR
    channel names. The values should be input data sources' channel names (these
    should be variables in the `xr_tbs` dataset).

    If the provided `xr_tbs` does not contain a variable given in the
    `mapping`'s keys, a null grid will be genereated for that channel and a
    warning is raised. For example, this happens at least once: on 10/10/1995 SH
    for 22v (from F13).
    """
    remapped_tbs = {}
    for dest_chan, source_chan in mapping.items():
        if source_chan in xr_tbs.variables.keys():
            remapped_tbs[dest_chan] = xr_tbs[source_chan].data
        else:
            # Create null tbs for this channel.
            logger.warning(
                f"WARNING: created NULL values for tb {dest_chan} because the"
                f" provided dataset for {hemisphere=} {resolution=} {data_source=} {date=} does not"
                f" contain the expected source channel {source_chan}"
            )
            remapped_tbs[dest_chan] = get_null_grid(
                hemisphere=hemisphere,
                resolution=resolution,
            )

    ecdr_tbs = EcdrTbs(**remapped_tbs)

    return ecdr_tbs


def _get_am2_tbs(*, date: dt.date, hemisphere: Hemisphere) -> EcdrTbData:
    tb_resolution: Final = "12.5"
    data_source: Final = "AU_SI12"
    try:
        xr_tbs = get_au_si_tbs(
            date=date,
            hemisphere=hemisphere,
            resolution="12",
        )
        ecdr_tbs = map_tbs_to_ecdr_channels(
            mapping=dict(
                v19="v18",
                h19="h18",
                v22="v23",
                v37="v36",
                h37="h36",
            ),
            xr_tbs=xr_tbs,
            hemisphere=hemisphere,
            resolution=tb_resolution,
            date=date,
            data_source=data_source,
        )
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
        data_source=data_source,
        platform="am2",
    )

    return ecdr_tb_data


def _get_ame_tbs(*, date: dt.date, hemisphere: Hemisphere) -> EcdrTbData:
    tb_resolution: Final = "12.5"
    ame_resolution_str = {
        "12.5": "12",
        "25": "25",
    }[tb_resolution]
    ame_resolution_str = cast(AMSR_RESOLUTIONS, ame_resolution_str)
    AME_DATA_DIR = Path("/ecs/DP4/AMSA/AE_SI12.003/")
    data_source: Final = "AE_SI12"
    try:
        xr_tbs = get_ae_si_tbs_from_disk(
            date=date,
            hemisphere=hemisphere,
            data_dir=AME_DATA_DIR,
            resolution=ame_resolution_str,
        )
        ecdr_tbs = map_tbs_to_ecdr_channels(
            # TODO/Note: this mapping is the same as used for `am2`.
            mapping=dict(
                v19="v18",
                h19="h18",
                v22="v23",
                v37="v36",
                h37="h36",
            ),
            xr_tbs=xr_tbs,
            hemisphere=hemisphere,
            resolution=tb_resolution,
            date=date,
            data_source=data_source,
        )
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
        data_source=data_source,
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
    data_source: Final = "NSIDC-0001"
    try:
        xr_tbs_0001 = get_nsidc_0001_tbs_from_disk(
            date=date,
            hemisphere=hemisphere,
            data_dir=NSIDC0001_DATA_DIR,
            resolution=nsidc0001_resolution,
            sat=platform,
        )

        ecdr_tbs = map_tbs_to_ecdr_channels(
            mapping=dict(
                v19="v19",
                h19="h19",
                v22="v22",
                v37="v37",
                h37="h37",
            ),
            xr_tbs=xr_tbs_0001,
            hemisphere=hemisphere,
            resolution=tb_resolution,
            date=date,
            data_source=data_source,
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
        data_source=data_source,
        platform=platform,  # type: ignore[arg-type]
    )

    return ecdr_tb_data


def _get_nsidc_0007_tbs(*, hemisphere: Hemisphere, date: dt.date) -> EcdrTbData:
    NSIDC0007_DATA_DIR = Path("/projects/DATASETS/nsidc0007_smmr_radiance_seaice_v01/")
    # resolution in km
    SMMR_RESOLUTION: Final = "25"
    data_source: Final = "NSIDC-0007"

    try:
        xr_tbs = get_nsidc_0007_tbs_from_disk(
            date=date,
            hemisphere=hemisphere,
            data_dir=NSIDC0007_DATA_DIR,
        )

        ecdr_tbs = map_tbs_to_ecdr_channels(
            mapping=dict(
                v19="v18",
                h19="h18",
                v22="v37",
                v37="v37",
                h37="h37",
            ),
            xr_tbs=xr_tbs,
            hemisphere=hemisphere,
            resolution=SMMR_RESOLUTION,
            date=date,
            data_source=data_source,
        )
    except FileNotFoundError:
        logger.warning(f"Using null TBs for SMMR/n07 on {date}")
        ecdr_tbs = get_null_ecdr_tbs(hemisphere=hemisphere, resolution=SMMR_RESOLUTION)

    ecdr_tb_data = EcdrTbData(
        tbs=ecdr_tbs,
        resolution=SMMR_RESOLUTION,
        data_source=data_source,
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
