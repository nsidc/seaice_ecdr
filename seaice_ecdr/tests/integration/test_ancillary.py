import datetime as dt
from pathlib import Path
from string import Template
from typing import Final, cast, get_args
from unittest.mock import patch

import numpy as np
import pandas as pd
from pm_tb_data._types import Hemisphere

from seaice_ecdr._types import ECDR_SUPPORTED_RESOLUTIONS
from seaice_ecdr.ancillary import (
    ANCILLARY_SOURCES,
    get_adj123_field,
    get_ancillary_daily_clim_filepath,
    get_ancillary_filepath,
    get_cdr_conc_threshold,
    get_non_ocean_mask,
    get_smmr_invalid_ice_mask,
)
from seaice_ecdr.constants import (
    DEFAULT_ANCILLARY_SOURCE,
    DEFAULT_CDR_RESOLUTION,
    ECDR_PRODUCT_VERSION,
    NSIDC_NFS_SHARE_DIR,
)
from seaice_ecdr.grid_id import get_grid_id
from seaice_ecdr.platforms.config import AM2_PLATFORM, F11_PLATFORM


def test_default_ancillary_source_is_valid():
    assert DEFAULT_ANCILLARY_SOURCE in get_args(ANCILLARY_SOURCES)


def test_get_smmr_invalid_ice_masks():
    for ancillary_source in get_args(ANCILLARY_SOURCES):
        for hemisphere in get_args(Hemisphere):
            icemask = get_smmr_invalid_ice_mask(
                date=dt.date(2023, 1, 29),
                hemisphere=hemisphere,
                resolution="25",
                ancillary_source=ancillary_source,
            )

            assert icemask.dtype == bool
            assert icemask.any()


def test_adj123_does_not_overlap_land():
    test_resolution: Final = "25"

    for hemisphere in get_args(Hemisphere):
        non_ocean_mask = get_non_ocean_mask(
            hemisphere=hemisphere,
            resolution=test_resolution,
            ancillary_source=DEFAULT_ANCILLARY_SOURCE,
        )

        adj123_mask = get_adj123_field(
            hemisphere=hemisphere,
            resolution=test_resolution,
            ancillary_source=DEFAULT_ANCILLARY_SOURCE,
        )

        is_land = non_ocean_mask.data
        is_adj1 = adj123_mask.data == 1
        is_adj2 = adj123_mask.data == 2
        is_adj3 = adj123_mask.data == 3

        """ Sample debugging segment
        try:
            assert not np.any(is_land & is_adj1)
        except AssertionError as e:
            print('adj123 value of 1 overlaps with land:')
            print(f'   ancillary_source: {DEFAULT_ANCILLARY_SOURCE}')
            print(f'         hemisphere: {hemisphere}')
            print(f'         resolution: {test_resolution}')
            breakpoint()
            raise e
        """
        assert not np.any(is_land & is_adj1)
        assert not np.any(is_land & is_adj2)
        assert not np.any(is_land & is_adj3)


def test_ancillary_filepaths():
    """test that the directory/names of the ancillary files
    for publication are as expected"""
    ancillary_source = cast(ANCILLARY_SOURCES, DEFAULT_ANCILLARY_SOURCE)
    resolution = cast(ECDR_SUPPORTED_RESOLUTIONS, DEFAULT_CDR_RESOLUTION)
    hemispheres = get_args(Hemisphere)
    product_version = ECDR_PRODUCT_VERSION

    # expected_dir = Path(f"/share/apps/G02202_V5/{product_version}_ancillary")
    expected_dir = Path(f"{NSIDC_NFS_SHARE_DIR}/{product_version}_ancillary")
    assert expected_dir.is_dir()

    expected_fp_template = Template("G02202-ancillary-${grid_id}-${product_version}.nc")
    expected_daily_fp_template = Template(
        "G02202-ancillary-${grid_id}-daily-invalid-ice-${product_version}.nc"
    )

    for hemisphere in hemispheres:
        grid_id = get_grid_id(
            hemisphere=hemisphere,
            resolution=resolution,
        )

        filename_info = {
            "grid_id": grid_id,
            "product_version": product_version,
        }

        actual_ancillary_filepath = get_ancillary_filepath(
            hemisphere=hemisphere,
            resolution=resolution,
            ancillary_source=ancillary_source,
        )

        expected_fn = expected_fp_template.safe_substitute(filename_info)
        expected_ancillary_filepath = Path(expected_dir) / Path(expected_fn)
        assert expected_ancillary_filepath == actual_ancillary_filepath
        assert actual_ancillary_filepath.is_file()

        actual_daily_ancillary_filepath = get_ancillary_daily_clim_filepath(
            hemisphere=hemisphere,
            resolution=resolution,
            ancillary_source=ancillary_source,
        )
        expected_daily_fn = expected_daily_fp_template.safe_substitute(filename_info)
        expected_daily_ancillary_filepath = Path(expected_dir) / Path(expected_daily_fn)
        assert expected_daily_ancillary_filepath == actual_daily_ancillary_filepath
        assert actual_daily_ancillary_filepath.is_file()


def test_get_cdr_conc_threshold_dmsp_non_leapyear():
    with patch(
        "seaice_ecdr.ancillary.pd.read_csv", side_effect=pd.read_csv
    ) as mock_read_csv:
        threshold = get_cdr_conc_threshold(
            date=dt.date(1995, 1, 1),
            hemisphere="north",
            platform=F11_PLATFORM,
        )

        mock_read_csv.assert_called_once()
        args, _kwargs = mock_read_csv.call_args
        read_csv_filepath = args[0]
        assert read_csv_filepath.name == "nh_final_thresholds.csv"

    assert threshold is not None
    assert isinstance(threshold, float)


def test_get_cdr_conc_threshold_dmsp_leapyear():
    with patch(
        "seaice_ecdr.ancillary.pd.read_csv", side_effect=pd.read_csv
    ) as mock_read_csv:
        threshold = get_cdr_conc_threshold(
            date=dt.date(1992, 5, 12),
            hemisphere="south",
            platform=F11_PLATFORM,
        )

        mock_read_csv.assert_called_once()
        args, _kwargs = mock_read_csv.call_args
        read_csv_filepath = args[0]
        assert read_csv_filepath.name == "sh_final_thresholds-leap-year.csv"

    assert threshold is not None
    assert isinstance(threshold, float)


def test_get_cdr_conc_threshold_am2():
    with patch(
        "seaice_ecdr.ancillary.pd.read_csv", side_effect=pd.read_csv
    ) as mock_read_csv:
        threshold = get_cdr_conc_threshold(
            date=dt.date(2024, 3, 15),
            hemisphere="south",
            platform=AM2_PLATFORM,
        )

        mock_read_csv.assert_not_called()

    # Non-DMSP has a static threshold of 10%
    assert threshold == 10.0
