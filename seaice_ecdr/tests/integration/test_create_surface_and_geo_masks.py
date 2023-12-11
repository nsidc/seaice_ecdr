"""Unit tests for fields included in aggregate files."""

import os

import numpy as np
import xarray as xr

from seaice_ecdr.create_surface_geo_mask import (
    SENSOR_LIST,
    SURFGEOMASK_FILE,
    get_geoarray_coord,
    get_geoarray_field,
    get_polehole_bitmask,
    get_polehole_mask,
    have_geoarray_inputs,
    have_polehole_inputs,
)

GRID_IDS = ["psn12.5", "pss12.5"]


def test_polehole_input_file_availability():
    """Test that necessary input files exist.

    Note: We only run the tests if the input files exist.  E.g., we don't
    expect the Circle CI process to have access to the input files.
    """
    from seaice_ecdr.create_surface_geo_mask import (
        SAMPLE_0051_DAILY_NH_NCFN,
    )

    if os.path.isfile(SAMPLE_0051_DAILY_NH_NCFN["smmr"]):
        assert have_polehole_inputs("smmr")
    if os.path.isfile(SAMPLE_0051_DAILY_NH_NCFN["f08"]):
        assert have_polehole_inputs("f08")
    if os.path.isfile(SAMPLE_0051_DAILY_NH_NCFN["f11"]):
        assert have_polehole_inputs("f11")
    if os.path.isfile(SAMPLE_0051_DAILY_NH_NCFN["f13"]):
        assert have_polehole_inputs("f13")
    if os.path.isfile(SAMPLE_0051_DAILY_NH_NCFN["f17"]):
        assert have_polehole_inputs("f17")


def test_geoarray_input_file_availability():
    """Test that necessary geolocation array input files exist."""
    assert have_geoarray_inputs("psn12.5")
    assert have_geoarray_inputs("pss12.5")


def test_geoarray_coords():
    """Verify extraction of coordinates from geoarrays."""
    coords = ["x", "y"]
    for gridid in GRID_IDS:
        if have_geoarray_inputs(gridid):
            for coord in coords:
                coord = get_geoarray_coord(gridid, coord)
            assert isinstance(coord, type(xr.DataArray()))


def test_geoarray_latlon():
    """Verify extraction of lat and lon from geoarrays."""
    geovars = ["latitude", "longitude"]
    for gridid in GRID_IDS:
        if have_geoarray_inputs(gridid):
            for geovar in geovars:
                geofield = get_geoarray_field(gridid, geovar)
            assert isinstance(geofield, type(xr.DataArray()))


def test_surfgeomask_files():
    """Verify existence of surface and geoarray mask files."""
    for gridid in GRID_IDS:
        surfgeomask_nh_ds = xr.load_dataset(SURFGEOMASK_FILE[gridid])
        assert surfgeomask_nh_ds is not None


def test_get_polehole_mask():
    """Test that each sensor returns a pole hole mask."""

    gridids_to_test = ("psn12.5",)
    sensors_to_test = ("amsr2", "smmr")

    # TODO: Wrap this in a try/except for lack of input files
    for gridid in gridids_to_test:
        for sensor in sensors_to_test:
            polemask = get_polehole_mask(gridid, sensor)
            assert isinstance(polemask, np.ndarray)


def test_get_polehole_bitmask():
    """Test creation of bitmask for  pole hole variable."""
    polehole_bitmask = get_polehole_bitmask("psn12.5", SENSOR_LIST)
    assert polehole_bitmask is not None
