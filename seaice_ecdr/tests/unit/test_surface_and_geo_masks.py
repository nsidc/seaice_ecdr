"""Unit tests for fields included in aggregate files."""

import sys

import xarray as xr
from loguru import logger

from seaice_ecdr.create_surface_geo_mask import (
    get_geoarray_coord,
    get_geoarray_field,
    have_geoarray_inputs,
    have_polehole_inputs,
)

# Set the default minimum log notification to Warning
# TODO: Think about logging holistically...
try:
    logger.remove(0)  # Removes previous logger info
    logger.add(sys.stderr, level="WARNING")
except ValueError:
    logger.debug(f"Started logging in {__name__}")
    logger.add(sys.stderr, level="WARNING")


def test_polehole_input_file_availability():
    """Test that necessary input files exist.

    Note: We only run the tests if the input files exist.  E.g., we don't
    expect the Circle CI process to have access to the input files.
    """
    import os

    from seaice_ecdr.create_surface_geo_mask import (
        SAMPLE_0051_DAILY_NH_F08,
        SAMPLE_0051_DAILY_NH_F11,
        SAMPLE_0051_DAILY_NH_F13,
        SAMPLE_0051_DAILY_NH_F17,
        SAMPLE_0051_DAILY_NH_SMMR,
    )

    if os.path.isfile(SAMPLE_0051_DAILY_NH_SMMR):
        assert have_polehole_inputs("smmr")
    if os.path.isfile(SAMPLE_0051_DAILY_NH_F08):
        assert have_polehole_inputs("f08")
    if os.path.isfile(SAMPLE_0051_DAILY_NH_F11):
        assert have_polehole_inputs("f11")
    if os.path.isfile(SAMPLE_0051_DAILY_NH_F13):
        assert have_polehole_inputs("f13")
    if os.path.isfile(SAMPLE_0051_DAILY_NH_F17):
        assert have_polehole_inputs("f17")


def test_geoarray_input_file_availability():
    """Test that necessary geolocation array input files exist."""
    import os

    from seaice_ecdr.create_surface_geo_mask import (
        GEO_PSN125,
        GEO_PSS125,
    )

    if os.path.isfile(GEO_PSN125):
        assert have_geoarray_inputs("psn12.5")
    if os.path.isfile(GEO_PSS125):
        assert have_geoarray_inputs("pss12.5")


def test_geoarray_coords():
    """Verify extraction of coordinates from geoarrays."""
    gridids = ["psn12.5", "pss12.5"]
    coords = ["x", "y"]
    for gridid in gridids:
        if have_geoarray_inputs(gridid):
            for coord in coords:
                coord = get_geoarray_coord(gridid, coord)
            assert isinstance(coord, type(xr.DataArray()))


def test_geoarray_latlon():
    """Verify extraction of lat and lon from geoarrays."""
    gridids = ["psn12.5", "pss12.5"]
    geovars = ["latitude", "longitude"]
    for gridid in gridids:
        if have_geoarray_inputs(gridid):
            for geovar in geovars:
                geofield = get_geoarray_field(gridid, geovar)
            assert isinstance(geofield, type(xr.DataArray()))