"""Create netCDF file with surface type and geolocation arrays.

create_surface_geo_mask.py
"""

from functools import cache

import xarray as xr

from seaice_ecdr.masks import psn_125_near_pole_hole_mask

GEO_PSN125 = "/projects/DATASETS/nsidc0771_polarstereo_anc_grid_info/NSIDC0771_LatLon_PS_N12.5km_v1.0.nc"
GEO_PSS125 = "/projects/DATASETS/nsidc0771_polarstereo_anc_grid_info/NSIDC0771_LatLon_PS_S12.5km_v1.0.nc"
SAMPLE_0051_DAILY_NH_SMMR = (
    "/ecs/DP1/PM/NSIDC-0051.002/1978.11.02/NSIDC0051_SEAICE_PS_N25km_19781102_v2.0.nc"
)
SAMPLE_0051_DAILY_NH_F08 = (
    "/ecs/DP1/PM/NSIDC-0051.002/1989.11.02/NSIDC0051_SEAICE_PS_N25km_19891102_v2.0.nc"
)
SAMPLE_0051_DAILY_NH_F11 = (
    "/ecs/DP1/PM/NSIDC-0051.002/1989.11.02/NSIDC0051_SEAICE_PS_N25km_19891102_v2.0.nc"
)
SAMPLE_0051_DAILY_NH_F13 = (
    "/ecs/DP1/PM/NSIDC-0051.002/1989.11.02/NSIDC0051_SEAICE_PS_N25km_19891102_v2.0.nc"
)
SAMPLE_0051_DAILY_NH_F17 = (
    "/ecs/DP1/PM/NSIDC-0051.002/1989.11.02/NSIDC0051_SEAICE_PS_N25km_19891102_v2.0.nc"
)


def have_polehole_inputs(input_type):
    """Verify that the expected input files exist."""
    import os

    if input_type == "smmr":
        return os.path.isfile(SAMPLE_0051_DAILY_NH_SMMR)
    elif input_type == "f08":
        return os.path.isfile(SAMPLE_0051_DAILY_NH_F08)
    elif input_type == "f11":
        return os.path.isfile(SAMPLE_0051_DAILY_NH_F11)
    elif input_type == "f13":
        return os.path.isfile(SAMPLE_0051_DAILY_NH_F13)
    elif input_type == "f17":
        return os.path.isfile(SAMPLE_0051_DAILY_NH_F17)
    elif input_type == "amsr2":
        # This is the function for the AMSR2 mask
        return psn_125_near_pole_hole_mask is not None

    raise RuntimeWarning(f"could not check polehole input for: {input_type}")


def have_geoarray_inputs(gridid):
    """Verify that geolocation files exist for this grid."""
    import os

    if gridid == "psn12.5":
        return os.path.isfile(GEO_PSN125)
    elif gridid == "pss12.5":
        return os.path.isfile(GEO_PSS125)


@cache
def open_geoarray_ds(gridid):
    """Return the geolocation dataset for this gridid."""
    if gridid == "psn12.5":
        return xr.load_dataset(GEO_PSN125)
    elif gridid == "pss12.5":
        return xr.load_dataset(GEO_PSS125)
    else:
        raise RuntimeWarning(f"Do not know how to open geoarray for {gridid}")


@cache
def get_geoarray_field(gridid, field_name):
    """Return lat and lon fields for gridid as DataArrays."""
    if not have_geoarray_inputs(gridid):
        raise RuntimeWarning(f"No geoarray input files for: {gridid}")

    ds_geo = open_geoarray_ds(gridid)
    geoarray = ds_geo.data_vars[field_name]

    return geoarray


def get_geoarray_coord(gridid, coord_name):
    """Return x or y for geoarray."""
    ds = open_geoarray_ds(gridid)
    try:
        return ds["latitude"][coord_name]
    except NameError:
        return None


if __name__ == "__main__":
    xvar_nh = get_geoarray_coord("psn12.5", "x")
    xvar_nh = get_geoarray_coord("psn12.5", "y")
    xvar_sh = get_geoarray_coord("pss12.5", "x")
    xvar_sh = get_geoarray_coord("pss12.5", "y")
    lat_nh = get_geoarray_field("psn12.5", "latitude")
    lat_sh = get_geoarray_field("pss12.5", "latitude")
    lon_nh = get_geoarray_field("psn12.5", "longitude")
    lon_sh = get_geoarray_field("pss12.5", "longitude")

    breakpoint()
