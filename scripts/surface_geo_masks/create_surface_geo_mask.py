"""Create netCDF file with surface type and geolocation arrays.

By default, both NH and SH ancillary files are created.
Sample usages:
    python seaice_ecdr/create_surface_geo_mask.py
    python seaice_ecdr/create_surface_geo_mask.py north
    python seaice_ecdr/create_surface_geo_mask.py south
"""

import os
import sys
from functools import cache

import numpy as np
import xarray as xr
from scipy.ndimage import zoom

from seaice_ecdr.masks import psn_125_near_pole_hole_mask
from seaice_ecdr.use_surface_geo_mask import get_surfacegeomask_filepath

nh_gridids = ("psn12.5",)

GEO_INFO_FILE = {
    "psn12.5": "/projects/DATASETS/nsidc0771_polarstereo_anc_grid_info/NSIDC0771_LatLon_PS_N12.5km_v1.0.nc",
    "pss12.5": "/projects/DATASETS/nsidc0771_polarstereo_anc_grid_info/NSIDC0771_LatLon_PS_S12.5km_v1.0.nc",
}
SURFGEOMASK_FILE = {
    "psn12.5": get_surfacegeomask_filepath("psn12.5"),
    "pss12.5": get_surfacegeomask_filepath("pss12.5"),
}

SURFTYPE_BIN_FILE = {
    "psn12.5": "/share/apps/amsr2-cdr/cdrv5_ancillary/landmask_psn12.5.dat",
    "pss12.5": "/share/apps/amsr2-cdr/cdrv5_ancillary/landmask_pss12.5.dat",
}

# Note: SENSOR_LIST includes non-0051, eg am2
SENSOR_LIST = [
    "smmr",
    "f08",
    "f11",
    "f13",
    "f17",
    "am2",
]

NSIDC0051_SOURCES = ("smmr", "f08", "f11", "f13", "f17")
ICVARNAME_0051 = {
    "smmr": "N07_ICECON",
    "f08": "F08_ICECON",
    "f11": "F11_ICECON",
    "f13": "F13_ICECON",
    "f17": "F17_ICECON",
}

SAMPLE_0051_DAILY_NH_NCFN = {
    "smmr": "/ecs/DP1/PM/NSIDC-0051.002/1978.11.03/NSIDC0051_SEAICE_PS_N25km_19781103_v2.0.nc",
    "f08": "/ecs/DP1/PM/NSIDC-0051.002/1989.11.02/NSIDC0051_SEAICE_PS_N25km_19891102_v2.0.nc",
    "f11": "/ecs/DP1/PM/NSIDC-0051.002/1993.11.02/NSIDC0051_SEAICE_PS_N25km_19931102_v2.0.nc",
    "f13": "/ecs/DP1/PM/NSIDC-0051.002/2000.11.02/NSIDC0051_SEAICE_PS_N25km_20001102_v2.0.nc",
    "f17": "/ecs/DP1/PM/NSIDC-0051.002/2020.11.02/NSIDC0051_SEAICE_PS_N25km_20201102_v2.0.nc",
}


def have_polehole_inputs(input_type):
    """Verify that the expected input files exist."""
    if input_type in NSIDC0051_SOURCES:
        return os.path.isfile(SAMPLE_0051_DAILY_NH_NCFN[input_type])
    elif input_type == "am2":
        # This is the function for the AM2 mask
        return psn_125_near_pole_hole_mask is not None

    raise RuntimeWarning(f"could not check polehole input for: {input_type}")


def have_geoarray_inputs(gridid):
    """Verify that geolocation files exist for this grid."""
    try:
        return os.path.isfile(GEO_INFO_FILE[gridid])
    except KeyError:
        raise RuntimeError(f"No geo_info_file found for: {gridid}")


@cache
def open_geoarray_ds(gridid):
    """Return the geolocation dataset for this gridid."""
    try:
        return xr.load_dataset(GEO_INFO_FILE[gridid])
    except KeyError:
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


def get_polehole_mask(gridid, sensor):
    """Return the polemask for this sensor."""
    if sensor == "am2":
        polemask_data = psn_125_near_pole_hole_mask()
    elif sensor in NSIDC0051_SOURCES:
        ds0051 = xr.load_dataset(
            SAMPLE_0051_DAILY_NH_NCFN[sensor],
            mask_and_scale=False,
        )
        icvarname = ICVARNAME_0051[sensor]
        siconc_25km = ds0051.variables[icvarname]
        flag_meanings_arr = siconc_25km.attrs["flag_meanings"].split(" ")
        pole_value_index = flag_meanings_arr.index("pole_hole_mask")
        pole_flag_value = siconc_25km.attrs["flag_values"][pole_value_index]
        is_polehole_25km = np.squeeze(siconc_25km.data == pole_flag_value)
        polemask_data = zoom(is_polehole_25km, 2, order=0)

    return polemask_data.astype(np.uint8)


def get_polehole_bitmask(
    gridid,
    sensor_list,
):
    """Return DataArray of polehole bit mask."""
    xvar = get_geoarray_coord(gridid, "x")
    yvar = get_geoarray_coord(gridid, "y")
    bitmask = np.zeros((yvar.shape[0], xvar.shape[0]), dtype=np.uint8)
    bit_value = 1
    flag_values_list = []
    flag_meanings = []
    for sensor in sensor_list:
        flag_values_list.append(bit_value)

        bit_name = f"{sensor}_polemask"
        flag_meanings.append(bit_name)

        polemask = get_polehole_mask(gridid, sensor)
        bitmask[polemask != 0] += bit_value

        # Increment the bit_value for the next pass through the loop
        bit_value = bit_value * 2

    assert len(flag_meanings) <= 8

    flag_values = np.array(flag_values_list, dtype=np.uint8)
    flag_meanings_str = " ".join(flag_meaning for flag_meaning in flag_meanings)
    flag_values_sum = np.sum(flag_values)

    bitmask_da = xr.DataArray(
        name="polehole_bitmask",
        data=bitmask,
        dims=["y", "x"],
        attrs={
            "long_name": "polehole_bitmask",
            "flag_values": flag_values,
            "flag_meanings": flag_meanings_str,
            "valid_range": np.array((0, flag_values_sum), dtype=np.uint8),
        },
    )

    return bitmask_da


def create_surfgeo_anc_file(gridid, ds_ncfn):
    """Create the surfgeo ancillary for a given gridid."""

    print(f"Creating surfgeo ancillary field for gridid: {gridid}")
    print(f"  {ds_ncfn}")

    xvar = get_geoarray_coord(gridid, "x")
    yvar = get_geoarray_coord(gridid, "y")
    latvar = get_geoarray_field(gridid, "latitude")
    lonvar = get_geoarray_field(gridid, "longitude")
    crsvar = get_geoarray_field(gridid, "crs")

    land = np.fromfile(SURFTYPE_BIN_FILE[gridid], dtype=np.uint8).reshape(latvar.shape)

    ds = xr.Dataset(
        data_vars=dict(
            crs=crsvar,
            x=xvar,
            y=yvar,
            latitude=latvar,
            longitude=lonvar,
        ),
        attrs={
            "comment": f"{gridid} surface type and geolocation arrays",
        },
    )

    encoding = {
        "latitude": {"zlib": True},
        "longitude": {"zlib": True},
        "surface_type": {"zlib": True},
    }

    surftype_flag_values_arr = None
    surftype_flag_meanings_str = None
    if gridid in nh_gridids:
        ds["polehole_bitmask"] = get_polehole_bitmask(gridid, SENSOR_LIST)
        encoding["polehole_bitmask"] = {"zlib": True}
    surftype_flag_values_arr = np.array((50, 75, 200, 250), dtype=np.uint8)
    surftype_flag_meanings_str = "ocean lake coast land"

    surftype = land.copy()
    ds["surface_type"] = xr.DataArray(
        name="surface_type_mask",
        data=surftype,
        dims=["y", "x"],
        attrs={
            "long_name": f"{gridid}_surfacetype",
            "flag_values": surftype_flag_values_arr,
            "flag_meanings": surftype_flag_meanings_str,
        },
    )

    ds.to_netcdf(ds_ncfn, encoding=encoding)


if __name__ == "__main__":
    gridid_mapping = {
        "psn12.5": "psn12.5",
        "nh": "psn12.5",
        "north": "psn12.5",
        "pss12.5": "pss12.5",
        "sh": "pss12.5",
        "south": "pss12.5",
    }

    if len(sys.argv) == 1:
        # No cmdline args, assume want both
        gridid = gridid_mapping["nh"]
        create_surfgeo_anc_file(
            gridid,
            SURFGEOMASK_FILE[gridid],
        )
        gridid = gridid_mapping["sh"]
        create_surfgeo_anc_file(
            gridid,
            SURFGEOMASK_FILE[gridid],
        )
    else:
        for arg in sys.argv[1:]:
            gridid = gridid_mapping[arg]
            create_surfgeo_anc_file(
                gridid,
                SURFGEOMASK_FILE[gridid],
            )
