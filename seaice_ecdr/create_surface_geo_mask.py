"""Create netCDF file with surface type and geolocation arrays.

create_surface_geo_mask.py
"""

from functools import cache

import numpy as np
import xarray as xr
from scipy.ndimage import zoom

from seaice_ecdr.masks import psn_125_near_pole_hole_mask

GEO_PSN125 = "/projects/DATASETS/nsidc0771_polarstereo_anc_grid_info/NSIDC0771_LatLon_PS_N12.5km_v1.0.nc"
GEO_PSS125 = "/projects/DATASETS/nsidc0771_polarstereo_anc_grid_info/NSIDC0771_LatLon_PS_S12.5km_v1.0.nc"

NSIDC0051_SOURCES = ("smmr", "f08", "f11", "f13", "f17")
SAMPLE_0051_DAILY_NH_NCFN = {
    "smmr": "/ecs/DP1/PM/NSIDC-0051.002/1978.11.03/NSIDC0051_SEAICE_PS_N25km_19781103_v2.0.nc",
    "f08": "/ecs/DP1/PM/NSIDC-0051.002/1989.11.02/NSIDC0051_SEAICE_PS_N25km_19891102_v2.0.nc",
    "f11": "/ecs/DP1/PM/NSIDC-0051.002/1993.11.02/NSIDC0051_SEAICE_PS_N25km_19931102_v2.0.nc",
    "f13": "/ecs/DP1/PM/NSIDC-0051.002/2000.11.02/NSIDC0051_SEAICE_PS_N25km_20001102_v2.0.nc",
    "f17": "/ecs/DP1/PM/NSIDC-0051.002/2020.11.02/NSIDC0051_SEAICE_PS_N25km_20201102_v2.0.nc",
}
ICVARNAME_0051 = {
    "smmr": "N07_ICECON",
    "f08": "F08_ICECON",
    "f11": "F11_ICECON",
    "f13": "F13_ICECON",
    "f17": "F17_ICECON",
}

SURFGEOMASK_PSN125_FILE = "/share/apps/amsr2-cdr/cdrv5_ancillary/surfgeomask_psn12.5.nc"
SURFGEOMASK_PSS125_FILE = "/share/apps/amsr2-cdr/cdrv5_ancillary/surfgeomask_pss12.5.nc"
SURFTYPE_BIN_PSN125_FILE = "/share/apps/amsr2-cdr/cdrv5_ancillary/landmask_psn12.5.dat"
SURFTYPE_BIN_PSS125_FILE = "/share/apps/amsr2-cdr/cdrv5_ancillary/landmask_pss12.5.dat"

SENSOR_LIST = [
    "smmr",
    "f08",
    "f11",
    "f13",
    "f17",
    "amsr2",
]


def have_polehole_inputs(input_type):
    """Verify that the expected input files exist."""
    import os

    if input_type in NSIDC0051_SOURCES:
        return os.path.isfile(SAMPLE_0051_DAILY_NH_NCFN[input_type])
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


def get_polehole_mask(gridid, sensor):
    """Return the polemask for this sensor."""
    if sensor == "amsr2":
        print(f"Generating polehole mask for {sensor}...")
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


if __name__ == "__main__":
    xvar_nh = get_geoarray_coord("psn12.5", "x")
    yvar_nh = get_geoarray_coord("psn12.5", "y")
    xvar_sh = get_geoarray_coord("pss12.5", "x")
    yvar_sh = get_geoarray_coord("pss12.5", "y")
    lat_nh = get_geoarray_field("psn12.5", "latitude")
    lat_sh = get_geoarray_field("pss12.5", "latitude")
    lon_nh = get_geoarray_field("psn12.5", "longitude")
    lon_sh = get_geoarray_field("pss12.5", "longitude")
    crs_nh = get_geoarray_field("psn12.5", "crs")
    crs_sh = get_geoarray_field("pss12.5", "crs")

    polemask_amsr2 = get_polehole_mask("psn12.5", "amsr2")

    land_nh = np.fromfile(SURFTYPE_BIN_PSN125_FILE, dtype=np.uint8).reshape(
        lat_nh.shape
    )
    land_sh = np.fromfile(SURFTYPE_BIN_PSS125_FILE, dtype=np.uint8).reshape(
        lat_sh.shape
    )

    # Create the NH ancillary netCDF file
    ds_nh_ncfn = "cdrv5_surfgeo_psn12.5.nc"
    ds_nh = xr.Dataset(
        data_vars=dict(
            crs=crs_nh,
            x=xvar_nh,
            y=yvar_nh,
            latitude=lat_nh,
            longitude=lon_nh,
        ),
        attrs={
            "comment": "PSN12.5 surface type and geolocation arrays",
        },
    )

    # Set encoding dictionary
    encoding_nh = {
        "latitude": {"zlib": True},
        "longitude": {"zlib": True},
        "surface_type": {"zlib": True},
        "polehole_bitmask": {"zlib": True},
    }

    surftype_nh = land_nh.copy()
    surftype_nh[polemask_amsr2 == 1] = 100
    ds_nh["surface_type"] = xr.DataArray(
        name="surface_type_mask",
        data=surftype_nh,
        dims=["y", "x"],
        attrs={
            "long_name": "nh_surfacetype",
            "flag_values": np.array((50, 75, 100, 200, 250), dtype=np.uint8),
            "flag_meanings": "ocean lake polehole_mask coast land",
        },
    )
    ds_nh["polehole_bitmask"] = get_polehole_bitmask("psn12.5", SENSOR_LIST)
    ds_nh.to_netcdf(ds_nh_ncfn, encoding=encoding_nh)

    # Create the SH ancillary netCDF file
    ds_sh_ncfn = "cdrv5_surfgeo_pss12.5.nc"
    ds_sh = xr.Dataset(
        data_vars=dict(
            crs=crs_sh,
            x=xvar_sh,
            y=yvar_sh,
            latitude=lat_sh,
            longitude=lon_sh,
        ),
        attrs={
            "comment": "PSS12.5 surface type and geolocation arrays",
        },
    )

    surftype_sh = land_sh.copy()
    ds_sh["surface_type"] = xr.DataArray(
        name="surface_type_mask",
        data=surftype_sh,
        dims=["y", "x"],
        attrs={
            "long_name": "sh_surfacetype",
            "flag_values": np.array((50, 75, 200, 250), dtype=np.uint8),
            "flag_meanings": "ocean lake coast land",
        },
    )
    encoding_sh = encoding_nh.copy()
    del encoding_sh["polehole_bitmask"]
    ds_sh.to_netcdf(ds_sh_ncfn, encoding=encoding_sh)
