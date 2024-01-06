"""Create 25km ancillary iles for cdrv5.

The sources for this information are:
    cdrv5 12.5km files [ecdr_anc]
        ecdr-ancillary-psn12.5.nc
        ecdr-ancillary-pss12.5.nc
    nsidc0771 files [latlon]
        NSIDC0771_LatLon_PS_N25km_v1.0.nc
        NSIDC0771_LatLon_PS_S25km_v1.0.nc
    nsidc0780 files [regions]
        NSIDC-0780_SeaIceRegions_PS-N25km_v1.0.nc
        NSIDC-0780_SeaIceRegions_PS-S25km_v1.0.nc
    code used to derive repeatedly used fields from the surfacetype mask

Specifically:
                crs: latlon
              month: <generated here>
                  x: latlon
                  y: latlon
           latitude: latlon
          longitude: latlon
       surface_type: regions
  min_concentration: ecdr_anc
   invalid_ice_mask: ecdr_anc
   polehole_bitmask: ecdr_anc
               l90c: <external code>
             adj123: <external code>
"""

import datetime as dt

import numpy as np
import xarray as xr

ecdr_anc_fn = {
    "psn12.5": "/share/apps/G02202_V5/v05r00_ancillary/ecdr-ancillary-psn12.5.nc",
    "pss12.5": "/share/apps/G02202_V5/v05r00_ancillary/ecdr-ancillary-pss12.5.nc",
    "psn25": "/share/apps/G02202_V5/v05r00_ancillary/ecdr-ancillary-psn25.nc",
    "pss25": "/share/apps/G02202_V5/v05r00_ancillary/ecdr-ancillary-pss25.nc",
}

nsidc0771_fn = {
    "psn12.5": "/projects/DATASETS/nsidc0771_polarstereo_anc_grid_info/NSIDC0771_LatLon_PS_N12.5km_v1.0.nc",
    "pss12.5": "/projects/DATASETS/nsidc0771_polarstereo_anc_grid_info/NSIDC0771_LatLon_PS_S12.5km_v1.0.nc",
    "psn25": "/projects/DATASETS/nsidc0771_polarstereo_anc_grid_info/NSIDC0771_LatLon_PS_N25km_v1.0.nc",
    "pss25": "/projects/DATASETS/nsidc0771_polarstereo_anc_grid_info/NSIDC0771_LatLon_PS_S25km_v1.0.nc",
}

nsidc0780_fn = {
    "psn12.5": "/projects/DATASETS/nsidc0780_seaice_masks_v1/netcdf/NSIDC-0780_SeaIceRegions_PS-N12.5km_v1.0.nc",
    "pss12.5": "/projects/DATASETS/nsidc0780_seaice_masks_v1/netcdf/NSIDC-0780_SeaIceRegions_PS-S12.5km_v1.0.nc",
    "psn25": "/projects/DATASETS/nsidc0780_seaice_masks_v1/netcdf/NSIDC-0780_SeaIceRegions_PS-N25km_v1.0.nc",
    "pss25": "/projects/DATASETS/nsidc0780_seaice_masks_v1/netcdf/NSIDC-0780_SeaIceRegions_PS-S25km_v1.0.nc",
}


def find_coast(mask, ocean_values=(50,), land_values=(75, 250)):
    """Find indices where land values are adjacent to ocean values."""
    from scipy.signal import convolve2d

    is_ocean = np.zeros(mask.shape, dtype=bool)
    for val in ocean_values:
        is_ocean[mask == val] = True

    is_land = np.zeros(mask.shape, dtype=bool)
    for val in land_values:
        is_land[mask == val] = True

    kernel = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    convolved = convolve2d(
        is_ocean.astype(np.uint8), kernel, mode="same", boundary="fill", fillvalue=0
    )
    is_coast = is_land & (convolved > 0)

    return is_coast


def calc_surfacetype_da(ds):
    """Calculate the DataArray for the CDRv5 surface_type given a regions DS.

    For the CDRv5 field, the ancillary surface types are:
        50: ocean
        75: lake
        200: coast
        250: land
    In the NSIDC-0780 files, the surface mask encodings are:
        0-20: (not all values exist)          ocean
        30: land ->               land
        32: fresh_free_water ->         lake
        33: ice_on_land ->        land
        34: floating_ice_shelf -> land
        35: ocean_disconnected ->       lake
    The coast value must be calculated from the adjacency of land and ocean
    """
    possible_mask_varnames = (
        "sea_ice_region_surface_mask",  # NSIDC-0780 NH files
        "sea_ice_region_NASA_surface_mask",  # NSIDC-0780 SH files
        "sea_ice_region_RH_surface_mask",  # NSIDC-0780 SH files
    )
    mask_varname = ""
    for possible_varname in possible_mask_varnames:
        if possible_varname in ds.data_vars.keys():
            mask_varname = possible_varname
            break

    print(f"mask_varname: {mask_varname}")
    da_regions = ds.data_vars[mask_varname]
    regions_data = da_regions.data
    is_regions_ocean = regions_data <= 20
    is_regions_lake = (regions_data == 32) | (regions_data == 35)
    is_regions_land = (regions_data == 30) | (regions_data == 33) | (regions_data == 34)
    surface_mask = np.zeros(regions_data.shape, dtype=np.uint8)
    surface_mask[is_regions_ocean] = 50
    surface_mask[is_regions_lake] = 75
    surface_mask[is_regions_land] = 250

    # Now, find the coast values
    is_coast = find_coast(
        surface_mask,
        ocean_values=(50,),
        land_values=(75, 250),
    )
    surface_mask[is_coast] = 200
    breakpoint()  # here


def generate_ecdr_anc_file(gridid):
    """Create the cdrv5 ancillary ile for this GridId.
    Note that this may use pre-existing cdrv5 ancillary fields
    as will as other primary sources."""
    latlon_fn = nsidc0771_fn[gridid]
    region_fn = nsidc0780_fn[gridid]
    if gridid == "psn25":
        hires_anc_fn = ecdr_anc_fn["psn12.5"]
        newres_anc_fn = ecdr_anc_fn["psn25"]
        scale_factor = 2
    elif gridid == "pss25":
        hires_anc_fn = ecdr_anc_fn["pss12.5"]
        newres_anc_fn = ecdr_anc_fn["pss25"]
        scale_factor = 2
    else:
        raise RuntimeError(f"GridID not implemented: {gridid}")

    # Set up a new xarray Dataset for the new ancillary file
    # with values from an authoritative geolocation source
    # Coords added: month, x, y
    # Variables added: crs
    # Attributes added: description and geo-informative attrs
    ds_latlon = xr.load_dataset(latlon_fn)
    months = np.arange(1, 13).astype(np.int16)
    ds = xr.Dataset(
        data_vars=dict(
            crs=ds_latlon.variables["crs"],
            latitude=ds_latlon.variables["latitude"],
            longitude=ds_latlon.variables["longitude"],
        ),
        coords=dict(
            month=(["month"], months),
            x=ds_latlon.variables["x"],
            y=ds_latlon.variables["y"],
        ),
        attrs=dict(
            title="Ancillary Information for Polar Stereo 25km Grid used in CDRv5",
            date_created=f"{dt.date.today()}",
            date_modified=f"{dt.date.today()}",
            date_metadata_modified=f"{dt.date.today()}",
            keywords="EARTH SCIENCE SERVICES > DATA ANALYSIS AND VISUALIZATION > GEOGRAPHIC INFORMATION SYSTEMS",
            Conventions="CF-1.10, ACDD-1.3",
            cdm_data_type="grid",
            geospatial_bounds=ds_latlon.attrs["geospatial_bounds"],
            geospatial_bounds_crs=ds_latlon.attrs["geospatial_bounds_crs"],
            geospatial_x_units=ds_latlon.attrs["geospatial_x_units"],
            geospatial_y_units=ds_latlon.attrs["geospatial_y_units"],
            geospatial_x_resolution=ds_latlon.attrs["geospatial_x_resolution"],
            geospatial_y_resolution=ds_latlon.attrs["geospatial_y_resolution"],
            geospatial_lat_min=ds_latlon.attrs["geospatial_lat_min"],
            geospatial_lat_max=ds_latlon.attrs["geospatial_lat_max"],
            geospatial_lon_min=ds_latlon.attrs["geospatial_lon_min"],
            geospatial_lon_max=ds_latlon.attrs["geospatial_lon_max"],
            geospatial_lat_units=ds_latlon.attrs["geospatial_lat_units"],
            geospatial_lon_units=ds_latlon.attrs["geospatial_lon_units"],
        ),
    )

    # Add in the region values
    ds_regions = xr.load_dataset(region_fn)
    surfacetype_np = calc_surfacetype_da(ds_regions)
    ds["surface_type"] = xr.DataArray(
        name="surface_type",
        data=surfacetype_np,
        dims=["y", "x"],
        coords=dict(
            y=ds.variables["y"],
            x=ds.variables["x"],
        ),
        attrs=dict(
            long_name=f"{gridid}_surfacetype",
            flag_values=np.uint8((50, 75, 200, 250)),
            flag_meanings="ocean lake coast land",
        ),
    )

    # Downscale fields from 12.5km fields for consistency
    # Downscale:  min_concentration is max of underlying min_concentration vals
    # Downscale:  invalid_ice_mask is True if any underlying val is True
    # Downscale:  polehole_bitmask has bit set if any underlying bit is set
    # WIP: Here: want to add minconc, inval, polemask derived from 12.5km file
    ds_hires_anc = xr.load_dataset(hires_anc_fn)

    assert scale_factor == 2

    # Write out ancillary file
    ds_hires_anc.to_netcdr(newres_anc_fn)


if __name__ == "__main__":
    import sys

    supported_gridids = ("psn25", "pss25")

    try:
        gridid = sys.argv[1]
        assert gridid in supported_gridids
    except IndexError:
        raise RuntimeError(
            "No GridID provided." f"\n  Valid values: {supported_gridids}"
        )
    except AssertionError:
        raise RuntimeError(
            "GridID {gridid} not recognized." f"\n  Valid values: {supported_gridids}"
        )

    generate_ecdr_anc_file(gridid)
