"""Create 25km ancillary iles for cdrv5.

NOTE: This will calculate the ancillary xr Dataset, but then NOT overwrite
      an existing netCDF file.

Usage:
    python adapt_cdrv5_anc.py psn25  (for Northern Hemisphere)
    python adapt_cdrv5_anc.py pss25  (for Southern Hemisphere)

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
             adj123: <code>
               l90c: <external code>
  min_concentration: ecdr_anc
   invalid_ice_mask: ecdr_anc
   polehole_bitmask: ecdr_anc
"""

import datetime as dt
import os
import warnings

import numpy as np
import xarray as xr
from pm_icecon.land_spillover import create_land90
from scipy.signal import convolve2d

ecdr_anc_fn = {
    # "psn12.5": "/share/apps/G02202_V5/v05r00_ancillary/ecdr-ancillary-psn12.5.nc",
    # "pss12.5": "/share/apps/G02202_V5/v05r00_ancillary/ecdr-ancillary-pss12.5.nc",
    # "psn25": "/share/apps/G02202_V5/v05r00_ancillary/ecdr-ancillary-psn25.nc",
    # "pss25": "/share/apps/G02202_V5/v05r00_ancillary/ecdr-ancillary-pss25.nc",
    # Prefer local versions
    "psn12.5": "./ecdr-ancillary-psn12.5.nc",
    "pss12.5": "./ecdr-ancillary-pss12.5.nc",
    "psn25": "./ecdr-ancillary-psn25.nc",
    "pss25": "./ecdr-ancillary-pss25.nc",
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


def calc_surfacetype_np(ds):
    """Calculate the DataArray for the CDRv5 surface_type given a regions DS.

    Note: Input is a Dataset that we can test for various surfacetype field
          names.
          Output is a numpy array, so that the data can be assigned to a
          DataArray with the overall Datasets coord variables.

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

    return surface_mask


def calc_adj123_np(surftype_da, ocean_val=50, coast_val=200):
    """Compute the land-adjacency field for this surfacetype mask.
    Input:
        DataArray with values:
            ocean: ocean_val (default 50)
            coast: coast_val (default 200)
    Output:
        Numpy array with adjacency values of 1, 2, 3
    """
    surftype = surftype_da.as_numpy()
    is_ocean = surftype == ocean_val
    is_coast = surftype == coast_val
    is_land = (~is_ocean) & (~is_coast)

    kernel = [
        [
            1,
            1,
            1,
        ],
        [1, 1, 1],
        [
            1,
            1,
            1,
        ],
    ]
    adj123_arr = np.zeros(surftype.shape, dtype=np.uint8)
    adj123_arr[is_land] = 255
    for adj_val in range(1, 4):
        is_unlabeled = adj123_arr == 0
        is_labeled = (adj123_arr == 255) | ((~is_unlabeled) & (adj123_arr < adj_val))
        convolved = convolve2d(
            is_labeled.astype(np.uint8),
            kernel,
            mode="same",
            boundary="fill",
            fillvalue=0,
        )
        is_newly_labeled = is_unlabeled & (convolved > 0)
        adj123_arr[is_newly_labeled] = adj_val

    # Remove the land grid cells from the adjacency matrix
    adj123_arr[adj123_arr == 255] = 0

    return adj123_arr


def calc_new_minconc(
    hires_minconc,
    scale_factor,
    adj123,
):
    # Compute min_concentration field
    # NOTE: The min_concentration field will only be defined for ocean grid
    #       cells that are 1, 2, or 3 grid cells away from land
    # NOTE: The methodology used here is an approximation.
    #       The min-concentration field is intended
    #       to represent the minimum concentration observed at a grid cell when
    #       the grid cell has no sea ice.  This approximates what "sea ice
    #       concentration" the nearby land appears to have.  Properly, this
    #       should be re-computed for each sensor and each grid.

    # Create a hires version of the adj123 matrix
    ydim, xdim = adj123.shape
    xdim_f = scale_factor * xdim
    ydim_f = scale_factor * ydim
    adj123_f = np.zeros((ydim_f, xdim_f), dtype=adj123.dtype)
    for joff in range(scale_factor):
        for ioff in range(scale_factor):
            adj123_f[joff::scale_factor, ioff::scale_factor] = adj123.data[:]

    # Re-arrange the high resolution array by the scale factor...
    hires = hires_minconc.data.copy()
    hires[adj123_f == 0] = np.nan
    hires[adj123_f > 3] = np.nan
    xdim_rescaled = hires_minconc.shape[1] // scale_factor
    hires_grouped = hires.reshape(-1, scale_factor, xdim_rescaled, scale_factor)
    with warnings.catch_warnings():
        # Ignore the many regions where there are no values
        warnings.filterwarnings("ignore", r"Mean of empty slice")
        new_minconc = np.nanmean(hires_grouped, (-1, -3))

    new_minconc[np.isnan(new_minconc)] = 0

    return new_minconc


def calc_invalid_ice_mask(
    hires_iim,
    hires_surftype,
    scale_factor,
    surftype,
):
    """Create an invalid ice mask from a higher resolution mask."""
    # Create a hires version of the surftype matrix
    ydim, xdim = surftype.shape
    xdim_f = scale_factor * xdim
    ydim_f = scale_factor * ydim
    surftype_f = np.zeros((ydim_f, xdim_f), dtype=surftype.dtype)
    for joff in range(scale_factor):
        for ioff in range(scale_factor):
            surftype_f[joff::scale_factor, ioff::scale_factor] = surftype.data[:]
    # Convert to float32; set non-ocean to NaN; set invalid to 1 and not to 0
    is_non_ocean = surftype_f != 50
    hires_iim_float = hires_iim.data.astype(np.float32)
    # Set lakes to invalid ice
    n_months, _, _ = hires_iim_float.shape
    new_iim_bool = np.zeros((n_months, ydim, xdim), dtype=bool)
    new_iim_bool[:] = False

    xdim_rescaled = hires_iim_float.shape[-1] // scale_factor
    for m in range(n_months):
        hires_iim_slice = hires_iim_float[m, :, :]
        hires_iim_slice[is_non_ocean] = np.nan
        # Note that the invalid ice mask is set to invalid for lakes
        # but we don't want to allow that here
        hires_iim_slice[hires_surftype.data == 75] = np.nan
        hires_iim_slice[hires_surftype.data == 200] = np.nan

        # Re-arrange the high resolution array by the scale factor...
        hires_grouped = hires_iim_slice.reshape(
            -1, scale_factor, xdim_rescaled, scale_factor
        )
        with warnings.catch_warnings():
            # Ignore the many regions where there are no values
            warnings.filterwarnings("ignore", r"Mean of empty slice")
            new_iim_slice = np.nansum(hires_grouped, (-1, -3))
        new_iim_bool_slice = new_iim_bool[m, :, :]
        new_iim_bool_slice[new_iim_slice > 0] = True

        # Manual correction noticed in QC
        if xdim == 304:
            # NH
            new_iim_bool_slice[359:380, 53:70] = 0
            if (m <= 5) or (m == 12):
                new_iim_bool_slice[297, 295] = 0
                new_iim_bool_slice[302, 297] = 0

    return new_iim_bool


def calc_polehole_bitmask(
    hires_polehole_bitmask_da,
    scale_factor,
):
    """Calculate the lower resolution pole hole bit mask."""
    hires_bitmask = hires_polehole_bitmask_da.data
    ydim, xdim = hires_bitmask.shape
    xdim_lowres = xdim // scale_factor
    ydim_lowres = ydim // scale_factor
    new_polehole_bitmask = np.zeros(
        (ydim_lowres, xdim_lowres), dtype=hires_bitmask.dtype
    )
    bitmask_values = hires_polehole_bitmask_da.attrs["flag_masks"]
    for val in bitmask_values:
        for joff in range(scale_factor):
            for ioff in range(scale_factor):
                subset = hires_bitmask[joff::scale_factor, ioff::scale_factor]
                is_val = np.bitwise_and(subset, val)
                new_polehole_bitmask[is_val > 0] = np.bitwise_or(
                    new_polehole_bitmask[is_val > 0], val
                )

    return new_polehole_bitmask


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
            month=(
                ["month"],
                months,
                {
                    "long_name": "month of the year",
                },
            ),
            x=ds_latlon.variables["x"],
            y=ds_latlon.variables["y"],
        ),
        # TODO: The attributes of the ancillary Dataset should be standardized
        #       and consistent for the ancillary files for all grids.
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
    # Ensure that dimensions do not have a _FillValue
    ds.x.encoding["_FillValue"] = None
    ds.y.encoding["_FillValue"] = None

    # Remove "missing_value" attribute because it has been deprecated
    # Note: this is an encoding, not an attribute
    ds.latitude.encoding["missing_value"] = None
    ds.longitude.encoding["missing_value"] = None

    # Add in the region values
    ds_regions = xr.load_dataset(region_fn)
    surfacetype_np = calc_surfacetype_np(ds_regions)
    ds["surface_type"] = xr.DataArray(
        name="surface_type",
        data=surfacetype_np,
        dims=["y", "x"],
        coords=dict(
            y=ds.variables["y"],
            x=ds.variables["x"],
        ),
        # TODO: The attributes of surface_type should be standardized
        #       and consistent for the ancillary files for all grids.
        attrs=dict(
            long_name=f"{gridid}_surfacetype",
            grid_mapping="crs",
            flag_values=np.uint8((50, 75, 200, 250)),
            flag_meanings="ocean lake coast land",
        ),
    )

    # Find the land-adjacency matrix which indicates whether grid cells are
    #  1, 2, or 3 grid cells away from land (coast).
    adj123_np = calc_adj123_np(ds["surface_type"])
    ds["adj123"] = xr.DataArray(
        name="adj123",
        data=adj123_np,
        dims=["y", "x"],
        coords=dict(
            y=ds.variables["y"],
            x=ds.variables["x"],
        ),
        # TODO: The attributes of adj123 should be standardized
        #       and consistent for the ancillary files for all grids.
        attrs=dict(
            short_name="adj123",
            long_name=f"{gridid}_adjacency_field",
            grid_mapping="crs",
            flag_values=np.uint8((0, 1, 2, 3)),
            flag_meanings="not_near_land one_gridcell_from_land two_gridcells_from_land three_gridcell_from_land",
        ),
    )

    # Calculate the land90conc field
    l90c_np = create_land90(adj123=ds["adj123"].data)
    ds["l90c"] = xr.DataArray(
        name="l90c",
        data=l90c_np,
        dims=["y", "x"],
        coords=dict(
            y=ds.variables["y"],
            x=ds.variables["x"],
        ),
        # TODO: The attributes of l90c should be standardized
        #       and consistent for the ancillary files for all grids.
        attrs=dict(
            short_name="l90c",
            long_name=f"{gridid}_land-as-90-percent-concentration_field",
            grid_mapping="crs",
            comment="The 'land90' array is a mock sea ice concentration"
            " array that is calculated from the land mask.  It assumes that"
            " the mock concentration value will be the average of a 7x7 array"
            " of local surface mask values centered on the center pixel."
            "  Water grid cells are considered to have a sea ice concentration"
            " of zero.  Land grid cells are considered to have a sea"
            " ice concentration of 90%.  The average of the 49 grid cells"
            " in the 7x7 array yields the `land90` concentration value.",
        ),
    )

    # Downscale fields from 12.5km fields for consistency
    # Downscale:  min_concentration is mean of underlying min_concentration vals
    # Downscale:  invalid_ice_mask is True if any underlying val is True
    # Downscale:  polehole_bitmask has bit set if any underlying bit is set
    # WIP: Here: want to add minconc, inval, polemask derived from 12.5km file
    ds_hires_anc = xr.load_dataset(hires_anc_fn)

    # Compute min_concentration field
    # NOTE: The min_concentration field will only be defined for ocean grid
    #       cells that are 1, 2, or 3 grid cells away from land
    # NOTE: The methodology used here is an approximation.
    #       The min-concentration field is intended
    #       to represent the minimum concentration observed at a grid cell when
    #       the grid cell has no sea ice.  This approximates what "sea ice
    #       concentration" the nearby land appears to have.  Properly, this
    #       should be re-computed for each sensor and each grid.
    new_minconc_np = calc_new_minconc(
        ds_hires_anc.data_vars["min_concentration"],
        scale_factor,
        ds.data_vars["adj123"],
    )
    ds["min_concentration"] = xr.DataArray(
        name="min_concentration",
        data=new_minconc_np,
        dims=["y", "x"],
        coords=dict(
            y=ds.variables["y"],
            x=ds.variables["x"],
        ),
        # TODO: The attributes of min_concentration should be standardized
        #       and consistent for the ancillary files for all grids.
        attrs=dict(
            short_name="min_concentration",
            long_name=f"{gridid} minimum observed NASATeam seaice concentration",
            grid_mapping="crs",
            comment="The 'min_concentration' array is the minimum sea ice"
            " concentration value observed at near-land pixels during"
            " no-sea-ice conditions.  It represents the amount of apparent"
            " sea ice concentration attributable to land spillover effect.",
        ),
    )

    # Set up the invalid ice mask
    # This will be: we want invalid ice if any of the underlying
    # ocean grid cells are invalid
    invalid_ice_mask_np = calc_invalid_ice_mask(
        ds_hires_anc.data_vars["invalid_ice_mask"],
        ds_hires_anc.data_vars["surface_type"],
        scale_factor,
        ds.data_vars["surface_type"],
    )
    ds["invalid_ice_mask"] = xr.DataArray(
        name="invalid_ice_mask",
        data=invalid_ice_mask_np,
        dims=["month", "y", "x"],
        coords=dict(
            month=ds.variables["month"],
            y=ds.variables["y"],
            x=ds.variables["x"],
        ),
        # TODO: The attributes of invalid_ice_mask should be standardized
        #       and consistent for the ancillary files for all grids.
        attrs=dict(
            short_name="invalid_ice_mask",
            long_name=f"{gridid} invalid sea ice mask",
            grid_mapping="crs",
            flag_values=np.uint8((0, 1)),
            flag_meanings="valid_seaice_location invalid_seaice_location",
            valid_range=np.uint8((0, 1)),
            units="1",
            comment="Mask indicating where seaice will not exist on this day based on climatology",
        ),
    )

    # Set up the pole hole bitmask
    # This will be: we want the bit mask set if any underlying pole hole bit
    # is set
    if "polehole_bitmask" in ds_hires_anc.data_vars.keys():
        polehole_bitmask_np = calc_polehole_bitmask(
            ds_hires_anc.data_vars["polehole_bitmask"],
            scale_factor,
        )
        ds["polehole_bitmask"] = xr.DataArray(
            name="polehole_bitmask",
            data=polehole_bitmask_np,
            dims=["y", "x"],
            coords=dict(
                y=ds.variables["y"],
                x=ds.variables["x"],
            ),
            # TODO: The attributes of polehole_bitmask should be standardized
            #       and consistent for the ancillary files for all grids.
            attrs=dict(
                short_name="polehole_bitmask",
                long_name=f"{gridid} polehole_bitmask",
                grid_mapping="crs",
                flag_masks=ds_hires_anc.data_vars["polehole_bitmask"].attrs[
                    "flag_masks"
                ],
                flag_meanings=ds_hires_anc.data_vars["polehole_bitmask"].attrs[
                    "flag_meanings"
                ],
                valid_range=np.uint8(
                    (
                        0,
                        np.sum(
                            ds_hires_anc.data_vars["polehole_bitmask"].attrs[
                                "flag_masks"
                            ]
                        )
                        - 1,
                    )
                ),
            ),
        )

    # Write out ancillary file
    if os.path.isfile(newres_anc_fn):
        raise RuntimeError(f"Output file exists, aborting\n{newres_anc_fn}")

    ds.to_netcdf(newres_anc_fn)
    print(f"Wrote new {gridid} ancillary file to: {newres_anc_fn}")


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
