"""Fix the 25km v5 ancillary files.

NOTE: This will calculate the ancillary xr Dataset, but then NOT overwrite
      an existing netCDF file.

NOTE: This assumes that the old files are:
    - misnamed
    - have the proper surface_type field
    - have WRONG adj123 field
    - have WRONG l90c (land-as-90% siconc) field
    - have _FillValue for 

Usage:
    python fix_cdrv5_anc.py psn25  (for Northern Hemisphere)
        input:
            ecdr-ancillary-psn25.nc
        output:
            g02202-ancillary-psn25-v05r00.nc

    python fix_cdrv5_anc.py pss25  (for Southern Hemisphere)
        input:
            ecdr-ancillary-pss25.nc
        output:
            g02202-ancillary-pss25-v05r00.nc

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

import os

import numpy as np
import xarray as xr
from pm_icecon.land_spillover import create_land90
from scipy.signal import convolve2d

old_ecdr_anc_fn = {
    "psn12.5": "/share/apps/G02202_V5/v05r00_ancillary/ecdr-ancillary-psn12.5.nc",
    "pss12.5": "/share/apps/G02202_V5/v05r00_ancillary/ecdr-ancillary-pss12.5.nc",
    "psn25": "./ecdr-ancillary-psn25.nc",
    "pss25": "./ecdr-ancillary-pss25.nc",
    # "psn25": "/share/apps/G02202_V5/v05r00_ancillary/ecdr-ancillary-psn25_new.nc",
    # "pss25": "/share/apps/G02202_V5/v05r00_ancillary/ecdr-ancillary-pss25_new.nc",
}

new_ecdr_anc_fn = {
    "psn12.5": "/share/apps/G02202_V5/v05r00_ancillary/ecdr-ancillary-psn12.5.nc",
    "pss12.5": "/share/apps/G02202_V5/v05r00_ancillary/ecdr-ancillary-pss12.5.nc",
    "psn25": "./g02202-ancillary-psn25-v05r00.nc",
    "pss25": "./g02202-ancillary-pss25-v05r00.nc",
    # "psn25": "/share/apps/G02202_V5/v05r00_ancillary/ecdr-ancillary-psn25_new.nc",
    # "pss25": "/share/apps/G02202_V5/v05r00_ancillary/ecdr-ancillary-pss25_new.nc",
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


def calc_adj123_np(surftype_da, ocean_val=50, coast_val=200):
    """Compute the land-adjacency field for this surfacetype mask.
    Input:
        DataArray with values:
            ocean: ocean_val (default 50)
            coast: coast_val (default 200)
    Output:
        Numpy array with adjacency values of 1, 2, 3
    """
    surftype = surftype_da.to_numpy()
    is_ocean = surftype == ocean_val
    is_land = ~is_ocean

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
    adj123_arr[~is_ocean] = 255
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

    # Swap land and unassigned
    # so that land has value of 0 and unlabeled ocean has value of 255
    is_unassigned_ocean = (adj123_arr == 0) & is_ocean

    adj123_arr[is_land] = 0
    adj123_arr[is_unassigned_ocean] = 255

    return adj123_arr


def fix_ecdr_anc_file(gridid):
    """Create the cdrv5 ancillary ile for this GridId.
    Note that this may use pre-existing cdrv5 ancillary fields
    as will as other primary sources."""
    if gridid == "psn25":
        old_anc_fn = old_ecdr_anc_fn["psn25"]
        newres_anc_fn = new_ecdr_anc_fn["psn25"]
    elif gridid == "pss25":
        old_anc_fn = old_ecdr_anc_fn["pss25"]
        newres_anc_fn = new_ecdr_anc_fn["pss25"]
    else:
        raise RuntimeError(f"GridID not implemented: {gridid}")

    if os.path.isfile(newres_anc_fn):
        raise RuntimeError(f"Output file exists, aborting\n{newres_anc_fn}")

    ds_old = xr.open_dataset(old_anc_fn)
    ds_new = ds_old.copy()

    """
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
    """

    # Find the land-adjacency matrix which indicates whether grid cells are
    #  1, 2, or 3 grid cells away from land (coast).
    adj123_np = calc_adj123_np(ds_old["surface_type"])
    ds_new["adj123"].data[:] = adj123_np[:]

    # Calculate the land90conc field
    l90c_np = create_land90(adj123=ds_new["adj123"].data)
    ds_new["l90c"].data[:] = l90c_np[:]

    """
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
                        ),
                    )
                ),
            ),
        )
    """

    # Write out ancillary file
    ds_new.to_netcdf(newres_anc_fn)
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

    fix_ecdr_anc_file(gridid)
