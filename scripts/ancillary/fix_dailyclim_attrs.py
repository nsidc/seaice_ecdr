"""
./fix_dailyclim_attrs.py

Add global attributes to daily climatology files


"""

import os

from netCDF4 import Dataset

source_dir = "/share/apps/G02202_V5/v05_ancillary/"

base_filename = {
    "psn25": "G02202-ancillary-psn25-daily-invalid-ice-v05r00.nc",
    "pss25": "G02202-ancillary-pss25-daily-invalid-ice-v05r00.nc",
}

new_global_attrs_for_gridid = {
    "psn25": {
        "title": "Day of year climatology invalid ice mask used in CDRv5",
        "date_created": "2024-11-06",
        "date_modified": "2024-11-06",
        "date_metadata_modified": "2024-11-06",
        "keywords": "EARTH SCIENCE SERVICES > DATA ANALYSIS AND VISUALIZATION > GEOGRAPHIC INFORMATION SYSTEMS",
        "Conventions": "CF-1.10, ACDD-1.3",
        "cdm_data_type": "grid",
        "geospatial_bounds": "POLYGON ((-3850000 5850000, 3750000 5850000, 3750000 -5350000, -3850000 -5350000, -3850000 5850000))",
        "geospatial_bounds_crs": "EPSG:3411",
        "geospatial_x_units": "meters",
        "geospatial_y_units": "meters",
        "geospatial_x_resolution": "25000 meters",
        "geospatial_y_resolution": "25000 meters",
        "geospatial_lat_min": 30.980564,
        "geospatial_lat_max": 90.0,
        "geospatial_lon_min": -180.0,
        "geospatial_lon_max": 180.0,
        "geospatial_lat_units": "degrees_north",
        "geospatial_lon_units": "degrees_east",
    },
    "pss25": {
        "title": "Day of year climatology invalid ice mask used in CDRv5",
        "date_created": "2024-11-06",
        "date_modified": "2024-11-06",
        "date_metadata_modified": "2024-11-06",
        "keywords": "EARTH SCIENCE SERVICES > DATA ANALYSIS AND VISUALIZATION > GEOGRAPHIC INFORMATION SYSTEMS",
        "Conventions": "CF-1.10, ACDD-1.3",
        "cdm_data_type": "grid",
        "geospatial_bounds": "POLYGON ((-3950000 4350000, 3950000 4350000, 3950000 -3950000, -3950000 -3950000, -3950000 4350000))",
        "geospatial_bounds_crs": "EPSG:3412",
        "geospatial_x_units": "meters",
        "geospatial_y_units": "meters",
        "geospatial_x_resolution": "25000 meters",
        "geospatial_y_resolution": "25000 meters",
        "geospatial_lat_min": -90.0,
        "geospatial_lat_max": -39.23089,
        "geospatial_lon_min": -180.0,
        "geospatial_lon_max": 180.0,
        "geospatial_lat_units": "degrees_north",
        "geospatial_lon_units": "degrees_east",
    },
}


if __name__ == "__main__":
    import sys

    all_gridids = ("psn25", "pss25")
    try:
        gridid_list = (sys.argv[1],)
        assert gridid_list[0] in all_gridids
    except IndexError:
        gridid_list = all_gridids
        print("No gridid given, using: {gridid_list}", flush=True)
    except AssertionError:
        err_message = f"""
        Invalid gridid
        {sys.argv[1]} not in {all_gridids}
        """
        raise ValueError(err_message)

    for gridid in gridid_list:
        ifn = source_dir + base_filename[gridid]
        assert os.path.isfile(ifn)

        ofn = "./" + base_filename[gridid]

        print(f" input file: {ifn}")
        print(f"output file: {ofn}")
        assert ifn != ofn

        os.system(f"cp -v {ifn} {ofn}")

        assert gridid in new_global_attrs_for_gridid.keys()

        with Dataset(ofn, "r+") as ds:

            new_attrs_dict = new_global_attrs_for_gridid[gridid]
            ds.setncatts(new_attrs_dict)

        print(f"Wrote: {ofn}")
