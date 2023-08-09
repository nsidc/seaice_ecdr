"""
gridid_to_xr_dataset

Return an xarray dataset with appropriate geolocation variables for
a given NSIDC gridid
"""


import datetime as dt

import numpy as np
import xarray as xr
from loguru import logger


def xwm(m="exiting in xwm()"):
    # Temporary debugging utility
    raise SystemExit(m)


def get_gridid_hemisphere(gridid):
    # Return the hemisphere of the gridid
    if "psn" in gridid:
        return "north"
    elif "e2n" in gridid:
        return "north"
    elif "pss" in gridid:
        return "south"
    elif "e2s" in gridid:
        return "south"
    else:
        raise ValueError(f"Could not find hemisphere for gridid: {gridid}")


def get_gridid_resolution(gridid):
    # Return the hemisphere of the gridid
    if "3.125" in gridid:
        return "3.125"
    elif "6.25" in gridid:
        return "6.25"
    elif "12.5" in gridid:
        return "12.5"
    elif "25" in gridid:
        return "25"
    else:
        raise ValueError(f"Could not find resolution for gridid: {gridid}")


def get_dataset_for_gridid(gridid, grid_date, return_dataset=True):
    """
    Return xarray dataset with appropriate geolocation dataarrays:
        x
        y
        time
        crs

    Because none of these need compression, no "encoding" dictionary is
    returned

    valid gridids are:
        psn3.125 psn6.25 psn12.5 psn25
        pss3.125 pss6.25 pss12.5 pss25
        e2n3.125 e2n6.25 e2n12.5 e2n25
        e2s3.125 e2s6.25 e2s12.5 e2s25

        More can be added easily
    """
    if return_dataset:
        logger.info(
            f"Creating georeferenced dataset on {gridid} grid for {grid_date}"
        )  # noqa
    crs_attrs = {}

    # CRS for polar stereo grids
    if gridid[:2] == "ps":
        crs_attrs["grid_mapping_name"] = "polar_stereographic"
        crs_attrs["false_easting"] = 0.0
        crs_attrs["false_northing"] = 0.0
        crs_attrs["semi_major_axis"] = 6378273.0
        crs_attrs["inverse_flattening"] = 298.279411123064
        if "psn" in gridid:
            # Note: two attrs will need resolution-based updates:
            #  long_name
            #  GeoTransform
            xleft = -3850000.0
            xright = 3750000.0
            ytop = 5850000.0
            ybottom = -5350000.0
            crs_attrs["long_name"] = "NSIDC_NH_PolarStereo_{res_km}km"
            crs_attrs["straight_vertical_longitude_from_pol"] = -45.0
            crs_attrs["latitude_of_projection_origin"] = 90.0
            crs_attrs["standard_parallel"] = 70.0
            crs_attrs[
                "proj4text"
            ] = "+proj=stere +lat_0=90 +lat_ts=70 +lon_0=-45 +k=1 +x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs"
            crs_attrs["srid"] = "urn:ogc:def:crs:EPSG::3411"
            crs_attrs[
                "crs_wkt"
            ] = 'PROJCS["NSIDC Sea Ice Polar Stereographic North",GEOGCS["Unspecified datum based upon the Hughes 1980 ellipsoid",DATUM["Not_specified_based_on_Hughes_1980_ellipsoid"’,SPHEROID["Hughes 1980",6378273,298.279411123061,AUTHORITY["EPSG","7058"]],AUTHORITY["EPSG","6054"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4054"]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",70],PARAMETER["central_meridian",-45],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["X",EAST],AXIS["Y",NORTH],AUTHORITY["EPSG","3411"]]'
            crs_attrs["GeoTransform"] = "{xleft} {res_m} 0 {ytop} 0 -{res_m}"
        if "pss" in gridid:
            # Note: two attrs will need resolution-based updates:
            #  long_name
            #  GeoTransform
            xleft = -3950000.0
            xright = 3950000.0
            ytop = 4350000.0
            ybottom = -3950000.0
            crs_attrs["long_name"] = "NSIDC_SH_PolarStereo_{res_km}km"
            crs_attrs["straight_vertical_longitude_from_pol"] = 0.0
            crs_attrs["latitude_of_projection_origin"] = -90.0
            crs_attrs["standard_parallel"] = -70.0
            crs_attrs[
                "proj4text"
            ] = "+proj=stere +lat_0=-90 +lat_ts=-70 +lon_0=0 +k=1 +x_0=0 +y_0=0 +a=6378273 +b=6356889.449 +units=m +no_defs"
            crs_attrs["srid"] = "urn:ogc:def:crs:EPSG::3412"
            crs_attrs[
                "crs_wkt"
            ] = 'PROJCS["NSIDC Sea Ice Polar Stereographic South",GEOGCS["Unspecified datum based upon the Hughes 1980 ellipsoid",DATUM["Not_specified_based_on_Hughes_1980_ellipsoid",SPHEROID["Hughes 1980",6378273,298.279411123061,AUTHORITY["EPSG","7058"]],AUTHORITY["EPSG","6054"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4054"]],PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-70],PARAMETER["central_meridian",0],PARAMETER["scale_factor",1],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["metre",1,AUTHORITY["EPSG","9001"]],AXIS["X",EAST],AXIS["Y",NORTH],AUTHORITY["EPSG","3412"]]'
            crs_attrs["GeoTransform"] = "{xleft} {res_m} 0 {ytop} 0 -{res_m}"

    elif gridid[:3] == "e2t":
        xleft = -17367530.44
        xright = 17367530.44
        ytop = 6756820.2
        ybottom = -6756820.2
        crs_attrs["long_name"] = "NSIDC_EASE2_T{res_km}km"
        crs_attrs["grid_mapping_name"] = "lambert_cylindrical_equal_area"
        crs_attrs["longitude_of_central_meridian"] = 0.0
        crs_attrs["standard_parallel"] = 30.0
        crs_attrs["false_easting"] = 0.0
        crs_attrs["false_northing"] = 0.0
        crs_attrs["semi_major_axis"] = 6378137.0
        crs_attrs["inverse_flattening"] = 298.257223563
        crs_attrs[
            "proj4text"
        ] = "+proj=cea +lon_0=0 +lat_ts=30 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
        crs_attrs["srid"] = "urn:ogc:def:crs:EPSG::6933"
        crs_attrs[
            "crs_wkt"
        ] = 'PROJCS["unnamed",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9108"]],AUTHORITY["EPSG","4326"]],PROJECTION["Cylindrical_Equal_Area"],PARAMETER["standard_parallel_1",30],PARAMETER["central_meridian",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["Meter",1],AUTHORITY["epsg","6933"]]'
        crs_attrs["GeoTransform"] = "-17367530.44 {res_tkm} 0 6756820.2 0 -{res_tkm}"

    elif gridid[:2] == "e2":
        xleft = -9000000.0
        xright = 9000000.0
        ytop = 9000000.0
        ybottom = -9000000.0
        crs_attrs["grid_mapping_name"] = "lambert_azimuthal_equal_area"
        crs_attrs["longitude_of_projection"] = 0.0
        crs_attrs["false_easting"] = 0.0
        crs_attrs["false_northing"] = 0.0
        crs_attrs["semi_major_axis"] = 6378137.0
        crs_attrs["inverse_flattening"] = 298.257223563
        crs_attrs["GeoTransform"] = "-9000000 {res_m} 0 9000000 0 -{res_m}"

        if gridid[:3] == "e2n":
            crs_attrs["long_name"] = "NSIDC_EASE2_N{res_km}km"
            crs_attrs["latitude_of_projection_origin"] = 90.0
            crs_attrs[
                "proj4text"
            ] = "+proj=laea +lat_0=90 +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
            crs_attrs["srid"] = "urn:ogc:def:crs:EPSG::6931"
            crs_attrs[
                "crs_wkt"
            ] = 'PROJCS["unnamed",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9108"]],AUTHORITY["EPSG","4326"]],PROJECTION["Lambert_Azimuthal_Equal_Area"],PARAMETER["latitude_of_center",90],PARAMETER["longitude_of_center",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["Meter",1],AUTHORITY["epsg","6931"]]atitude_of_center",90],PARAMETER["longitude_of_center",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["Meter",1],AUTHORITY["epsg","6931"]]'
        elif gridid[:3] == "e2s":
            crs_attrs["long_name"] = "NSIDC_EASE2_S{res_km}km"
            crs_attrs["latitude_of_projection_origin"] = -90.0
            crs_attrs[
                "proj4text"
            ] = "+proj=laea +lat_0=-90 +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
            crs_attrs["srid"] = "urn:ogc:def:crs:EPSG::6932"
            crs_attrs[
                "crs_wkt"
            ] = 'PROJCS["unnamed",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],TOWGS84[0,0,0,0,0,0,0],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9108"]],AUTHORITY["EPSG","4326"]],PROJECTION["Lambert_Azimuthal_Equal_Area"],PARAMETER["latitude_of_center",-90],PARAMETER["longitude_of_center",0],PARAMETER["false_easting",0],PARAMETER["false_northing",0],UNIT["Meter",1],AUTHORITY["epsg","6932"]]'
        else:
            raise ValueError(f"Could not determine EASE2 hemisphere: {gridid}")

    res_km = gridid[3:]
    res_m = int(float(res_km) * 1000)
    crs_attrs["long_name"] = crs_attrs["long_name"].format(res_km=res_km)
    if gridid[:3] == "e2t":
        res_m = 1.0010104 * res_m
        res_tkm = 1.0010104 * float(res_km)
        crs_attrs["GeoTransform"] = crs_attrs["GeoTransform"].format(
            res_km=res_km,
            res_tkm=res_tkm,
            xleft=xleft,
            ytop=ytop,
        )
    else:
        crs_attrs["GeoTransform"] = crs_attrs["GeoTransform"].format(
            res_km=res_km,
            res_m=res_m,
            xleft=xleft,
            ytop=ytop,
        )

    xdim = int(np.round((xright - xleft) / res_m))
    ydim = int(np.round((ytop - ybottom) / res_m))

    xleft_center = xleft + res_m / 2
    xright_center = xright - res_m / 2
    ytop_center = ytop - res_m / 2
    ybottom_center = ybottom + res_m / 2

    xs = np.linspace(xleft_center, xright_center, num=xdim)
    ys = np.linspace(ytop_center, ybottom_center, num=ydim)

    x_da = xr.DataArray(
        name="x",
        data=xs,
        dims=["x"],
        attrs={
            "standard_name": "projection_x_coordinate",
            "long_name": "x",
            "axis": "X",
            "units": "meters",
            "coverage_content_type": "coordinate",
            "valid_range": [xleft, xright],
            "actual_range": [xleft_center, xright_center],
        },
    )

    y_da = xr.DataArray(
        name="y",
        data=ys,
        dims=["y"],
        attrs={
            "standard_name": "projection_y_coordinate",
            "long_name": "y",
            "axis": "Y",
            "units": "meters",
            "coverage_content_type": "coordinate",
            "valid_range": [ybottom, ytop],
            "actual_range": [ybottom_center, ytop_center],
        },
    )

    time_as_int = (grid_date - dt.date(1970, 1, 1)).days
    time_da = xr.DataArray(
        name="time",
        data=[int(time_as_int)],
        dims=["time"],
        attrs={
            "standard_name": "time",
            "long_name": "ANSI date",
            "calendar": "standard",
            "axis": "T",
            "units": "days since 1970-01-01",
            "coverage_content_type": "coordinate",
            "valid_range": [int(0), int(30000)],
        },
    )

    crs_da = xr.DataArray(
        name="crs",
        data="a",
        attrs=crs_attrs,
    )

    if return_dataset:
        ds = xr.Dataset(
            data_vars=dict(
                crs=crs_da,
                time=time_da,
                y=y_da,
                x=x_da,
            ),
            attrs={
                "description": f"Geolocation for gridid {gridid}",
            },
        )

        return ds
    else:
        return {
            "crs": crs_da,
            "time": time_da,
            "y": y_da,
            "x": x_da,
        }

    return None


if __name__ == "__main__":
    import os
    import sys

    try:
        gridid = sys.argv[1]
    except IndexError:
        gridid = "psn25"
        print(f"Using default gridid: {gridid}")

    try:
        grid_date = dt.datetime.strptime(sys.argv[2], "%Y%m%d").date()
    except IndexError:
        grid_date = dt.date(1970, 1, 1)
        print(f"Using default grid_date: {grid_date}")

    if len(sys.argv) > 2:
        add_random_field = True
    else:
        add_random_field = False

    ds = get_dataset_for_gridid(gridid, grid_date)

    if add_random_field:
        # Generate an appropriate time value
        sample_date = dt.date(2022, 12, 25)
        time_units = ds.variables["time"].attrs["units"]
        assert "days since" in time_units
        ref_date_str = time_units.split()[2]
        ref_date = dt.datetime.strptime(ref_date_str, "%Y-%m-%d").date()
        sample_date_value = (sample_date - ref_date).days

        old_time = ds.coords["time"]
        orig_x = ds.coords["x"]
        orig_y = ds.coords["y"]
        new_time = xr.DataArray(
            name=old_time.name,
            data=[int(sample_date_value)],
            dims=old_time.dims,
            attrs=old_time.attrs,
        )

        ds = ds.assign_coords(
            {
                "time": new_time,
                "y": orig_y,
                "x": orig_x,
            }
        )

        xdim = np.size(ds.variables["x"].data)
        ydim = np.size(ds.variables["y"].data)

        # Generate random data
        # field = np.random.randn(1, ydim, xdim)
        field = np.zeros((1, ydim, xdim))
        yvals, xvals = np.meshgrid(ds.variables["x"].data, ds.variables["y"].data)
        field[0, :, :] = xvals[:, :] + yvals[:, :]

        ds["sample_field"] = (
            ("time", "y", "x"),
            field,
            {
                "_FillValue": 0.0,
                "grid_mapping": "crs",
                "sample_field_attr_1": "attribute 1",
                "sample_field_attr_2": 2.0,
                "sample_field_attr_3": [int(3), int(3333)],
                "sample_field_attr_4": [np.int16(4), np.int16(44)],
                "sample_field_attr_5": [np.int16(5), np.float32(55)],
                "sample_field_attr_6": [np.float64(6), np.float64(66)],
            },
            {
                "zlib": True,
            },
        )
        ds.attrs["comment"] = "This version has a sample data field"

    try:
        ofn = f"geolocation_{gridid}.nc"
        ds.to_netcdf(ofn, unlimited_dims=["time"])
        print(f"Wrote: {ofn}")
    except AttributeError:
        print("No dataset returned")

    print("Finished")
