# Create Surface/Geo mask ancillary files

This directory includes a script for creating the following ancillary files
based on representative inputs from nsidc0771 and nsidc0051:

* `/share/apps/G02202_V5/cdrv5_ancillary/cdrv5_surfgeo_psn12.5.nc`
* `/share/apps/G02202_V5/cdrv5_ancillary/cdrv5_surfgeo_pss12.5.nc`

These surface/geo files include the following variables:

* `crs`
* `latitude`
* `longitude`
* `surface_type`: mask field with values for ocean, lake, coast, and land.
* `polehole_bitmask` (Northern hemisphere only): bitmask encoding representative pole hole for each satellite utilized by the ECDR.
* `x`: projected x coord
* `y`: projected y coord


## Running the script

Before running the script, it is reccomended that the tests be run first:

```
$ pytest scripts/surface_geo_masks/test_create_surface_and_geo_masks.py
```

Then, run the script:

```
$ python scripts/surface_geo_masks/create_surface_geo_mask.py
Creating surfgeo ancillary field for gridid: psn12.5
  /share/apps/G02202_V5/cdrv5_ancillary/cdrv5_surfgeo_psn12.5.nc
Creating surfgeo ancillary field for gridid: pss12.5
  /share/apps/G02202_V5/cdrv5_ancillary/cdrv5_surfgeo_pss12.5.nc
```
