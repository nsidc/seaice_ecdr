"""
./gen_dailyclim_from0079.py

Generate a daily climatology from Bootstrap (NSIDC-0079) data
  - Note: this is expected to be valuable because 0079 is manually QC'd

Usage:
    python gen_dailyclim_from0079.py [hem]
  where hem is nh or sh
  and if hem is omitted, both will be generated

Assumes daily 0079 data is located in "standard" ecs directories:
    /ecs/DP1/PM/NSIDC-0079.004/<yyyy.mm.dd>/NSIDC0079_SEAICE_PS_{HEM}25km_{YYYYMMDD}_v4.0.nc
    where YYYYMMDD is 8-digit date string
               HEM is capital N or S

Overall approach:
    - Initialize uint8 daily fields for day-of-year 1-366 to 255
       - ignore index zero
       - treat last day of year as doy 365
         - by accumulating values in both index 365 and 366, and ORing the
           final fields
       - use separate fields for sea ice extent found (siext) and siext_notfound
         - use 10% min for siext threshold
    - Loop through all days from 11/01/1978 (start of 0079)
      through end of 2023 (current latest-date
      - apply any siext to all days +/- 5 day-of-years from current day

"""

import datetime as dt
import os

import numpy as np
import xarray as xr
from netCDF4 import Dataset
from scipy.signal import convolve2d

fn_0079_ = (
    "/ecs/DP1/PM/NSIDC-0079.004/{dotymd}/NSIDC0079_SEAICE_PS_{hem}25km_{strymd}_v4.0.nc"
)
fn_dayclim_ = "daily_siext_from0079_{d0}-{d1}_{gridid}.nc"

reference_gridid_files = {
    "psn25": "/share/apps/G02202_V5/v05_ancillary/ecdr-ancillary-psn25.nc",
    "pss25": "/share/apps/G02202_V5/v05_ancillary/ecdr-ancillary-pss25.nc",
}

# first_date = dt.date(1986, 1, 1)   # testcase 1986-1988
# last_date = dt.date(1988, 12, 31)  # testcase 1986-1988

first_date = dt.date(1978, 11, 1)  # 0079 starts Nov 1, 1978
last_date = dt.date(2023, 12, 31)  # as of Sept 2024, last full year is 2023

# Will include a doy's observation in mask for doys +/- doy_offset days of
#   doy with observations
doy_offset = 4


def iter_adj_doys(date, offset):
    """Return iterator of days-of-year surrounding date"""
    d = date - dt.timedelta(days=offset)
    while d <= date + dt.timedelta(days=offset):
        doy = int(d.strftime("%j").lstrip("0"))
        yield doy

        d += dt.timedelta(days=1)


def dilate_siext(sie, nosie):
    """Return an array where siext is dilated into land"""
    kernel = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    # max_iterations = 2
    # max_iterations = 5
    max_iterations = 10

    ndays, ydim, xdim = sie.shape
    mask = np.zeros((ndays, ydim, xdim), dtype=np.uint8)
    mask[:] = 255

    for doy in range(1, 366 + 1):
        if doy % 50 == 0:
            print(f"Working on doy: {doy} of 366...")

        yea = sie[doy, :, :]
        nay = nosie[doy, :, :]

        for _ in range(max_iterations):
            # Dilate yea
            is_yea = yea == 100
            is_nay = nay == 100
            yea_conv = convolve2d(
                is_yea, kernel, mode="same", boundary="fill", fillvalue=0
            )
            fill_with_yea = (yea_conv > 0) & ~is_nay
            yea[fill_with_yea] = 100

            # Dilate nay
            is_nay = nay == 100
            is_yea = yea == 100
            nay_conv = convolve2d(
                is_nay, kernel, mode="same", boundary="fill", fillvalue=0
            )
            fill_with_nay = (nay_conv > 0) & ~is_yea
            nay[fill_with_nay] = 100

        # Use those values to fill mask
        is_nay = nay == 100
        is_yea = yea == 100

        msk = mask[doy, :, :]
        msk[is_nay] = 0
        msk[is_yea] = 100
        # breakpoint()

    return mask


def find_sie_0079(gridid, d0, d1):
    if gridid == "psn25":
        xdim = 304
        ydim = 448
        hem = "N"
    elif gridid == "pss25":
        xdim = 316
        ydim = 332
        hem = "S"
    else:
        raise ValueError(f"Cannot figure out xdim, ydim for gridid {gridid}")

    sie_exists = np.zeros((367, ydim, xdim), dtype=np.uint8)
    nosie_exists = np.zeros((367, ydim, xdim), dtype=np.uint8)

    date = d0
    while date <= d1:
        # if date.day == 1 and (date.month % 3 == 0):
        if date.month == 1 and date.day == 1:
            print(f"Working on: {date=}", flush=True)

        try:
            fn_format_dict = {
                "dotymd": date.strftime("%Y.%m.%d"),
                "strymd": date.strftime("%Y%m%d"),
                "hem": hem,
            }
            fn0079 = fn_0079_.format(**fn_format_dict)
            ds0079 = Dataset(fn0079)
            ds0079.set_auto_maskandscale(False)
        except FileNotFoundError:
            print(f"FileNotFound: {fn0079}", flush=True)

        icecon_vars = [key for key in ds0079.variables.keys() if "_ICECON" in key]
        if len(icecon_vars) > 1:
            print(f"More than one icecon for {date=}: {icecon_vars}", flush=True)
        elif len(icecon_vars) == 0:
            # Skip this day
            pass
        else:
            # Values of siconc are 0-1000 (0-100%), 1100 missing, 1200 land
            siconc = np.array(ds0079.variables[icecon_vars[0]][0, :, :])

            is_siext = (siconc >= 10) & (siconc <= 1000)
            is_nosiext = siconc < 10

            for doy in iter_adj_doys(date, doy_offset):
                sie_exists[doy, is_siext] = 100
                nosie_exists[doy, is_nosiext] = 100

        date += dt.timedelta(days=1)

    # Finalize by ensuring that doy 365 and 366 are the same
    has_365 = sie_exists[365, :, :] == 100
    hasno_365 = nosie_exists[365, :, :] == 100
    has_366 = sie_exists[366, :, :] == 100
    hasno_366 = nosie_exists[366, :, :] == 100

    sie_exists[366, :, :][has_365] = 100
    sie_exists[365, :, :][has_366] = 100

    nosie_exists[366, :, :][hasno_365] = 100
    nosie_exists[365, :, :][hasno_366] = 100

    assert np.all(sie_exists[365, :, :] == sie_exists[366, :, :])
    assert np.all(nosie_exists[365, :, :] == nosie_exists[366, :, :])

    return sie_exists, nosie_exists


def get_surface_type(gridid, vernum):
    # Return the surface_type field for this hem and version
    if gridid == "psn25":
        hem = "n"
    elif gridid == "pss25":
        hem = "s"

    if vernum == 4:
        verstr = "-v04r00"
    elif vernum == 5:
        verstr = ""

    fn = f"/share/apps/G02202_V5/v05_ancillary/ecdr-ancillary-ps{hem}25{verstr}.nc"
    ds_anc = Dataset(fn)
    ds_anc.set_auto_maskandscale(False)
    surftype = np.array(ds_anc["surface_type"])

    return surftype


def gen_dailyclim_0079(hem, d0, d1):

    print("Will gen daily climatology from 0079 for:")
    print(f"  hem: {hem}")
    print(f"  from: {d0} through {d1}", flush=True)

    if hem == "nh":
        gridid = "psn25"
        xdim = 304
        ydim = 448
    elif hem == "sh":
        gridid = "pss25"
        xdim = 316
        ydim = 332
    else:
        raise ValueError(f"Cannot figure out gridid for hem: {hem}")

    nc_fn = f"ecdr-ancillary-{gridid}-dailyclim.nc"
    if os.path.isfile(nc_fn):
        print(f"Output file exists: {nc_fn}")
        return

    mask_fn = f'dailyclim_{hem}_{d0.strftime("%Y%m%d")}-{d1.strftime("%Y%m%d")}.dat'
    if os.path.isfile(mask_fn):
        # Read in the .dat values instead of calculating them
        mask = np.fromfile(mask_fn, dtype=np.uint8).reshape(367, ydim, xdim)
        print(f"  Read mask values from:  {mask_fn}", flush=True)
    else:
        # Calculate the day-of-year [doy] mask values
        sie_fn = f'sie_exists_{hem}_{d0.strftime("%Y%m%d")}-{d1.strftime("%Y%m%d")}.dat'
        nosie_fn = (
            f'nosie_exists_{hem}_{d0.strftime("%Y%m%d")}-{d1.strftime("%Y%m%d")}.dat'
        )

        if os.path.isfile(sie_fn) and os.path.isfile(nosie_fn):
            print("Will read from:")
            print(f"  yes: {sie_fn}")
            print(f"  not: {nosie_fn}", flush=True)
            sie_exists = np.fromfile(sie_fn, dtype=np.uint8).reshape(367, ydim, xdim)
            nosie_exists = np.fromfile(nosie_fn, dtype=np.uint8).reshape(
                367, ydim, xdim
            )
        else:
            print("Will write to:")
            print(f"  yes: {sie_fn}")
            print(f"  not: {nosie_fn}", flush=True)

            sie_exists, nosie_exists = find_sie_0079(gridid, d0, d1)

            sie_exists.tofile(sie_fn)
            nosie_exists.tofile(nosie_fn)

        # Fill lake values with 255 if they have 0 or 100 in either sie or nosie
        v4_surftype = get_surface_type(gridid, 4)
        v5_surftype = get_surface_type(gridid, 5)

        is_lake_v4 = v4_surftype == 75
        is_lake_v5 = v5_surftype == 75

        for doy in range(1, 366 + 1):
            sie_exists[doy, is_lake_v4] = 255
            sie_exists[doy, is_lake_v5] = 255

            nosie_exists[doy, is_lake_v4] = 255
            nosie_exists[doy, is_lake_v5] = 255

        # Now, process the fields by dilation of yea and nay
        mask = dilate_siext(sie_exists, nosie_exists)
        mask_fn = f'dailyclim_{hem}_{d0.strftime("%Y%m%d")}-{d1.strftime("%Y%m%d")}.dat'
        mask.tofile(mask_fn)
        print(f"  Wrote:  {mask_fn}", flush=True)

    # Create the netCDF file from the mask value
    # netcdf file will get its CRS information from a sample file
    # The field will be 'invalid_ice_mask'
    #   - a value of 1 (or True) indicates where sea ice was not observed
    #     near this day-of-year
    #   - a value of 0 means that sea ice is permitted for this doy
    #     ...if the grid cell is ocean (not land)
    # In the mask field here:
    #   0: invalid ice
    #   100: potentially valid sea ice (though could be land, depending on land/surfacetype mask)
    #   255: sea ice validity/invalidity not determined because far from ocean
    #   Note: lakes were filled here and will never have valid sea ice

    # Load the reference dataset
    reference_ds = xr.open_dataset(reference_gridid_files[gridid])

    iim = mask.copy()
    iim[mask == 0] = 1
    iim[mask != 0] = 0

    invalid_ice_mask_arr = xr.DataArray(
        iim,
        dims=("doy", "y", "x"),
        attrs={
            "short_name": "daily invalid ice mask",
            "long_name": f"{gridid} daily invalid ice mask from NSIDC-0079",
            "grid_mapping": "crs",
            "flag_values": np.array((0, 1), dtype=np.uint8),
            "flag_meanings": "valid_seaice_location invalid_seaice_location",
            "units": 1,
            "comment": "Mask indicating where seaice will not be found on this day based on climatology from NSIDC-0079",
        },
    )
    invalid_ice_mask_arr.encoding["_FillValue"] = None

    iim_ds = xr.Dataset(
        data_vars=dict(
            invalid_ice_mask=invalid_ice_mask_arr[1:, :, :],  # exclude doy 0
            crs=reference_ds.crs,
        ),
        coords=dict(
            doy=np.arange(1, 366 + 1, dtype=np.int16),
            y=reference_ds.y,
            x=reference_ds.x,
        ),
    )

    # TODO: Does this var need other attrs?
    iim_ds.doy.attrs = dict(
        long_name="Day of year",
        comment="366 days are provided to account for leap years.",
    )

    iim_ds.doy.encoding["_FillValue"] = None
    iim_ds.y.encoding["_FillValue"] = None
    iim_ds.x.encoding["_FillValue"] = None

    # TODO: Global attributes will likely be wrong because they are copied!
    iim_ds.to_netcdf(nc_fn)
    print(f"Wrote: {nc_fn}")


if __name__ == "__main__":
    import sys

    all_hems = ("nh", "sh")
    try:
        hem_list = (sys.argv[1],)
        assert hem_list[0] in all_hems
    except IndexError:
        hem_list = all_hems
        print("No hem given, using: {hem_list}", flush=True)
    except AssertionError:
        err_message = f"""
        Invalid hemisphere
        {sys.argv[1]} not in {all_hems}
        """
        raise ValueError(err_message)

    for hem in hem_list:
        gen_dailyclim_0079(hem, first_date, last_date)
