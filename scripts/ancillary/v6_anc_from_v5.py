"""Convert v05r00 ancillary files into v06r00 ancillary files.

Changes:

* Filenames are updated to replace "v05r00" to "v06r00"
* "title" global attr is updated to replace "CDRv5" with "CDRv6"
* Update `G02202-ancillary-psn25-v06r00.nc`'s `polehole_bitmask` for 0802 AMSR2
  pole hole, which is bigger than the AU_SI25 pole hole used in v5.

The `polehole_bitmask_v6.dat` file was created by Scott Stweart. He provided
this documentation:

> So, we can extract the polehole bitmask field with:
>
> `ncks -C -O -v polehole_bitmask -b G02202-ancillary-psn25-v05r00_polehole_bitmask.dat G02202-ancillary-psn25-v05r00.nc dummy.nc`
>
> This yields the file: G02202-ancillary-psn25-v05r00_polehole_bitmask.dat (the
> arg of the -b option in the ncks command per ncdump -h of the .nc file, the
> flags are:
>
> ```
> polehole_bitmask:flag_masks = 1UB, 2UB, 4UB, 8UB, 16UB, 32UB, 64UB ;
> polehole_bitmask:flag_meanings = "n07_polemask F08_polemask F11_polemask F13_polemask F17_polemask ame_polemask am2_polemask"  ;
> ```
>
> So, we want to mask the 7th bit ("64")
>
> I grabbed a day of 0803 and made a png:  NSIDC-0803_SEAICE_AMSR2_N_20240101_V1.0.png
>
> So, the current AMSR2 polehole mask is: data[230:238, 150:158] with three grid
> cells missing from each corner.  The new mask should be data[149:159, 229:239]
> with only the single cornermost grid removed.  So,
>
> ```
> >>> orig = np.fromfile('G02202-ancillary-psn25-v05r00_polehole_bitmask.dat', dtype=np.uint8).reshape(448, 304)
>
> >>> newmask = np.zeros((448, 304), dtype=np.uint8)
> >>> newmask[229:239, 149:159] = 64
> >>> newmask[229, 149] = 0
> >>> newmask[238, 149] = 0
> >>> newmask[238, 158] = 0
> >>> newmask[229, 158] = 0
> >>> newmask.tofile('newmask.dat')
>
> >>> updated = np.bitwise_or(orig, newmask)
> >>> updated.tofile('polemask_updated.dat')
> ```

"""

from pathlib import Path

import numpy as np
import xarray as xr

V6_POLEHOLE_BITMASK_FP = Path(__file__).parent / "polehole_bitmask_v6.dat"

V5_ANC_DIR = Path("/share/apps/G02202_V5/v05_ancillary/")
# TODO: this is temporary, we plan on a new share for v6.
V6_ANC_DIR = Path("/share/apps/G02202_V5/v06r00_ancillary/")


if __name__ == "__main__":

    for anc_file in V5_ANC_DIR.glob("*.nc"):
        if anc_file.is_file():
            ds = xr.open_dataset(anc_file, mask_and_scale=False, engine="netcdf4")
            # Update the title
            ds.attrs["title"] = ds.title.replace("CDRv5", "CDRv6")
            # Update the polehole bitmask
            if anc_file.name == "G02202-ancillary-psn25-v05r00.nc":
                v6_polehole_bitmask = np.fromfile(
                    V6_POLEHOLE_BITMASK_FP, dtype=np.uint8
                ).reshape(448, 304)
                ds.polehole_bitmask.data[:] = v6_polehole_bitmask

            ds.to_netcdf(V6_ANC_DIR / anc_file.name.replace("v05r00", "v06r00"))

    # Update varnames
