"""Code for producing the monthly ECDR.

Follows the same procedure as CDR v4.

Variables:

* `nsidc_nt_seaice_conc_monthly`: Average the daily NASA Team sea ice concentration
  values over each month of data
* `nsidc_bt_seaice_conc_monthly: Average the daily Bootstrap sea ice concentration
  values over each month of data
* `cdr_seaice_conc_monthly`: Create combined monthly sea ice concentration
* `stdev_of_cdr_seaice_conc_monthly`: Calculate standard deviation of sea ice
  concentration
* `qa_of_cdr_seaice_conc_monthly`: QA Fields (Weather filters, land
  spillover, valid ice mask, spatial and temporal interpolation, melt onset)
* `melt_onset_day_cdr_seaice_conc_monthly`: Melt onset day (Value from the last
  day of the month)

Notes about CDR v4:

* Requires a minimum of 20 days for monthly calculations, unless the
  platform is 'n07', in which case we use a minimum of 10 days.
* Skips 1987-12 and 1988-01.
* Only determines monthly melt onset day for NH.
* CDR is not re-calculated from the monthly nt and bt fields. Just the average
  of the CDR conc fields.
"""

import xarray as xr

from seaice_ecdr.constants import TEMPORAL_INTERP_DAILY_OUTPUT_DIR


if __name__ == "__main__":
    year = 2022
    month = 3

    data_list = list(TEMPORAL_INTERP_DAILY_OUTPUT_DIR.glob(f"*{year}{month:02}*.nc"))
    breakpoint()

    # *** xarray.core.merge.MergeError: conflicting values for variable
    # *** 'h18_day_si' on objects to be combined. You can skip this check by
    # *** specifying compat='override'.
    conflicting = (
        "/share/apps/amsr2-cdr/ecdr_v5_outputs/standard/"
        "temporal_interp_daily/tiecdr_NH_20220315_ausi_12km.nc"
    )

    other = (
        "/share/apps/amsr2-cdr/ecdr_v5_outputs/standard/"
        "temporal_interp_daily/tiecdr_NH_20220308_ausi_12km.nc"
    )

    conflicting_ds = xr.open_dataset(conflicting)
    other_ds = xr.open_dataset(other)
    breakpoint()

    ds = xr.open_mfdataset([conflicting, other])
