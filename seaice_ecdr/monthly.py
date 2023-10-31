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
"""
