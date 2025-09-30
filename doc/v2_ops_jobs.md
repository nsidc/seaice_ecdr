# Ops v2 jenkins jobs

This document outlines what daily/monthly/yearly ops jenkins processing needs to
be setup for the Sea Ice ECDR v2 (G10016 v4 and G02202 v6).

## G10016 NRT Processing

### Daily 

Daily NRT processing for AMSR2 should occur by running this command: 

```
daily-nrt --last-n-days 5  --hemisphere both --overwrite
```

Running the last 5 days ensures that data get produced when data get delayed by
a few days.

Once processing is complete, rsync the contents of
`/share/apps/G10016_V4/v04r00/production/CDR/complete/` to
`/disks/sidads_ftp/pub/DATASETS/NOAA/G10016_V4/CDR/`

This should roughly replicate the existing ops jobs for the ECDR v1 / G10016 v3:
[G10016v3_CDR_NRT_F17_Daily_Production](https://ci.jenkins-ops-2022.apps.int.nsidc.org/job/G10016v3_CDR_NRT_F17_Daily_Production). With
the following differences:

* instead of targeting the `seaice_ecdr` VM, `seaice_ecdr_v2` should be targeted.
* The `--nrt-platform-id` option should be removed 
* Paths should be updated to reflect G10016 v4.

### Monthly processing


Monthly processing should roughly replicate the existing ops job for ECDR
v1/G10016 v3:
[G10016v3_CDR_NRT_Monthly_Production](https://ci.jenkins-ops-2022.apps.int.nsidc.org/job/G10016v3_CDR_NRT_Monthly_Production/). With the following differences:

* instead of targeting the `seaice_ecdr` VM, `seaice_ecdr_v2` should be targeted.
* Paths should be updated to reflect G10016 v4.
* As a follow-on (this could be another job), all G10016 NRT data files that are
  older than 1 month should be cleaned up from the public archive. This is
  because the fully temporally-interpolated G02202 files should replace the NRT
  ones over time, and become the permanent CDR record.

## G02202 "final" Processing

Final data will be written to
`/share/apps/G02202_V6/v06r00/production/complete/`. The contents of this
directory should be rsync-ed to `/disks/sidads_ftp/pub/DATASETS/NOAA/G02202_V6`
after successful completion of each G02202 procesing job.

"Final" processing of AMSR2 data for G02202 occurs at a lag of 5 days to allow
for complete, two-sided temporal interpolation.

Unlike ECDR v1/G02202 V5, which did G02202 processing in 1-2 batches throughout
the year, ECDR v2/G02202 v6 will be produced daily/monthly. The jobs that will
be created for this processing can take inspiration from
[G02202v5_Generate_Dataset_Production](https://ci.jenkins-ops-2022.apps.int.nsidc.org/job/G02202v5_Generate_Dataset_Production/configure).

### Daily

Daily processing should be run at a 6 day lag to allow complete, two-sided
interpolation. 

This means that if we are processing data on Sept. 29, 2025, we
should target daily processing for Sept. 23, 2025.


```
daily --date 2025-09-23 --hemisphere both
```

Alternatively, to compensate for possible data delays, a week of data could be
processed with with the `--overwrite` flag:


```
daily --start-date 2025-09-17 --end-date 2025-09-23 --hemisphere both --overwrite
```


### Monthly 

At the begninning of each month (on the 6th or after), when a full month's worth
of daily data is ready from the previous month, monthly data files should be
created:

```
monthly --year YYYY --month mm --hemisphere both
```

The monthly aggregate file should also be updated (only one file per
hemisphere):

```
monthly-aggregate --hemisphere north
monthly-aggregate --hemisphere south
```

After monthly data products have been produced, the validation CLI for that
month should be run:

```
validate-outputs --hemisphere both --start-date YYYY-MM-DD --end-date YYYY-MM-DD
```

This produces log files in
`/share/apps/G02202_V6/v06r00_outputs/production/validation/` that should be
reviewed by the NOAA@NSIDC project manager responsible for the sea ice CDR.


### Yearly processing

At the beginning of each year (on the 6th of Jan. or later), daily data from the
previous should be aggregated into yearly aggregate files:

```
daily-aggregate --year YYYY --hemisphere north
daily-aggregate --year YYYY --hemisphere south
```
