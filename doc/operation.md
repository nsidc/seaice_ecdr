# Sea Ice E-CDR Operations

This document outlines how the `seaice_ecdr` code is leveraged in operations.


## Command-line Interface (CLI)

The `./scripts/cli.sh` script is the primary entrypoint for interacting with the
CLI. Not all CLI subcommands are used in operations (e.g., they are used
exclusively in dev). The CLI subcommands outlined below should be used in
production.

**NOTE**: on NSIDC production VMs, the CLI is setup to be available on the
system PATH as `ecdr`. E.g.,:

```
$ ecdr --help
```

## G10016 NRT Processing

###  Daily processing

There are two daily NRT processing streams: one for F17 data and the other for
AMSR2.

Note that the data produced for these processing steams will be output to
different locations and should be rsynced to separate locations on
`/disks/sidads_ftp/pub/DATASETS/NOAA/G10016_V3/`, which are noted below.

#### F17 processing

Daily NRT processing for F17 should occur by running this command:

```
daily-nrt --last-n-days 5 --hemisphere both --nrt-platform-id F17
```

Note that the `--overwrite` flag can be used to re-create NRT data if e.g., a
data gap is filled a few days late.

Once processing is complete, rsync the contents of
`/share/apps/G10016_V3/v03r00/production/CDR/complete/` to
`/disks/sidads_ftp/pub/DATASETS/NOAA/G10016_V3/CDR/`


#### am2 processing

Daily NRT processing for AMSR2 should be run at a 1-week lag. E.g, if today is
Nov. 5, 2024, we should create a data file for one week ago: Oct. 30th, 2024:


```
daily-nrt --date 2024-10-30 --hemisphere both --nrt-platform-id am2
```

Once processing is complete, rsync the contents of
`/share/apps/G10016_V3/v03r00/production/prototype_amsr2/complete/` to
`/disks/sidads_ftp/pub/DATASETS/NOAA/G10016_V3/prototype_amsr2/`

### Monthly processing

We produce NRT monthly files for F17 (not AM2).

To run for a single month:
```
monthly-nrt --year YYYY --month MM --hemisphere both
```

To process multiple years/months:

```
monthly-nrt --year YYYY --month MM --end-year YYYY --end-month MM --hemisphere both
```


## G02202 "final" Processing

Final data will be written to
`/share/apps/G02202_V5/v05r00/production/complete/`. The contents of this
directory should be rsync-ed to `/disks/sidads_ftp/pub/DATASETS/NOAA/G02202_V5`
after successful completion of each G02202 procesing job.

Typically, "final" procesing occurs all at once, as data becomes
finalized/available for NSIDC-0001. In other words, the following do not need to
be run on a daily/monthly basis, but instead can be bundled into one job. See
[the ops job for
v4](https://ci.jenkins-ops-2022.apps.int.nsidc.org/job/G02202_Generate_Dataset_Production)
as an example.

### Daily processing

To create daily data:

```
daily --start-date YYYY-MM-DD --end-date YYYY-MM-DD --hemisphere both
```

Once daily data for a year is available, this data should be aggregated with the
`daily-aggregate` command:

```
daily-aggregate --year YYYY --hemisphere {north|south}
```

There will be one daily aggregate file per year per hemisphere.

### Monthly processing

When a month's worth of daily data is available, monthly data files can be produced:

```
monthly --year YYYY --month mm --hemisphere both
```

A range of years/months can also be specified:


```
monthly --year YYYY --month mm --end-year YYYY --end-month MM --hemisphere {north|south}
```

Each time a new monthly file is produced, the monthly aggregate file should be
updated. There will always only be one monthly aggregate file per hemisphere:

```
monthly-aggregate --hemisphere {north | south}
```

### Validation

Each time finalized data is produced, the validation CLI should be run:


```
validate-outputs --hemisphere {north|south} --start-date YYYY-MM-DD --end-date YYYY-MM-DD
```

This produces log files in
`/share/apps/G02202_V5/v05r00_outputs/production/validation/` that should be
published to the production location. TODO: confirm this is accurate. Does not
look like v4 does this.
