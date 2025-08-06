# v2.0.0

* Replace use of `AU_SI25` for AMSR2 brightness temperatures with
  `NSIDC-0802`. `AU_SI25` is expected to be discontinued in September 2025.
* Remove support for NRT F17 from `NSIDC-0080`, which was expected to be
  discontinued at the end of July 2025 (now Sept. 2026). The primary NRT CDR
  will be produced from `NSIDC-0802`. There will be no prototype CDR for now,
  although we anticipate possibly wanting to designate AMSR3 data as "prototype"
  once it is available.
* Update CDR concentration calculation code to use a variable minimum threshold
  for sea ice concentration based on day of year for data from DMSP
  platforms. This implements the "Seki" method approach to aligning DMSP-derived
  concentrations with AMSR2.


# v1.1.0

* Update NRT daily processing code to fetch NSIDC-0080 (F17) via
  `earthaccess`. This will allow G10016 v3 to continue processing F17 data after
  the NSIDC on-prem ECS system is shut down.


# v1.0.2

* Fix for subprocess calls for `git` operations (e.g., to get the current
  version) with potentially incorrect working dir.
* Remove OBE assertion
* Fix monthly validation code to process all months in date range.
* Update validation code to process data in final, publication-ready state.

# v1.0.1

* Monthly melt onset day uses DOY 244 as the last day in the melt season unless
  it is not present. Then the lastest available date is used.
* Remove unnecessary attributes from netcdf files
* Southern hemisphere melt onset omits the
  `at_least_one_day_during_month_has_melt_detected` bitmask.

# v1.0.0

* First production release of 25km CDR

* Add new `daily` cli that produces daily NC files ready for publication. During
  the AMSR2 period, a `prototype_am2` group is added to the final output files
  with AMSR2-derived fields.

* Support daily and monthly NRT processing. Daily NRT processing supports
  producing AMSR2 and F17 outputs. Monthly processing supports F17 output, but
  not AMSR2.

* Removes much of the code required to match CDR v4.

# match-cdrv4 (commit 87271ec)

* `match-cdrv4` tag marks the commit at which this code supports replicating the
  behavior of CDRv4 at 25km resolution. This is useful for comparison
  purposes. Future commits will remove support for producing CDRv4 in favor of
  the improvements and bugfixes provided by CDRv5.

# v0.2.0

* Produce 25km CDR instead of 12.5km.
* Refactor how platforms are handled to support overriding platform start dates
  via yaml configuration files.


# v0.1.0

* Initial version of the ECDR.
