# v0.3.0

* Add new `daily` cli that produces daily NC files ready for publication. During
  the AMSR2 period, a `prototype_amsr2` group is added to the final output files
  with AMSR2-derived fields.

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
